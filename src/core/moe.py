"""Token-choice top-k mixture-of-experts feed-forward, with a capacity factor.

A port of src/core/moe.py on main (removed here in 3001bb62). Changes from it:
- static shapes, so compile_blocks_full traces a block whole: an assignment over
  capacity goes to a spare slot instead of being filtered out with a boolean mask;
- experts are GatedFeedForward modules built from the dense ff_layer_fn partials, so
  every expert weight is an ordinary 2D parameter and the config merges onto the
  dense node (configs/_ff_layer/moe.yaml);
- eval routes without dropping (capacity = token count), so right padding in the
  downstream eval can't push real tokens out.

Follow-up, not built: grouped-GEMM kernels in place of the loop over experts.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.model import GatedFeedForward, llm_random_weight_init


def load_balancing_loss(router_probs, expert_ids, num_experts):
    """Switch-style balance term on pre-capacity routing; 1 under uniform routing."""
    one_hot = expert_ids.reshape(-1, 1) == torch.arange(
        num_experts, device=expert_ids.device
    )
    assignment_share = one_hot.float().mean(0)
    return num_experts * (router_probs.mean(0) * assignment_share).sum()


class MoE(nn.Module):
    # what forward() stores in self.stats, logged by the trainer as train/moe_<name>
    STAT_NAMES = ("balance_loss", "z_loss", "dropped_fraction")

    def __init__(
        self,
        ff_pre_act_fn,
        ff_post_act_fn,
        gate_fn,
        num_experts: int,
        topk: int,
        capacity_factor: float,
        load_balancing_loss_factor: float,
        router_z_loss_factor: float,
    ):
        super().__init__()
        if not 0 < topk <= num_experts:
            raise ValueError(f"topk={topk} must be in [1, num_experts={num_experts}].")
        if capacity_factor <= 0:
            raise ValueError(f"capacity_factor must be > 0, got {capacity_factor}.")
        self.num_experts = num_experts
        self.topk = topk
        self.capacity_factor = capacity_factor
        self.load_balancing_loss_factor = load_balancing_loss_factor
        self.router_z_loss_factor = router_z_loss_factor

        self.experts = nn.ModuleList(
            GatedFeedForward(ff_pre_act_fn, ff_post_act_fn, gate_fn)
            for _ in range(num_experts)
        )
        dmodel = self.experts[0].ff_pre_act.in_features
        self.router_weight = nn.Parameter(torch.empty(num_experts, dmodel))
        llm_random_weight_init(dmodel, scale=1)(self.router_weight)

    def capacity(self, n_tokens):
        if not self.training:
            return n_tokens  # no expert gets a token twice, so nothing is dropped
        slots = math.ceil(
            self.capacity_factor * n_tokens * self.topk / self.num_experts
        )
        return min(n_tokens, slots)

    def slot_in_expert(self, expert_ids, weights):
        """Each assignment's place in its expert's queue, higher router weight first."""
        order = weights.argsort(descending=True, stable=True)
        one_hot = expert_ids[order].unsqueeze(1) == torch.arange(
            self.num_experts, device=expert_ids.device
        )
        sorted_slots = (one_hot.long().cumsum(0) * one_hot).sum(1) - 1
        return torch.empty_like(sorted_slots).scatter_(0, order, sorted_slots)

    def forward(self, x):
        tokens = x.reshape(-1, x.shape[-1])
        n_tokens, dmodel = tokens.shape
        logits = F.linear(tokens, self.router_weight).float()
        probs = logits.softmax(dim=-1)
        weights, expert_ids = probs.topk(self.topk, dim=-1)
        expert_ids = expert_ids.reshape(
            -1
        )  # token-major: token t owns rows t*k..t*k+k-1

        capacity = self.capacity(n_tokens)
        slots = self.slot_in_expert(expert_ids, weights.reshape(-1))
        kept = slots < capacity
        # over-capacity assignments go to a spare row past the last expert's slots
        spare = self.num_experts * capacity
        dispatch = torch.where(kept, expert_ids * capacity + slots, spare)

        expert_inputs = tokens.new_zeros(spare + 1, dmodel).index_add(
            0, dispatch, tokens.repeat_interleave(self.topk, dim=0)
        )
        expert_outputs = [
            expert(expert_input)
            for expert, expert_input in zip(
                self.experts, expert_inputs[:-1].view(self.num_experts, capacity, -1)
            )
        ]
        # dropped assignments read the zero spare row
        expert_outputs = torch.cat(expert_outputs + [tokens.new_zeros(1, dmodel)])
        # not expert_outputs[dispatch]: its bf16 backward serializes the spare row's duplicates
        per_assignment = expert_outputs.index_select(0, dispatch)
        per_assignment = per_assignment.view(n_tokens, self.topk, dmodel)
        output = (per_assignment * weights.unsqueeze(-1).to(tokens.dtype)).sum(1)

        if self.training:
            balance = load_balancing_loss(probs, expert_ids, self.num_experts)
            z_loss = torch.logsumexp(logits, dim=-1).square().mean()
            self.aux_loss = (
                self.load_balancing_loss_factor * balance
                + self.router_z_loss_factor * z_loss
            )
            self.stats = torch.stack(
                [balance, z_loss, 1 - kept.float().mean()]
            ).detach()
        return output.reshape(x.shape)
