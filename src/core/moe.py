import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
import logging
import math


logger = logging.getLogger(__name__)


@torch.no_grad()
def _truncated_normal_(weight: torch.Tensor, fan_in: int, scale: float) -> None:
    std = scale * (1 / fan_in) ** 0.5
    trunc_normal_(weight, mean=0.0, std=std, a=-2 * std, b=2 * std)


class MoE(nn.Module):
    def __init__(
        self,
        dmodel: int,
        dff: int,
        num_experts: int,
        num_experts_per_tok: int,
        capacity_factor: float = 1.25,
        moe_load_balancing_loss_factor: float = 0.0,
        activation_function: str = "swiglu",
        init_scale: float = 1.0,
        **_ignored_kwargs,
    ):
        super().__init__()

        if activation_function != "swiglu":
            raise ValueError(f"MoE supports only swiglu, got {activation_function}.")
        if num_experts_per_tok > num_experts:
            raise ValueError(
                f"num_experts_per_tok={num_experts_per_tok} must be <= num_experts={num_experts}."
            )
        if capacity_factor <= 0:
            raise ValueError(f"capacity_factor must be > 0, got {capacity_factor}.")

        self.dmodel = dmodel
        self.dff = dff
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.capacity_factor = capacity_factor
        self.moe_load_balancing_loss_factor = moe_load_balancing_loss_factor
        self.is_moe = True
        self.aux_loss = None

        self.router_weight = nn.Parameter(torch.empty(num_experts, dmodel))
        self.ff_pre_act_weight = nn.Parameter(torch.empty(num_experts, dff, dmodel))
        self.gate_weight = nn.Parameter(torch.empty(num_experts, dff, dmodel))
        self.ff_post_act_weight = nn.Parameter(torch.empty(num_experts, dmodel, dff))

        _truncated_normal_(self.router_weight, dmodel, init_scale)
        _truncated_normal_(self.ff_pre_act_weight, dmodel, init_scale)
        _truncated_normal_(self.gate_weight, dmodel, init_scale)
        _truncated_normal_(self.ff_post_act_weight, dff, init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        hidden_states = x.reshape(-1, self.dmodel)
        num_tokens = hidden_states.size(0)

        # Router
        router_logits = torch.einsum(
            "th,eh->te",
            hidden_states,
            self.router_weight,
        )
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        # For each token, keep only the top-k experts and their routing probabilities
        topk_probs, selected_experts = torch.topk(
            router_probs,
            k=self.num_experts_per_tok,
            dim=-1,
        )
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(topk_probs.dtype).eps
        )

        # Keep only the highest-gated assignments per expert up to its capacity
        flat_tokens = torch.arange(
            num_tokens, device=hidden_states.device, dtype=torch.long
        ).repeat_interleave(self.num_experts_per_tok)
        flat_experts = selected_experts.reshape(-1)
        flat_weights = topk_probs.reshape(-1)
        total_assignments = flat_experts.numel()
        capacity = max(
            1,
            math.ceil(self.capacity_factor * total_assignments / self.num_experts),
        )
        weight_order = torch.argsort(flat_weights, descending=True, stable=True)
        grouped_order = torch.argsort(flat_experts[weight_order], stable=True)
        sort_order = weight_order[grouped_order]
        sorted_experts = flat_experts[sort_order]
        sorted_tokens = flat_tokens[sort_order]
        sorted_weights = flat_weights[sort_order]
        expert_counts = sorted_experts.bincount(minlength=self.num_experts)
        expert_offsets = expert_counts.cumsum(0) - expert_counts
        slot_in_expert = (
            torch.arange(total_assignments, device=hidden_states.device)
            - expert_offsets[sorted_experts]
        )
        keep = slot_in_expert < capacity
        kept_experts = sorted_experts[keep]
        kept_tokens = sorted_tokens[keep]
        kept_slots = slot_in_expert[keep]
        kept_weights = sorted_weights[keep]
        token_weight_sums = torch.zeros(
            num_tokens,
            dtype=kept_weights.dtype,
            device=hidden_states.device,
        )
        token_weight_sums = token_weight_sums.index_add(
            0,
            kept_tokens,
            kept_weights,
        )
        kept_weights = kept_weights / token_weight_sums[kept_tokens].clamp_min(
            torch.finfo(kept_weights.dtype).eps
        )

        # Dispatch the surviving tokens into expert-capacity slots and run the expert MLP batched per expert
        flat_capacity = self.num_experts * capacity
        dispatch_index = kept_experts * capacity + kept_slots
        expert_inputs = hidden_states.new_zeros(flat_capacity, self.dmodel)
        expert_inputs.index_copy_(0, dispatch_index, hidden_states[kept_tokens])
        expert_inputs = expert_inputs.view(
            self.num_experts,
            capacity,
            self.dmodel,
        )
        ff_pre_act = torch.einsum(
            "ech,edh->ecd",
            expert_inputs,
            self.ff_pre_act_weight,
        )
        gate = torch.einsum(
            "ech,edh->ecd",
            expert_inputs,
            self.gate_weight,
        )
        expert_outputs = torch.einsum(
            "ecd,ehd->ech",
            ff_pre_act * F.silu(gate),
            self.ff_post_act_weight,
        )

        # Gather only the kept expert outputs back to tokens and sum the top-k contributions
        token_updates = expert_outputs.view(flat_capacity, self.dmodel).index_select(
            0, dispatch_index
        )
        token_updates = token_updates * kept_weights.to(hidden_states.dtype).unsqueeze(
            -1
        )
        output = hidden_states.new_zeros(num_tokens, self.dmodel)
        output = output.index_add(0, kept_tokens, token_updates)
        output = output.reshape(original_shape)

        # Match the switch-style load-balancing term using pre-capacity routing statistics
        if self.training:
            expert_frequency = flat_experts.bincount(minlength=self.num_experts)
            expert_frequency = expert_frequency.to(router_probs.dtype)
            expert_frequency = expert_frequency / expert_frequency.sum().clamp_min(1)
            self.aux_loss = (
                self.num_experts * (router_probs.mean(dim=0) * expert_frequency).sum()
            )
        else:
            self.aux_loss = None

        return output
