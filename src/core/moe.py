"""Dropless token-choice top-k mixture-of-experts feed-forward, on grouped matmuls.

Each of the T*k assignments gets its own row in a buffer grouped by expert, and each
expert projection is one grouped matmul over all experts (F.grouped_mm). Nothing is
dropped, so there is no capacity factor, train and eval route the same way, and right
padding in the downstream eval can't touch real tokens. The buffer has exactly T*k rows,
so compile_blocks_full still traces a block whole; only the group boundaries depend on
the routing.

Expert weights are 3D parameters stacked from the dense ff_layer_fn partials, so their
init matches the dense layer's: gate_up_weight (num_experts, 2, dff, dmodel) holds gate
and up, which one grouped matmul computes together (torchtitan's w13), and
post_act_weight is (num_experts, dmodel, dff). Muon has to treat them as batches of
matrices, gate and up separately. FSDP2 shards them along the expert dimension.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.model import GatedFeedForward, llm_random_weight_init


def load_balancing_loss(router_probs, expert_ids, num_experts):
    """Switch-style balance term on the routing; 1 under uniform routing."""
    one_hot = expert_ids.reshape(-1, 1) == torch.arange(
        num_experts, device=expert_ids.device
    )
    assignment_share = one_hot.float().mean(0)
    return num_experts * (router_probs.mean(0) * assignment_share).sum()


def _bf16_cuda(*tensors):
    return all(t.is_cuda and t.dtype == torch.bfloat16 for t in tensors)


class RouterLogits(torch.autograd.Function):
    """x @ weight.T with an fp32 output, saving only x and weight as they are.

    bf16 on CUDA (torchtitan's HiMidLoLinear): a bf16 tensor-core matmul that accumulates
    and returns fp32, and a backward that splits the fp32 grad into two bf16 pieces
    (hi + mid) for near-fp32 grads. Anything else (CPU tests): the same in fp32.
    """

    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        if _bf16_cuda(x, weight):
            return torch.mm(x, weight.T, out_dtype=torch.float32)
        return x.float() @ weight.float().T

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        if _bf16_cuda(x, weight):
            hi = grad.bfloat16()
            mid = (grad - hi.float()).bfloat16()
            mm = lambda a, b: torch.mm(a, b, out_dtype=torch.float32)
            grad_x = mm(hi, weight) + mm(mid, weight)
            grad_weight = mm(hi.T, x) + mm(mid.T, x)
        else:
            grad_x, grad_weight = grad @ weight.float(), grad.T @ x.float()
        return grad_x.to(x.dtype), grad_weight.to(weight.dtype)


class DispatchRows(torch.autograd.Function):
    """x[token_of_row]: every token's k assignments in expert order, one gather.

    The backward gathers each token's k rows by dest (its rows, token-major) and sums
    them, so neither direction zero-fills a buffer or adds atomically.
    """

    @staticmethod
    def forward(ctx, x, token_of_row, dest, topk):
        ctx.save_for_backward(dest)
        ctx.topk = topk
        return x.index_select(0, token_of_row)

    @staticmethod
    def backward(ctx, grad):
        (dest,) = ctx.saved_tensors
        per_token = grad.index_select(0, dest).unflatten(0, (-1, ctx.topk))
        return per_token.sum(1), None, None, None


class TakeUniqueRows(torch.autograd.Function):
    """x[index] for distinct row indices; the backward scatters instead of atomically adding."""

    @staticmethod
    def forward(ctx, x, index):
        ctx.save_for_backward(index)
        ctx.rows = x.shape[0]
        return x.index_select(0, index)

    @staticmethod
    def backward(ctx, grad):
        (index,) = ctx.saved_tensors
        return grad.new_zeros(ctx.rows, grad.shape[-1]).index_copy(0, index, grad), None


def grouped_linear(x, weight, ends):
    """x @ weight[e].T for each expert's row group, weight (num_experts, out, in)."""
    if x.is_cuda:  # the CUDA kernels take bf16 only
        return F.grouped_mm(x.bfloat16(), weight.bfloat16().mT, offs=ends).type_as(x)
    return F.grouped_mm(x, weight.mT, offs=ends)


class MoE(nn.Module):
    # what forward() stores in self.stats, logged by the trainer as train/moe_<name>
    STAT_NAMES = ("balance_loss", "z_loss", "max_load", "bias_range")

    def __init__(
        self,
        ff_pre_act_fn,
        ff_post_act_fn,
        gate_fn,
        num_experts: int,
        topk: int,
        load_balancing_loss_factor: float,
        router_z_loss_factor: float,
        expert_bias_update_rate: float,
    ):
        super().__init__()
        if not 0 < topk <= num_experts:
            raise ValueError(f"topk={topk} must be in [1, num_experts={num_experts}].")
        self.num_experts = num_experts
        self.topk = topk
        self.load_balancing_loss_factor = load_balancing_loss_factor
        self.router_z_loss_factor = router_z_loss_factor
        # DeepSeek-V3's auxiliary-loss-free balancing (arXiv 2408.15664); 0 turns it off
        self.expert_bias_update_rate = expert_bias_update_rate
        self.register_buffer("expert_bias", torch.zeros(num_experts))
        # assignments per expert since the last bias update, summed over micro-batches
        self.register_buffer("expert_load", torch.zeros(num_experts), persistent=False)

        # built in the order the per-expert ModuleList used, so the init is unchanged
        experts = [
            GatedFeedForward(ff_pre_act_fn, ff_post_act_fn, gate_fn)
            for _ in range(num_experts)
        ]
        self.gate_up_weight = nn.Parameter(
            torch.stack(
                [torch.stack([e.gate.weight, e.ff_pre_act.weight]) for e in experts]
            ).detach()
        )
        self.post_act_weight = nn.Parameter(
            torch.stack([e.ff_post_act.weight for e in experts]).detach()
        )
        dmodel = self.gate_up_weight.shape[-1]
        self.router_weight = nn.Parameter(torch.empty(num_experts, dmodel))
        llm_random_weight_init(dmodel, scale=1)(self.router_weight)

    def expert_weights(self):
        return (self.gate_up_weight, self.post_act_weight)

    @torch.no_grad()
    def update_expert_bias(self, load):
        """Bias step from this step's per-expert load: overloaded experts down, the rest up."""
        delta = self.expert_bias_update_rate * torch.sign(load.mean() - load)
        # centred, as torchtitan does: a shared shift changes no top-k choice
        self.expert_bias.add_(delta - delta.mean())
        self.expert_load.zero_()

    def group_by_expert(self, expert_ids):
        """Each assignment's row in a buffer where experts' groups are contiguous.

        Returns (ends, dest, token_of_row, load): each group's end (int32), each
        assignment's unique row, each row's token, and the assignments per expert.
        """
        one_hot = expert_ids.unsqueeze(1) == torch.arange(
            self.num_experts, device=expert_ids.device
        )
        load = one_hot.sum(0)
        place = (one_hot.long().cumsum(0) * one_hot).sum(1) - 1
        ends = load.cumsum(0)
        dest = (ends - load)[expert_ids] + place
        assignments = torch.arange(dest.numel(), device=dest.device)
        token_of_row = torch.empty_like(dest).scatter_(
            0, dest, assignments // self.topk
        )
        return ends.int(), dest, token_of_row, load

    def forward(self, x):
        tokens = x.reshape(-1, x.shape[-1])
        n_tokens, dmodel = tokens.shape
        logits = RouterLogits.apply(tokens, self.router_weight)
        probs = logits.softmax(dim=-1)
        # the bias only picks the experts; their weights come from the unbiased probs
        expert_ids = (probs + self.expert_bias).topk(self.topk, dim=-1).indices
        weights = probs.gather(-1, expert_ids)
        # token-major: token t owns assignments t*k..t*k+k-1
        expert_ids = expert_ids.reshape(-1)

        ends, dest, token_of_row, load = self.group_by_expert(expert_ids)
        grouped = DispatchRows.apply(tokens, token_of_row, dest, self.topk)
        gate_up = grouped_linear(grouped, self.gate_up_weight.flatten(1, 2), ends)
        gate, up = gate_up.unflatten(-1, (2, -1)).unbind(-2)
        hidden = F.silu(gate) * up
        expert_outputs = grouped_linear(hidden, self.post_act_weight, ends)
        # dest is distinct per assignment, so no atomics in the backward (index_select's has them)
        per_assignment = TakeUniqueRows.apply(expert_outputs, dest)
        per_assignment = per_assignment.view(n_tokens, self.topk, dmodel)
        output = (per_assignment * weights.unsqueeze(-1).to(tokens.dtype)).sum(1)

        if self.training:
            balance = load_balancing_loss(probs, expert_ids, self.num_experts)
            z_loss = torch.logsumexp(logits, dim=-1).square().mean()
            self.aux_loss = (
                self.load_balancing_loss_factor * balance
                + self.router_z_loss_factor * z_loss
            )
            # busiest expert's assignments over the mean (1 = balanced)
            max_load = load.max() * self.num_experts / expert_ids.numel()
            bias_range = self.expert_bias.max() - self.expert_bias.min()
            self.stats = torch.stack(
                [balance, z_loss, max_load.float(), bias_range]
            ).detach()
            if self.expert_bias_update_rate > 0:
                self.expert_load.add_(load.float())
        return output.reshape(x.shape)
