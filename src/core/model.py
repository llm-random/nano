from functools import partial
import math
import os
import torch
import torch.nn as nn
from typing import Callable, Optional
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.attention import SDPBackend
from torch.nn.init import trunc_normal_ as trunc_normal_
from torch.utils.checkpoint import checkpoint
from torch.nn import (
    LayerNorm as LayerNorm,
)  # used by FSDP, but it keeps getting removed during file formatting

from torch.nn.modules.normalization import RMSNorm as RMSNorm


def llm_random_weight_init(fan_in, scale):
    std = scale * (1 / fan_in) ** 0.5
    low = -2 * std
    high = 2 * std
    return partial(trunc_normal_, mean=0.0, std=std, a=low, b=high)


class TransformerEmbedding(nn.Embedding):
    def __init__(
        self, vocab_size: int, dmodel: int, init_fn: Optional[Callable], **kwargs
    ):
        self.init_fn = init_fn
        super().__init__(vocab_size, dmodel, **kwargs)

    def reset_parameters(self):
        if hasattr(self, "init_fn") and self.init_fn is not None:
            with torch.no_grad():
                self.init_fn(self.weight)
        else:
            super().reset_parameters()


class Linear(nn.Linear):
    def __init__(self, *args, partial_init_fn, **kwargs):
        if "bias" not in kwargs:
            kwargs["bias"] = False
        super().__init__(*args, **kwargs)
        init_fn = partial_init_fn(self.in_features)
        init_fn(self.weight)


class Residual(nn.Module):
    def __init__(self, norm, layer, log_name):
        super(Residual, self).__init__()
        self.norm = norm
        self.layer = layer
        self.metric_logger = None
        self.log_name = log_name
        # Most layers are a function of the residual stream alone. A layer whose
        # scores are indexed by token id (src.core.ame) needs the ids too, and
        # says so by declaring `wants_token_ids`. Resolved once here, at
        # construction, so the branch in forward is a Python constant dynamo
        # folds away rather than a per-step attribute lookup.
        self.wants_token_ids = getattr(layer, "wants_token_ids", False)

    def set_metric_logger(self, metric_logger):
        self.metric_logger = metric_logger

    def forward(self, x, token_ids=None):
        normalized = self.norm(x)
        if self.wants_token_ids:
            out = self.layer(normalized, token_ids)
        else:
            out = self.layer(normalized)
        if self.metric_logger is not None:
            self._accumulate_intermediate_metrics(x, out)
        return out + x

    @torch.compiler.disable
    def _accumulate_intermediate_metrics(self, x, out):
        # Kept out of the traced path on purpose: dynamo would otherwise guard
        # the compiled graph on the per-block log_name string, forcing one
        # recompile per block until the recompile_limit silently drops blocks
        # back to eager (the original block-level-compile failure). Inside a
        # compiler-disabled function the string is never traced; with the
        # logger unset, the branch in forward is constant-folded away.
        self.metric_logger.accumulate_metrics(
            layer_name=f"{self.log_name}",
            transform_fn=Residual.intermediate_norms,
            calculate_fn=Residual.calculate_metrics,
            metrics={
                "residual_stream": x,
                "updates": out,
            },
        )

    @staticmethod
    def intermediate_norms(residual_stream: torch.Tensor, updates: torch.Tensor):

        with torch.no_grad():
            update_norms = torch.norm(updates, dim=-1)
            residual_norms = torch.norm(residual_stream, dim=-1)

            return {
                "update_norms_list": update_norms,
                "residual_norms_list": residual_norms,
            }

    @staticmethod
    def calculate_metrics(
        update_norms_list: torch.Tensor, residual_norms_list: torch.Tensor
    ):
        update_norms_concat = torch.cat(update_norms_list)
        residual_norms_concat = torch.cat(residual_norms_list)

        if dist.is_initialized():
            world_size = int(os.environ["WORLD_SIZE"])
            gpu_batch_size, seq_len = residual_norms_concat.shape
            update_norms = torch.empty(
                world_size * gpu_batch_size,
                seq_len,
                device=update_norms_concat.device,
                dtype=update_norms_concat.dtype,
            )
            dist.all_gather_into_tensor(update_norms, update_norms_concat)

            residual_norms = torch.empty(
                world_size * gpu_batch_size,
                seq_len,
                device=residual_norms_concat.device,
                dtype=residual_norms_concat.dtype,
            )
            dist.all_gather_into_tensor(residual_norms, residual_norms_concat)
        else:
            update_norms = update_norms_concat
            residual_norms = residual_norms_concat

        with torch.no_grad():
            update_norms_std, update_norms_mean = torch.std_mean(update_norms)
            residual_norms_std, residual_norms_mean = torch.std_mean(residual_norms)

            update_to_residual_ratio = update_norms / residual_norms
            ratio_std, ratio_mean = torch.std_mean(update_to_residual_ratio)

            return {
                "update_norms/mean": update_norms_mean.item(),
                "update_norms/std": update_norms_std.item(),
                "residual_norms/mean": residual_norms_mean.item(),
                "residual_norms/std": residual_norms_std.item(),
                "update_to_residual_ratio/mean": ratio_mean.item(),
                "update_to_residual_ratio/std": ratio_std.item(),
            }


class RoPE(nn.Module):
    """
    This code is mostly taken from HF for Llama #TODO (add url)
    Standard setup for models:
    - Llama base=500000, scale_freqs=True
    - llm-random base=10000, scale_freqs=False
    """

    # features are paired x_i, x_{i + d_head/2}
    def __init__(
        self,
        dhead,
        length,
        base,
        apply_freq_scaling,
        factor,
        low_freq_factor,
        high_freq_factor,
        original_max_position_embeddings,
    ):
        super().__init__()
        self.dhead = dhead
        self.length = length
        self.base = base
        self.apply_freq_scaling = apply_freq_scaling
        self.factor = factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.register_freqs()

    def register_freqs(self):
        angle_exponents = (
            torch.arange(0, self.dhead, 2, dtype=torch.int64).float() / self.dhead
        )
        angles = 1.0 / torch.pow(self.base, angle_exponents).reshape(1, -1)
        if self.apply_freq_scaling:
            angles = self.scale_freqs(angles)

        angle_per_token = angles * torch.arange(0, self.length).reshape(-1, 1)
        self.register_buffer(
            "sin", torch.sin(angle_per_token).repeat(1, 2), persistent=False
        )
        self.register_buffer(
            "cos", torch.cos(angle_per_token).repeat(1, 2), persistent=False
        )

    def scale_freqs(self, freqs):
        factor = self.factor
        low_freq_factor = self.low_freq_factor
        high_freq_factor = self.high_freq_factor
        old_context_len = self.original_max_position_embeddings

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        wavelen = 2 * math.pi / freqs
        # wavelen < high_freq_wavelen: do nothing
        # wavelen > low_freq_wavelen: divide by factor
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, freqs / factor, freqs)
        # otherwise: interpolate between the two, using a smooth factor
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smoothed_inv_freq = (
            1 - smooth_factor
        ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        return inv_freq_llama

    def forward(self, x):
        [y1, y2] = torch.chunk(x, chunks=2, dim=-1)
        x_rotated = torch.cat([-y2, y1], dim=-1)
        cos_scaler = self.cos[: x.shape[-2], :].to(x.device, dtype=x.dtype)
        sin_scaler = self.sin[: x.shape[-2], :].to(x.device, dtype=x.dtype)
        return x * cos_scaler + x_rotated * sin_scaler


class RoPEAttention(nn.Module):
    # QK-norm's temperature and the KV-shift coefficients are scales, not
    # weights: decoupled decay would move them whatever their gradient says --
    # shrink tau, flattening every head, and walk alpha_2/beta_2 toward 0, the
    # "model declined the shift" signal -- the reasons src.core.ame keeps its
    # own out of decay too. parameter_groups skips a name the module does not
    # have, so with both options off this declares nothing and the optimizer
    # stays one group.
    no_weight_decay_parameters = ("tau", "alpha", "beta")
    # [kv_heads, 2] is 2-D but no matmul: src.core.mfu would otherwise charge
    # 6*P FLOPs for them. Skipped when absent, like the line above.
    non_matmul_parameters = ("alpha", "beta")

    def __init__(
        self,
        q_proj_fn,
        k_proj_fn,
        v_proj_fn,
        o_proj_fn,
        dmodel,
        q_heads,
        kv_heads,
        seq_len,
        rope_base,
        rope_scale_freqs: bool,
        factor=32,
        low_freq_factor=1,
        high_freq_factor=4,
        original_max_position_embeddings=8192,
        rope: bool = True,
        qk_norm: bool = False,
        shift: bool = False,
    ):
        super().__init__()
        self.q_proj = q_proj_fn()
        self.k_proj = k_proj_fn()
        self.v_proj = v_proj_fn()
        self.o_proj = o_proj_fn()
        self.attention_mechanism = AttentionMechanism()

        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.dhead = self.q_proj.weight.shape[0] // self.q_heads
        self.dmodel = dmodel

        # rope=False is NoPE: no positional encoding at all, so the only positional
        # signal left is the causal mask (Haviv et al. 2022). RoPE holds no
        # parameters and its buffers are not persistent, so the state dict is the
        # same either way.
        self.rope = (
            RoPE(
                dhead=self.dhead,
                length=seq_len,
                base=rope_base,
                apply_freq_scaling=rope_scale_freqs,
                factor=factor,
                low_freq_factor=low_freq_factor,
                high_freq_factor=high_freq_factor,
                original_max_position_embeddings=original_max_position_embeddings,
            )
            if rope
            else None
        )

        # QK-norm, the operator src.core.ame uses: q and k l2-normalized per
        # head, and the score scaled by a learned per-head temperature tau in
        # place of 1/sqrt(dhead). tau starts at sqrt(dhead): two random unit
        # vectors in d dims have dot-product std 1/sqrt(d), so the scores start
        # at std ~1. Off adds no parameter -- the state dict, the FLOP count and
        # every existing run stay as they were.
        self.qk_norm = qk_norm
        if qk_norm:
            self.tau = nn.Parameter(
                torch.full((q_heads,), math.sqrt(self.dhead), dtype=torch.float32)
            )

        # KV shifting (Xu et al. 2024, arXiv 2411.19574), the operator
        # src.core.ame applies to its banks, here on the projected keys and
        # values: k_j <- a1*k_j + a2*k_{j-1} and v_j <- b1*v_j + b2*v_{j-1},
        # BEFORE the rotary embedding as in the paper, so a mixed key is rotated
        # as position j. Queries are not shifted. One (c1, c2) pair per kv head,
        # c1 ~ U(0, 1) and c2 = 1 - c1 -- AME's _init_mix -- so the plain head
        # (1, 0) and the fully shifted one (0, 1) are both reachable. Drawn after
        # the projections, so a seed builds the same projections either way.
        # Off adds no parameter and draws nothing.
        self.shift = shift
        if shift:
            for name in ("alpha", "beta"):
                c1 = torch.rand(kv_heads, 1, dtype=torch.float32)
                setattr(self, name, nn.Parameter(torch.cat([c1, 1.0 - c1], dim=-1)))

    def forward(self, x):
        query_states = self.q_proj(x)
        key_states = self.k_proj(x)
        value_states = self.v_proj(x)

        # [B, n, heads, D] for the per-head work, [B, heads, n, D] from RoPE on.
        batch, seq_len = x.shape[:-1]
        q = query_states.view(batch, seq_len, self.q_heads, -1)
        k = key_states.view(batch, seq_len, self.kv_heads, -1)
        v = value_states.view(batch, seq_len, self.kv_heads, -1)
        if self.qk_norm:
            # Per head and before the shift, as src.core.ame normalizes its rows
            # before the mix; before RoPE, a rotation, so that order changes
            # nothing; on the kv heads, before repeat_kv copies them.
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        if self.shift:
            # AME's two lines (src/core/ame.py _mixed_qkv), on kv heads.
            # Imported lazily: src.core.ame imports this module.
            from src.core.ame import _shift1

            alpha = self.alpha.to(k.dtype)
            beta = self.beta.to(v.dtype)
            a1, a2 = alpha[:, 0].view(1, 1, -1, 1), alpha[:, 1].view(1, 1, -1, 1)
            b1, b2 = beta[:, 0].view(1, 1, -1, 1), beta[:, 1].view(1, 1, -1, 1)
            k = a1 * k + a2 * _shift1(k)
            v = b1 * v + b2 * _shift1(v)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)
        scale = None  # SDPA's default, 1/sqrt(dhead)
        if self.qk_norm:
            # tau folded into q, so the kernel runs at scale 1 -- as in
            # src.core.ame. q is [B, H, n, D] here.
            q = q * self.tau.to(q.dtype).view(1, -1, 1, 1)
            scale = 1.0

        # Imported lazily: src.core.llama imports AttentionMechanism from this
        # module, so a top-level import here would be circular.
        from src.core.llama import repeat_kv

        k = repeat_kv(k, self.q_heads // self.kv_heads)
        v = repeat_kv(v, self.q_heads // self.kv_heads)
        attention_output = self.attention_mechanism(
            query=q, key=k, value=v, causal=True, scale=scale
        )

        output = self.o_proj(attention_output.transpose(1, 2).contiguous().flatten(-2))

        return output

    def attention_score_flops_per_token(self, seq_len):
        """QK^T and AV at 2*s*datt each per token forward, tripled for the
        backward: the PaLM convention, no causal discount. src.core.mfu reads
        this off every attention module, so each type states its own cost."""
        return 12 * seq_len * self.q_heads * self.dhead


class GatedFeedForward(nn.Module):
    def __init__(self, ff_pre_act_fn, ff_post_act_fn, gate_fn):
        super().__init__()
        self.silu = nn.SiLU()
        self.ff_pre_act = ff_pre_act_fn()
        self.ff_post_act = ff_post_act_fn()
        self.gate = gate_fn()

    def forward(self, x):
        gated = self.gate(x)
        gated = self.silu(gated)
        x = self.ff_pre_act(x)
        x = x * gated
        x = self.ff_post_act(x)
        return x


class GeluFeedForward(nn.Module):
    """nanoGPT-style 2-matrix MLP: up-projection, GELU, down-projection.

    The architecture-comparison counterpart of GatedFeedForward (same dff,
    one fewer matrix, no gating) -- used by the nanoGPT-arch ablation to run
    GPT-2's FF inside this framework (documents/notes/mfu_ablation_plan.md).
    """

    def __init__(self, ff_pre_act_fn, ff_post_act_fn):
        super().__init__()
        self.gelu = nn.GELU()
        self.ff_pre_act = ff_pre_act_fn()
        self.ff_post_act = ff_post_act_fn()

    def forward(self, x):
        x = self.ff_pre_act(x)
        x = self.gelu(x)
        x = self.ff_post_act(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        block_id,
        norm_fn,
        attention_fn,
        ff_layer_fn,
    ):
        super().__init__()
        self.log_name = f"block[{block_id}]"

        # attention_fn takes the block id so the attention type can vary per
        # layer (see src.core.attention.build_attention and the mix patterns).
        self.attention_layer = Residual(
            norm=norm_fn(),
            layer=attention_fn(block_id),
            log_name=f"{self.log_name}/residual_attention",
        )
        self.ff_layer = Residual(
            norm=norm_fn(),
            layer=ff_layer_fn(),
            log_name=f"{self.log_name}/residual_feedforward",
        )

    def forward(self, x, token_ids=None):
        x = self.attention_layer(x, token_ids)
        x = self.ff_layer(x)
        return x


class TransformerEncoder(nn.Module):
    def get_model_dimensions(self):
        # Works only for llama3 transforermer architecture
        dmodel = self.blocks[0].ff_layer.layer.ff_pre_act.weight.shape[1]
        dff = self.blocks[0].ff_layer.layer.ff_pre_act.weight.shape[0]
        datt = self.blocks[0].attention_layer.layer.q_proj.weight.shape[0]
        n_att_heads = self.blocks[0].attention_layer.layer.q_heads
        n_kvatt_heads = self.blocks[0].attention_layer.layer.kv_heads
        nlayers = len(self.blocks)

        head_dim = datt / n_att_heads

        return dmodel, dff, n_att_heads, n_kvatt_heads, head_dim, nlayers

    def __init__(
        self,
        block_fn: Callable[[int], nn.Module],
        n_blocks: int,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([block_fn(i) for i in range(n_blocks)])

    def forward(self, x, token_ids=None):
        # token_ids travels beside the activations because an id-indexed
        # attention layer needs them at every depth, not only at the embedding.
        # Layers that do not want them never see them (see Residual).
        for block in self.blocks:
            x = block(x, token_ids)
        return x


class _ChunkedLinearCE(torch.autograd.Function):
    """Cross-entropy over a projection, computed in chunks, gradients and all.

    The first O11 implementation chunked with `torch.utils.checkpoint`, which
    is correct by construction and **measurably too slow**: it reruns each
    chunk's projection AND its softmax in backward. Measured at h16/1 GPU
    (jobs 32825 vs 32862) that cost 447 ms/step, 2.65 MFU points -- more than
    the larger micro-batch it unlocks was worth.

    The recompute is avoidable. The loss is the final scalar, so this operation
    is the end of the graph and every gradient below it is linear in the
    incoming `grad_output`. That means the gradients can be computed in the
    *forward* pass with `grad_output = 1` and scaled afterwards, which is what
    this does: each chunk's logits are turned into `dlogits` in place, then
    contracted straight into `dx` and `dweight` and dropped. One pass, one
    chunk of logits live at a time, no recompute.

    What is saved for backward is `dx` ([N, d], the size of the input) and
    `dweight` ([V, d], the size of the weight) -- not the [N, V] logits, which
    is the whole point.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, targets, chunk_size, ignore_index):
        rows = x.shape[0]
        valid = (targets != ignore_index).sum().clamp(min=1)

        # fp32 or better, never worse -- `.float()` would silently DOWNCAST a
        # float64 model, which is exactly the precision the section 1.4 gate
        # runs at, and the gate would then be measuring float32 reassociation
        # instead of this function. Caught by that gate rather than in review.
        acc = torch.promote_types(x.dtype, torch.float32)

        loss = torch.zeros((), device=x.device, dtype=acc)
        grad_x = torch.zeros_like(x, dtype=acc)
        grad_w = torch.zeros_like(weight, dtype=acc)
        grad_b = None if bias is None else torch.zeros_like(bias, dtype=acc)

        for start in range(0, rows, chunk_size):
            stop = min(start + chunk_size, rows)
            hidden = x[start:stop]
            target = targets[start:stop]

            logits = F.linear(hidden, weight, bias).to(acc)
            keep = (target != ignore_index).to(acc)
            safe = target.clamp(min=0)

            # log-softmax in fp32, as F.cross_entropy does internally whatever
            # dtype it is handed -- the vocabulary is 50304 wide and the
            # log-sum-exp is where bf16 would actually hurt.
            lse = torch.logsumexp(logits, dim=-1)
            picked = logits.gather(1, safe[:, None]).squeeze(1)
            loss = loss + ((lse - picked) * keep).sum()

            # d(sum loss)/d(logits) = softmax - onehot, zeroed on ignored rows,
            # then divided by the valid count so it matches reduction="mean".
            #
            # In place, over the logits buffer: a [chunk, 50304] fp32 tensor is
            # 824 MB at chunk 4096, and holding the softmax beside the logits
            # would put two of them live at once -- which is the allocation
            # this whole function exists to avoid. `loss`, `lse` and `picked`
            # are already read out above, so nothing below needs the logits.
            probabilities = logits.sub_(lse[:, None]).exp_()
            probabilities.scatter_add_(
                1, safe[:, None], torch.full_like(picked[:, None], -1.0)
            )
            probabilities *= (keep / valid)[:, None]

            # The two contractions run in the MODEL's dtype, not in `acc`.
            #
            # This is the whole performance story of this function. `acc` is
            # fp32 for a bf16 model, and an fp32 matmul on an H100 runs at
            # ~60 TFLOP/s against ~989 for bf16 -- a 16x penalty on the largest
            # matmul in the step. Measured: casting these two to `acc` put h16
            # at 20.85% MFU against 37.87% for the unfused path (job 32870).
            # fp32 is needed for the log-sum-exp above, where 50304 terms are
            # summed; it is not needed here, where cuBLAS already accumulates
            # a bf16 product in fp32 internally.
            gradient = probabilities.to(x.dtype)
            grad_x[start:stop] = (gradient @ weight).to(acc)
            grad_w += (gradient.t() @ hidden).to(acc)
            if grad_b is not None:
                grad_b += probabilities.sum(0)

        ctx.save_for_backward(grad_x.to(x.dtype), grad_w.to(weight.dtype), grad_b)
        ctx.has_bias = bias is not None
        return loss / valid.to(loss.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x, grad_w, grad_b = ctx.saved_tensors
        # Linear in grad_output, which is why computing at 1 and scaling here
        # is exact rather than an approximation.
        return (
            grad_x * grad_output,
            grad_w * grad_output,
            (grad_b.to(grad_w.dtype) * grad_output) if ctx.has_bias else None,
            None,
            None,
            None,
        )


class TransformerHead(nn.Module):
    def __init__(self, linear_fn: Callable, norm_fn: Callable):
        super().__init__()
        self.norm = norm_fn()
        self.linear = linear_fn()

    def forward(self, x, targets=None, chunk_size=None, ignore_index=-100):
        """Logits, or -- given `targets` -- the loss, without ever building the
        logits (O11, fused-linear-CE).

        `targets=None` is the original path and is bit-for-bit what it was.

        With `targets`, the projection and the cross-entropy are computed in
        chunks of rows and each chunk's logits are dropped as soon as its loss
        is accumulated. The full `[micro_batch, seq, vocab]` tensor is the
        largest activation in the step -- 6.14 GiB at h16/micro-batch 32, which
        is the allocation job 32853 died on -- and it never exists here.

        **Why `targets` is an argument rather than a separate method.** Under
        FSDP2 this module's parameters are sharded, and `fully_shard` installs
        a pre-forward hook that all-gathers them. The hook fires on `__call__`.
        A `head.compute_loss(...)` method would bypass it and read a sharded
        DTensor as if it were the whole weight -- not an error FSDP2 reports,
        just a wrong result.

        The chunking uses `torch.utils.checkpoint` per chunk rather than a
        hand-written `autograd.Function`. The gradient is then correct by
        construction instead of by my arithmetic, which matters for a path
        whose failure mode is a silently wrong loss. The cost is that each
        chunk's projection runs twice, once in forward and once in backward --
        the head matmul only, not the model. That is O11's actual price and it
        is why O11 is expected to pay in *memory*, not in step time.

        Returns the mean over non-ignored targets, matching
        `F.cross_entropy(..., reduction="mean")`.
        """
        x = self.norm(x)
        if targets is None:
            return self.linear(x)

        flat = x.reshape(-1, x.shape[-1])
        flat_targets = targets.reshape(-1)
        rows = flat.shape[0]
        if chunk_size is None or chunk_size <= 0:
            chunk_size = rows
        linear = self.linear

        def chunk_loss(hidden, target):
            # reduction="sum": the chunks carry different numbers of valid
            # targets, so a mean per chunk could not be averaged back.
            return F.cross_entropy(
                linear(hidden), target, reduction="sum", ignore_index=ignore_index
            )

        # `torch.utils.checkpoint` per chunk, NOT the recompute-free
        # `_ChunkedLinearCE` below it. That ordering is a measurement, not a
        # preference, and it is the opposite of what the literature suggests:
        #
        #     no O11                                    37.96%  (job 32825)
        #     v1, checkpoint per chunk  (THIS)          35.31%  (job 32862)
        #     v3, gradients computed in forward         30.38%  (job 32876)
        #
        # v3 avoids the recompute entirely -- valid because the loss is the
        # final scalar, so every gradient below it is linear in `grad_output`
        # and can be computed at 1 and scaled afterwards. It is still slower,
        # because a hand-rolled logsumexp + gather + scatter_add + exp over a
        # [chunk, 50304] buffer costs more than ATen's fused `cross_entropy`
        # plus the recompute it was meant to save. **The recompute was cheaper
        # than replacing the kernel.**
        #
        # `_ChunkedLinearCE` is kept because it is the right base for anyone
        # who writes a Triton kernel for that softmax -- at which point the
        # ordering should flip. Until then it is not the default.
        # See documents/notes/ame_optimization_report.md §1.13.
        total = None
        for start in range(0, rows, chunk_size):
            hidden = flat[start : start + chunk_size]
            target = flat_targets[start : start + chunk_size]
            if torch.is_grad_enabled() and hidden.requires_grad:
                piece = checkpoint(chunk_loss, hidden, target, use_reentrant=False)
            else:
                piece = chunk_loss(hidden, target)
            total = piece if total is None else total + piece
        valid = (flat_targets != ignore_index).sum()
        return total / valid.clamp(min=1).to(total.dtype)


class LLM(nn.Module):
    def __init__(
        self,
        embedding: nn.Module,
        encoder: nn.Module,
        head: nn.Module,
    ):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.head = head

    def forward(self, input_ids, targets=None, ce_chunk_size=None):
        x = self.embedding(input_ids)
        x = self.encoder(x, input_ids)
        # `targets` threaded through to the head rather than handled here: the
        # head owns the projection whose output we are refusing to materialise,
        # and going through `self.head(...)` keeps FSDP2's all-gather hook in
        # the path (see TransformerHead.forward). `targets=None` is the
        # original call and returns logits.
        if targets is None:
            return self.head(x)
        return self.head(x, targets=targets, chunk_size=ce_chunk_size)


def parameter_groups(model, weight_decay, exclude=None, *, lr=None, bank_lr_ratio=1.0):
    """AdamW param groups: declared scales at wd=0, everything else at `weight_decay`.

    AdamW's decay is *decoupled* -- `p <- p*(1 - lr*wd)` runs whether or not the
    parameter has a gradient -- so it moves temperatures, gates and per-head
    scalars for which decay means nothing, and moves them hardest where the
    gradient is weakest. For most models that is a rounding error. For one whose
    research question is "does this coefficient go to zero" (src.core.ame) it is
    the answer, supplied by the optimizer.

    A module opts its own parameters out by naming them in a
    `no_weight_decay_parameters` attribute. **A model where nothing declares any
    returns a single group holding exactly `model.parameters()`**, so every
    existing config keeps the optimizer it had and no committed number moves.

    `exclude` drops parameters from the optimizer entirely -- not from decay,
    from the optimizer. The memory-parallel bank controller
    (src.core.memory_parallel) steps its own fp32 masters with its own AdamW, so
    the compute parameters it owns must not also be stepped here; they would be
    stepped on a gradient that is permanently None, which AdamW silently skips
    while still counting them in the state it allocates.

    `bank_lr_ratio` (trainer.bank_lr_ratio) moves the parameters a module names
    in `bank_parameters` -- AME's Q/K memory banks -- into a group of their own
    at `lr * bank_lr_ratio`, keeping their decay setting. The scheduler holds
    the ratio for the whole run (src.core.schedulers.CosineAnnealingToFraction).
    At 1.0, with no banks in the model, or with every bank excluded because a
    memory-parallel controller steps them (it applies the ratio itself), the
    groups are exactly the ones described above.
    """
    excluded_ids = {id(parameter) for parameter in (exclude or ())}

    def resolve(module, name):
        """`"tau"` or a dotted path like `"q_bank.weight"`."""
        for part in name.split("."):
            module = getattr(module, part, None)
            if module is None:
                return None
        return module

    def declared(attribute):
        ids = set()
        for module in model.modules():
            for name in getattr(module, attribute, ()):
                parameter = resolve(module, name)
                # Buffers answer getattr too -- a pinned alpha is a buffer, not
                # a parameter -- and must never reach the optimizer.
                if isinstance(parameter, nn.Parameter):
                    ids.add(id(parameter))
        return ids

    undecayed = declared("no_weight_decay_parameters")
    trainable = [p for p in model.parameters() if id(p) not in excluded_ids]

    banks = []
    if bank_lr_ratio != 1.0:
        bank_ids = declared("bank_parameters")
        banks = [p for p in trainable if id(p) in bank_ids]
        if banks and lr is None:
            raise ValueError(
                "parameter_groups: bank_lr_ratio needs the base `lr` to set the "
                "bank group's learning rate"
            )
        trainable = [p for p in trainable if id(p) not in bank_ids]

    if not undecayed:
        groups = [{"params": trainable, "weight_decay": weight_decay}]
    else:
        split = {True: [], False: []}
        for parameter in trainable:
            split[id(parameter) in undecayed].append(parameter)
        groups = [
            {"params": split[False], "weight_decay": weight_decay},
            {"params": split[True], "weight_decay": 0.0},
        ]

    for is_undecayed, decay in ((False, weight_decay), (True, 0.0)):
        params = [p for p in banks if (id(p) in undecayed) == is_undecayed]
        if params:
            groups.append(
                {"params": params, "weight_decay": decay, "lr": lr * bank_lr_ratio}
            )
    return groups


def attention_mechanism(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool,
    scale: Optional[float] = None,
):
    # `scale=None` is SDPA's own default, 1/sqrt(dhead); QK-norm passes 1.0
    # because its temperature is already folded into the query.
    # https://github.com/pytorch/pytorch/blob/ce503c1b40207dab770c28cbd4568cd9e105277b/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp#L556
    with torch.nn.attention.sdpa_kernel(
        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    ):
        return F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=None,
            is_causal=causal,
            scale=scale,
        )


class AttentionMechanism(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool,
        scale: Optional[float] = None,
    ):
        return attention_mechanism(
            query=query,
            key=key,
            value=value,
            causal=causal,
            scale=scale,
        )
