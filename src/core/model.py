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

    def forward(self, input_ids):
        if input_ids.dim() == 3:
            # Token superposition (arXiv 2605.06546 §3.1): fp32 mean of each bag.
            bags = input_ids.flatten(0, 1)
            means = F.embedding_bag(bags, self.weight.float(), mode="mean")
            return means.to(self.weight.dtype).unflatten(0, input_ids.shape[:2])
        return super().forward(input_ids)


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

    def set_metric_logger(self, metric_logger):
        self.metric_logger = metric_logger

    def forward(self, x):
        normalized = self.norm(x)
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


class ZeroCenteredRMSNorm(nn.Module):
    """RMSNorm over the last dim with gain (1 + weight), weight zero-init, in fp32 (Qwen3.8 q/k norm)."""

    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        normed = F.rms_norm(x.float(), (x.shape[-1],), eps=self.eps)
        return (normed * (1.0 + self.weight.float())).type_as(x)


class RoPEAttention(nn.Module):
    def __init__(
        self,
        q_proj_fn,
        k_proj_fn,
        v_proj_fn,
        o_proj_fn,
        g_proj_fn,
        dmodel,
        q_heads,
        kv_heads,
        seq_len,
        rope_base,
        rope_scale_freqs: bool,
        rope: bool,
        qk_norm: bool,
        factor=32,
        low_freq_factor=1,
        high_freq_factor=4,
        original_max_position_embeddings=8192,
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

        # NoPE when rope=False: no RoPE module, which holds no state-dict entries.
        self.use_rope = rope
        if rope:
            self.rope = RoPE(
                dhead=self.dhead,
                length=seq_len,
                base=rope_base,
                apply_freq_scaling=rope_scale_freqs,
                factor=factor,
                low_freq_factor=low_freq_factor,
                high_freq_factor=high_freq_factor,
                original_max_position_embeddings=original_max_position_embeddings,
            )

        # Per-head gains, built last and only when on, so the off model is unchanged.
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = ZeroCenteredRMSNorm(self.dhead, eps=1e-6)
            self.k_norm = ZeroCenteredRMSNorm(self.dhead, eps=1e-6)

        # Output gate projection (None = off), built last so the off model's RNG draws don't move.
        self.output_gate = g_proj_fn is not None
        if self.output_gate:
            self.g_proj = g_proj_fn()

    def forward(self, x):
        query_states = self.q_proj(x)
        key_states = self.k_proj(x)
        value_states = self.v_proj(x)

        batch, seq_len = x.shape[:-1]
        q = query_states.view(batch, seq_len, self.q_heads, -1).transpose(1, 2)
        k = key_states.view(batch, seq_len, self.kv_heads, -1).transpose(1, 2)
        if self.qk_norm:
            # Before RoPE; k on its kv heads, before repeat_kv.
            q, k = self.q_norm(q), self.k_norm(k)
        if self.use_rope:
            q = self.rope(q)
            k = self.rope(k)

        v = value_states.view(batch, seq_len, self.kv_heads, -1).transpose(1, 2)

        # Imported lazily: src.core.llama imports AttentionMechanism from this
        # module, so a top-level import here would be circular.
        from src.core.llama import repeat_kv

        k = repeat_kv(k, self.q_heads // self.kv_heads)
        v = repeat_kv(v, self.q_heads // self.kv_heads)
        attention_output = self.attention_mechanism(
            query=q, key=k, value=v, causal=True
        )

        attention_output = attention_output.transpose(1, 2).contiguous().flatten(-2)
        if self.output_gate:
            # Elementwise, head-specific, query-dependent gate (Qwen's best variant).
            attention_output = attention_output * torch.sigmoid(self.g_proj(x))
        output = self.o_proj(attention_output)

        return output


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

    def forward(self, x):
        x = self.attention_layer(x)
        x = self.ff_layer(x)
        return x

    def set_log_pass(self, loop_pass):
        """Log this block's metrics per pass when a looped encoder runs it several times."""
        prefix = f"{self.log_name}/pass[{loop_pass}]"
        self.attention_layer.log_name = f"{prefix}/residual_attention"
        self.ff_layer.log_name = f"{prefix}/residual_feedforward"


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
        dmodel: int,
        loop,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([block_fn(i) for i in range(n_blocks)])
        # Looped depth: blocks [n_prelude, n_blocks - n_coda) run n_loops times.
        self.n_loops = loop["n_loops"]
        self.core_start = loop["n_prelude"]
        self.core_end = n_blocks - loop["n_coda"]
        if self.n_loops < 1:
            raise ValueError(f"loop.n_loops must be >= 1, got {self.n_loops}")
        if not 0 <= self.core_start < self.core_end <= n_blocks:
            raise ValueError(
                f"loop needs n_prelude, n_coda >= 0 and a non-empty core, got "
                f"n_prelude={loop['n_prelude']}, n_coda={loop['n_coda']} of {n_blocks} blocks"
            )
        if loop["injection"] not in ("none", "concat"):
            raise ValueError(
                f"loop.injection must be none or concat, got {loop['injection']!r}"
            )
        self.injection = (
            ConcatInjection(dmodel) if loop["injection"] == "concat" else None
        )
        # Lazy: src.core.moe imports this module.
        from src.core.moe import MoE

        # MoE keeps only its last call's aux loss and stats, so forward merges a looped one's passes.
        core = self.blocks[self.core_start : self.core_end]
        self._looped_moe = (
            [m for m in core.modules() if isinstance(m, MoE)]
            if self.n_loops > 1
            else []
        )

    def looped_modules(self):
        """The modules forward runs n_loops times: the core blocks and the injection."""
        core = list(self.blocks[self.core_start : self.core_end])
        return core + ([self.injection] if self.injection is not None else [])

    def forward(self, x):
        for block in self.blocks[: self.core_start]:
            x = block(x)
        prelude_out = x
        moe_calls = []
        for loop_pass in range(self.n_loops):
            if self.injection is not None:
                x = self.injection(x, prelude_out)
            for block in self.blocks[self.core_start : self.core_end]:
                if self.n_loops > 1:
                    block.set_log_pass(loop_pass)
                x = block(x)
            if self.training and self._looped_moe:
                moe_calls.append([(m.aux_loss, m.stats) for m in self._looped_moe])
        if moe_calls:
            self._merge_moe_calls(moe_calls)
        for block in self.blocks[self.core_end :]:
            x = block(x)
        return x

    def _merge_moe_calls(self, moe_calls):
        """Per looped MoE layer: sum the aux losses of its passes, average their stats."""
        for moe, calls in zip(self._looped_moe, zip(*moe_calls)):
            moe.aux_loss = sum(aux_loss for aux_loss, _ in calls)
            moe.stats = torch.stack([stats for _, stats in calls]).mean(0)


class ConcatInjection(nn.Module):
    """Huginn-style input injection: x <- W [x; e], W init [I | 0] so it starts as plain looping."""

    def __init__(self, dmodel):
        super().__init__()
        self.proj = nn.Linear(2 * dmodel, dmodel, bias=False)
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.weight[:, :dmodel].copy_(torch.eye(dmodel))

    def forward(self, x, prelude_out):
        return self.proj(torch.cat([x, prelude_out], dim=-1))


class TransformerHead(nn.Module):
    def __init__(self, linear_fn: Callable, norm_fn: Callable):
        super().__init__()
        self.norm = norm_fn()
        self.linear = linear_fn()

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x


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

    def forward(self, *args, **kwargs):
        x = self.embedding(*args, **kwargs)
        x = self.encoder(x)
        x = self.head(x)
        return x


def attention_mechanism(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool,
):
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
    ):
        return attention_mechanism(
            query=query,
            key=key,
            value=value,
            causal=causal,
        )
