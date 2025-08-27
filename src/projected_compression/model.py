from functools import partial
import math
import os
import torch
import torch.nn as nn
from typing import Callable, Optional
from torch.nn import (
    LayerNorm as LayerNorm,
)  # used by FSDP, but it keeps getting removed during file formatting

from torch.nn.modules.normalization import RMSNorm as RMSNorm
from torchtune.modules.position_embeddings import (
    RotaryPositionalEmbeddings as RotaryPositionalEmbeddings,
)

from torch.nn import Embedding as Embedding

from src.core.llama import repeat_kv
from src.core.model import AttentionMechanism, Residual
from torch.nn.init import trunc_normal_
from torch import zeros as zeros
import torch.distributed as dist
import torch.nn.functional as F


def llm_random_weight_init(fan_in, scale):
    std = scale * (1 / fan_in) ** 0.5
    low = -2 * std
    high = 2 * std
    return partial(trunc_normal_, mean=0.0, std=std, a=low, b=high)


# linear takes partial function which returns init_fn upon giving fan_in as an input, but sometimes it does not depend on fan_in
dummy_weight_init = lambda _: trunc_normal_
dummy_zeros = lambda _: zeros


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dmodel: int, init_fn: Optional[Callable]):
        super().__init__()
        weight = torch.empty(vocab_size, dmodel, dtype=torch.float32)
        init_fn(weight)
        self.embedding = nn.Embedding(vocab_size, dmodel, _weight=weight)

    def forward(self, x):
        return self.embedding(x)


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
            self.metric_logger.accumulate_metrics(
                layer_name=f"{self.log_name}",
                transform_fn=Residual.intermediate_norms,
                calculate_fn=Residual.calculate_metrics,
                metrics={
                    "residual_stream": x,
                    "updates": out,
                },
            )
        return out + x

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
    def __init__(self, dhead, length, base, apply_freq_scaling):
        super().__init__()
        self.dhead = dhead
        self.length = length
        self.base = base
        self.apply_freq_scaling = apply_freq_scaling
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

    def scale_freqs(self, freqs, factor=32):
        # factor = `8` in the original implementation according to HuggingFace
        low_freq_factor = 1  # `1` in the original implementation
        high_freq_factor = 4  # `4` in the original implementation
        old_context_len = 8192  # `8192` in the original implementation

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

        self.rope = RoPE(
            dhead=self.dhead,
            length=seq_len,
            base=rope_base,
            apply_freq_scaling=rope_scale_freqs,
        )

    def forward(self, x):
        query_states = self.q_proj(x)
        key_states = self.k_proj(x)
        value_states = self.v_proj(x)

        batch, seq_len = x.shape[:-1]
        q = query_states.view(batch, seq_len, self.q_heads, -1).transpose(1, 2)
        q = self.rope(q)
        k = key_states.view(batch, seq_len, self.kv_heads, -1).transpose(1, 2)
        k = self.rope(k)

        v = value_states.view(batch, seq_len, self.kv_heads, -1).transpose(1, 2)

        k = repeat_kv(k, self.q_heads // self.kv_heads)
        v = repeat_kv(v, self.q_heads // self.kv_heads)
        attention_output = self.attention_mechanism(
            query=q, key=k, value=v, causal=True
        )

        output = self.o_proj(attention_output.transpose(1, 2).contiguous().flatten(-2))

        return output


class FeedForward(nn.Module):
    def __init__(self, ff_pre_act_fn, ff_post_act_fn):
        super().__init__()
        self.relu = nn.ReLU()
        self.ff_pre_act = ff_pre_act_fn()
        self.ff_post_act = ff_post_act_fn()

    def forward(self, x):
        x = self.ff_pre_act(x)
        x = self.relu(x)
        x = self.ff_post_act(x)
        return x


class LlamaFeedForward(nn.Module):
    def __init__(self, dmodel, dff, linear_fn):
        super().__init__()
        self.ff_pre_act = linear_fn(dmodel, dff)
        self.gate = linear_fn(dmodel, dff)
        self.silu = nn.SiLU()
        self.ff_post_act = linear_fn(dff, dmodel)

    def forward(self, x):
        gated = self.gate(x)
        gated = self.silu(gated)
        x = self.ff_pre_act(x)
        x = x * gated
        x = self.ff_post_act(x)
        return x


class ProjectedLlamaFeedForward(nn.Module):
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

        self.attention_layer = Residual(
            norm=norm_fn(),
            layer=attention_fn(),
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


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        block_fn: Callable[[int], nn.Module],
        n_blocks: int,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([block_fn(i) for i in range(n_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


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


class ProjectedLinear(nn.Module):
    __constants__ = [
        "result_in_features",
        "result_out_features",
        "base_in_features",
        "base_out_features",
    ]
    result_in_features: Optional[int]
    result_out_features: Optional[int]
    base_in_features: int
    base_out_features: int

    def __init__(
        self,
        result_in_features: Optional[int],
        result_out_features: Optional[int],
        base_in_features: int,
        base_out_features: int,
        device=None,
        dtype=None,
    ) -> nn.Module:

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.result_in_features = result_in_features
        self.result_out_features = result_out_features
        self.base_in_features = base_in_features
        self.base_out_features = base_out_features
        self.initialized_compression = False

        self.weight = nn.Parameter(
            torch.rand((base_out_features, base_in_features), **factory_kwargs)
        )
        self.projection_in_weight = None
        self.projection_out_weight = None
        self.auxiliary_weight = None

    def activate_projections(self):
        if self.result_in_features is not None:
            self.projection_in_weight.requires_grad_(True)

        if self.result_out_features is not None:
            self.projection_out_weight.requires_grad_(True)
        

    def init_projections(
        self,
        proj_in_topk_indices: Optional[torch.Tensor],
        proj_out_topk_indices: Optional[torch.Tensor],
        factory_kwargs={},
    ):
        if proj_in_topk_indices is None:
            assert (
                self.projection_in_weight is None
            ), "Projection 'in' is decalred, but not passed."
        else:
            assert (
                len(proj_in_topk_indices) == self.result_in_features
            ), "projection 'in' dimension mismatch."

        if proj_out_topk_indices is None:
            assert (
                self.projection_out_weight is None
            ), "Projection 'out' is decalred, but not passed."
        else:
            assert (
                len(proj_out_topk_indices) == self.result_out_features
            ), "projection 'out' dimension mismatch."

        if self.result_in_features is not None:
            weight = torch.zeros(
                self.base_in_features, self.result_in_features, **factory_kwargs
            )
            weight[proj_in_topk_indices, torch.arange(self.result_in_features)] = 1
            self.projection_in_weight = nn.Parameter(weight, requires_grad=True)

        if self.result_out_features is not None:
            weight = torch.zeros(
                self.result_out_features, self.base_out_features, **factory_kwargs
            )
            weight[torch.arange(self.result_out_features), proj_out_topk_indices] = 1
            self.projection_out_weight = nn.Parameter(weight, requires_grad=True)

        if self.result_in_features is not None or self.result_out_features is not None:
            final_in_features = (
                self.result_in_features
                if self.result_in_features is not None
                else self.base_in_features
            )
            final_out_features = (
                self.result_out_features
                if self.result_out_features is not None
                else self.base_out_features
            )
            weight = torch.zeros(
                final_out_features, final_in_features, **factory_kwargs
            )
            self.auxiliary_weight = nn.Parameter(weight, requires_grad=True)

        self.initialized_compression = True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # gradient magic happens here - PC optimization
        # TODO maybe order of projections matter for speed
        weight = self.weight

        if self.result_in_features is not None:
            weight = weight @ self.projection_in_weight

        if self.result_out_features is not None:
            weight = self.projection_out_weight @ weight

        if (
            self.result_in_features is not None
            or self.result_out_features is not None
        ):
            weight += self.auxiliary_weight

        return F.linear(input, weight, bias=None)

    def extra_repr(self) -> str:
        if self.result_in_features is not None:
            result = f"(projection_in_weight) ({self.base_in_features}, {self.result_in_features})\n"
        else:
            result = ""
        result += f"(weight) ({self.base_out_features}, {self.base_in_features})"
        if self.result_out_features is not None:
            result += f"\n(projection_out_weight) ({self.result_out_features}, {self.base_out_features})"
        return result

class ProjectedEmbedding(nn.Module):
    def __init__(
        self,
        embedding: nn.Module,
        result_out_features: int,
    ):
        super().__init__()
        self.embedding = embedding
        self.projection = None
        self.result_out_features = result_out_features
        self.initialized_compression = False

    def forward(self, x):
        result = self.embedding(x)
        if self.initialized_compression:
            result = F.linear(result, self.projection, bias=None)
            result += self.auxiliary_weight(x)
        return result

    def init_projection(self, topk_dmodel_indices, **factory_kwargs):
        assert len(topk_dmodel_indices) == self.result_out_features
        vocab_size, dmodel = self.embedding.weight.shape
        weight = torch.zeros(self.result_out_features, dmodel, **factory_kwargs)
        weight[torch.arange(self.result_out_features), topk_dmodel_indices] = 1
        self.projection = nn.Parameter(weight, requires_grad=True)
        self.initialized_compression = True

        zeros = torch.zeros(vocab_size, self.result_out_features, **factory_kwargs)
        self.auxiliary_weight = nn.Embedding(
            vocab_size, self.result_out_features, _weight=zeros
        )
