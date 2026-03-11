import math

import torch
import torch.nn as nn

from src.core.model import AttentionMechanism


class RoPE(nn.Module):
    """
    RoPE with configurable freq scaling parameters for context extension.

    Extends src.core.model.RoPE by making the freq scaling params
    (factor, low_freq_factor, high_freq_factor, original_max_position_embeddings)
    configurable instead of hardcoded. This allows setting
    original_max_position_embeddings to the source training length when
    extending context.

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
        factor=32,
        low_freq_factor=1,
        high_freq_factor=4,
        original_max_position_embeddings=8192,
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
    def __init__(
        self,
        q_proj_fn,
        k_proj_fn,
        v_proj_fn,
        o_proj_fn,
        pre_attn_fn,
        dmodel,
        q_heads,
        kv_heads,
        seq_len,
        rope_base,
        rope_scale_freqs: bool,
        compile: bool = False,
        rope_factor: int = 32,
        rope_low_freq_factor: int = 1,
        rope_high_freq_factor: int = 4,
        rope_original_max_position_embeddings: int = 8192,
    ):
        super().__init__()
        self.q_proj = q_proj_fn()
        self.k_proj = k_proj_fn()
        self.v_proj = v_proj_fn()
        self.o_proj = o_proj_fn()
        self.pre_attn_fn = pre_attn_fn() if pre_attn_fn is not None else None
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
            factor=rope_factor,
            low_freq_factor=rope_low_freq_factor,
            high_freq_factor=rope_high_freq_factor,
            original_max_position_embeddings=rope_original_max_position_embeddings,
        )

        if compile:
            self.forward = torch.compile(
                self.forward,
                mode="max-autotune-no-cudagraphs",
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

        from src.core.llama import repeat_kv

        k = repeat_kv(k, self.q_heads // self.kv_heads)
        v = repeat_kv(v, self.q_heads // self.kv_heads)

        # Apply QKNorm before reshape (normalizes over full datt, not per-head)
        if self.pre_attn_fn is not None:
            q, k, v = self.pre_attn_fn(q, k, v)

        attention_output = self.attention_mechanism(
            query=q, key=k, value=v, causal=True
        )

        output = self.o_proj(attention_output.transpose(1, 2).contiguous().flatten(-2))

        return output
