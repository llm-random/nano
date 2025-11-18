from functools import partial
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
import logging

from src.core.model import AttentionMechanism, RoPE

logger = logging.getLogger(__name__)


# useful for deterministic tests
def deterministic_weight_init(fan_in, scale):
    std = scale * (1 / fan_in) ** 0.5
    low = -2 * std
    high = 2 * std
    generator = torch.Generator().manual_seed(42)
    return partial(trunc_normal_, mean=0.0, std=std, a=low, b=high, generator=generator)


class RoPETopKAttention(nn.Module):
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
        top_k: int,
        top_k_before_softmax: bool = True,
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

        self.top_k = top_k
        self.top_k_before_softmax = top_k_before_softmax

        self.rope = RoPE(
            dhead=self.dhead,
            length=seq_len,
            base=rope_base,
            apply_freq_scaling=rope_scale_freqs,
        )

    def __apply_topk_mask(self, x, fill_value: float):
        top_k_values, _ = torch.topk(x, self.top_k, dim=-1)
        threshold = top_k_values[..., -1].unsqueeze(-1)
        mask_topk = x < threshold
        return x.masked_fill(mask_topk, fill_value)

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

        # standard attention if seq_len is smaller or equal top_k
        if seq_len <= self.top_k:
            attention_output = self.attention_mechanism(
                query=q, key=k, value=v, causal=True
            )
            return self.o_proj(
                attention_output.transpose(1, 2).contiguous().flatten(-2)
            )

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dhead)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=attention_scores.device), diagonal=1
        ).bool()
        attention_scores = attention_scores.masked_fill(causal_mask, float("-inf"))

        if self.top_k_before_softmax:
            attention_scores = self.__apply_topk_mask(
                attention_scores, fill_value=float("-inf")
            )
            attention_weight = F.softmax(attention_scores, dim=-1)
        else:
            # notice that we do not renormalize scores after masking
            attention_weight = F.softmax(attention_scores, dim=-1)
            attention_weight = self.__apply_topk_mask(attention_weight, fill_value=0.0)

        attention_output = torch.matmul(attention_weight, v)

        return self.o_proj(attention_output.transpose(1, 2).contiguous().flatten(-2))
