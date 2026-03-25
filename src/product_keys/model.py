from functools import partial
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
import logging

from src.core.model import AttentionMechanism, RoPE
from src.projected_compression.model import LLM as LLM_projected_compression, \
    TransformerEncoder as TransformerEncoder_projected_compression, \
    Residual as Residual_projected_compression

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
        causal: bool = False
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

        self.causal = causal

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

    def forward(self, x, attention_mask=None):
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
                query=q, key=k, value=v, causal=self.causal
            )
            return self.o_proj(
                attention_output.transpose(1, 2).contiguous().flatten(-2)
            )

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dhead)

        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=attention_scores.device), diagonal=1
            ).bool()
            attention_scores = attention_scores.masked_fill(causal_mask, float("-inf"))

        if attention_mask is not None:
            pad_mask = attention_mask.unsqueeze(1).unsqueeze(2) == 0
            attention_scores = attention_scores.masked_fill(pad_mask, float("-inf"))

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


class RoPEProductKeysEncoderAttention(nn.Module):
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
        top_k: int
    ):
        super().__init__()

        # will work only with constant seq_len, in encoder-only setting
        assert math.sqrt(seq_len).is_integer(), "seq_len must be a perfect square"
        self.m = int(math.sqrt(seq_len))

        self.q_proj = q_proj_fn()
        self.k_proj = k_proj_fn()
        self.v_proj = v_proj_fn()
        self.o_proj = o_proj_fn()

        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.dhead = self.q_proj.weight.shape[0] // self.q_heads
        self.dhead_half = self.dhead // 2
        self.dmodel = dmodel
        self.seq_len = seq_len

        self.top_k = top_k

        self.rope = RoPE(
            dhead=self.dhead,
            length=seq_len,
            base=rope_base,
            apply_freq_scaling=rope_scale_freqs,
        )

    def __get_topk_candidates(self, query, key):
        scores = torch.matmul(query, key.transpose(-2, -1))

        _, indices = torch.topk(scores, k=self.top_k, dim=-1)

        key_expanded = key.unsqueeze(2).expand(-1, -1, self.seq_len, -1, -1)
        ind_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, -1, self.dhead_half)
        selected_vecs = torch.gather(key_expanded, 3, ind_expanded)

        return selected_vecs, indices

    @staticmethod
    def __gather_selected(source_tensor, idx_tensor):
        # source: (..., num_Candidates, D)
        # idx:    (..., K) -> expand to (..., K, D)
        idx_expanded = idx_tensor.unsqueeze(-1).expand(
            -1, -1, -1, -1, source_tensor.size(-1)
        )
        return torch.gather(source_tensor, 3, idx_expanded)

    def forward(self, x, attention_mask=None):
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

        # Split and aggregate keys
        k = k.view(batch, self.q_heads, self.m, self.m, self.dhead)
        k1 = k[..., : self.dhead_half].sum(-2)  # (B, H, m, d/2)
        k2 = k[..., self.dhead_half :].sum(-3)  # (B, H, m, d/2)

        # Split queries
        q1 = q[..., : self.dhead_half]  # (B, H, S, d/2)
        q2 = q[..., self.dhead_half :]  # (B, H, S, d/2)

        # --- First Retrieval (get top-k from each half) ---
        k1_vecs, k1_idxs = self.__get_topk_candidates(q1, k1)
        k2_vecs, k2_idxs = self.__get_topk_candidates(q2, k2)

        # --- Second Retrieval (find closest among combinations) ---
        # Expand to form a grid of all combinations of the selected K neighbors from both halves
        # Size after expansion: (B, H, S, K, K, D/2)
        c1 = k1_vecs.unsqueeze(-2).expand(
            -1, -1, -1, self.top_k, self.top_k, self.dhead_half
        )
        c2 = k2_vecs.unsqueeze(-3).expand(
            -1, -1, -1, self.top_k, self.top_k, self.dhead_half
        )

        # Concatenate halves to form full candidate vectors
        # candidates shape: (B, H, S, K*K, D)
        candidates = torch.cat([c1, c2], dim=-1).view(
            batch, self.q_heads, seq_len, -1, self.dhead
        )

        # Calculate similarity between full Q and the reconstructed candidates
        # q needs unsqueeze to broadcast: (B, H, S, 1, D) @ (B, H, S, K*K, D).T
        scores_final = (q.unsqueeze(-2) * candidates).sum(dim=-1)

        # Select top K closest combinations
        # selection_indices shape: (B, H, S, K)
        _, selection_indices = torch.topk(scores_final, k=self.top_k, dim=-1)

        # --- Gather final selected keys and values ---
        final_k = self.__gather_selected(candidates, selection_indices)

        idx_in_k1 = selection_indices // self.top_k
        idx_in_k2 = selection_indices % self.top_k

        final_row_idxs = torch.gather(k1_idxs, 3, idx_in_k1)
        final_col_idxs = torch.gather(k2_idxs, 3, idx_in_k2)

        v_indices = (final_row_idxs * self.m) + final_col_idxs
        v_flat_exp = v.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
        final_v = self.__gather_selected(v_flat_exp, v_indices)  # (B, H, S, K, D)

        # --- Attention: Softmax(Q @ K.T) @ V ---
        # q needs unsqueeze to broadcast: (B, H, S, 1, D) @ (B, H, S, K, D).T
        attn_scores = torch.matmul(
            q.unsqueeze(-2), final_k.transpose(-2, -1)
        ) / math.sqrt(self.dhead)

        # ! no causal mask as we work in encoder-only setting
        # causal_mask = torch.triu(
        #     torch.ones(seq_len, seq_len, device=attn_scores.device), diagonal=1
        # ).bool()
        # attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
        
        if attention_mask is not None:
            pad_mask = attention_mask.unsqueeze(1).unsqueeze(2) == 0
            attn_scores = attn_scores.masked_fill(pad_mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, final_v)
        attn_output = attn_output.squeeze(-2)

        return self.o_proj(attn_output.transpose(1, 2).contiguous().flatten(-2))


class LLM(LLM_projected_compression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        if "attention_mask" in kwargs:
            attention_mask = kwargs.pop("attention_mask")
        x = self.embedding(*args, **kwargs)
        x = self.encoder(x, attention_mask=attention_mask)
        x = self.head(x)
        return x


class TransformerEncoder(TransformerEncoder_projected_compression):
    def forward(self, x, *args, **kwargs):
        for block in self.blocks:
            x = block(x, *args, **kwargs)
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

    def forward(self, x, attention_mask=None):
        x = self.attention_layer(x, attention_mask=attention_mask)
        x = self.ff_layer(x)
        return x


class Residual(Residual_projected_compression):
    def forward(self, x, *args, **kwargs):
        normalized = self.norm(x)
        out = self.layer(normalized, *args, **kwargs)
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
