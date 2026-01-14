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
        top_k: int,
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
        # TODO
        # ! we've calculated both halves separately, we can reuse that
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

        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, final_v)
        attn_output = attn_output.squeeze(-2)

        return self.o_proj(attn_output.transpose(1, 2).contiguous().flatten(-2))


class ProductKeysMemory(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        query_dim: int, 
        n_sub_keys: int, 
        k_neighbors: int, 
        n_heads: int = 4,
        **kwargs,  # To ignore unused args
    ):
        super().__init__()
        self.n_heads = n_heads
        self.k = k_neighbors
        self.n_sub_keys = n_sub_keys
        self.query_dim = query_dim

        # Query Network
        # Projects input to query space. BatchNorm is crucial for PKM stability/convergence.
        self.query_proj = nn.Linear(d_model, n_heads * query_dim)
        self.query_bn = nn.BatchNorm1d(n_heads * query_dim)

        # Sub-Keys (Codebooks)
        # Two separate sets of keys for the product quantization
        self.c1 = nn.Parameter(torch.randn(n_heads, n_sub_keys, query_dim // 2))
        self.c2 = nn.Parameter(torch.randn(n_heads, n_sub_keys, query_dim // 2))

        # Memory Values
        # The actual values retrieved. Size is (n_sub_keys^2, d_model)
        self.values = nn.Embedding(n_sub_keys * n_sub_keys, d_model)
        nn.init.normal_(self.values.weight, mean=0, std=d_model**-0.5)

    def _get_knn(self, queries, codebooks):
        """
        Calculates dot product scores and retrieves top-k indices and values.
        """
        # queries: (batch, head, sub_dim)
        # codebooks: (head, n_sub_keys, sub_dim)
        
        # Calculate similarity (dot product)
        scores = torch.einsum("bhd,hnd->bhn", queries, codebooks)
        
        # Select top-k
        top_scores, top_indices = torch.topk(scores, k=self.k, dim=-1, largest=True)
        return top_scores, top_indices

    def forward(self, x):
        bs, seq_len, d_model = x.shape

        # 1. Query Projection
        x_flat = x.view(-1, d_model)
        q = self.query_proj(x_flat)
        q = self.query_bn(q)
        q = q.view(bs * seq_len, self.n_heads, self.query_dim)

        # Split query into two halves for product quantization
        q1, q2 = torch.chunk(q, 2, dim=-1)

        # 2. Retrieve Top-K candidates for each half
        scores1, idx1 = self._get_knn(q1, self.c1)
        scores2, idx2 = self._get_knn(q2, self.c2)

        # 3. Cartesian Product of Scores
        # Sum every score from the first half with every score from the second half
        # (BS, H, K, 1) + (BS, H, 1, K) -> (BS, H, K, K)
        all_scores = scores1.unsqueeze(3) + scores2.unsqueeze(2)

        # Flatten the KxK grid to K^2 to find the global top-k
        all_scores_flat = all_scores.view(bs * seq_len, self.n_heads, -1)
        
        # Select the best combinations (global top-k)
        global_scores, global_top_indices = torch.topk(all_scores_flat, self.k, dim=-1)

        # 4. Index Mapping
        # Map the flattened indices back to the original codebook indices
        idx1_pos = global_top_indices // self.k
        idx2_pos = global_top_indices % self.k

        # Gather the actual sub-key indices
        real_idx1 = torch.gather(idx1, 2, idx1_pos)
        real_idx2 = torch.gather(idx2, 2, idx2_pos)

        # Calculate the global memory index: i * N_keys + j
        memory_indices = real_idx1 * self.n_sub_keys + real_idx2

        # 5. Read from Memory
        attn_weights = F.softmax(global_scores, dim=-1) # (BS, H, K)

        flat_indices = memory_indices.view(-1)
        values_selected = self.values(flat_indices) 
        values_selected = values_selected.view(bs * seq_len, self.n_heads, self.k, d_model)

        # Weighted sum of retrieved values
        out_heads = (values_selected * attn_weights.unsqueeze(-1)).sum(dim=2)

        # 6. Aggregation
        # Sum outputs across all heads
        output = out_heads.sum(dim=1) # (BS, d_model)
        
        return output.view(bs, seq_len, d_model)
