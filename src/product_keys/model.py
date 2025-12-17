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
        self, input_dim: int, query_dim: int, n_sub_keys: int, k_neighbors: int, n_heads:int=4
    ):
        super().__init__()
        self.n_heads = n_heads
        self.k = k_neighbors
        self.n_sub_keys = n_sub_keys
        self.query_dim = query_dim
        self.sub_key_dim = query_dim // 2
        self.memory_dim = input_dim  # todo remove this param, rename input_dim to smth like dmodel

        # Query Network (Learnable)
        self.query_proj = nn.Linear(input_dim, n_heads * query_dim)
        self.query_bn = nn.BatchNorm1d(n_heads * query_dim)

        # Sub-Keys (Learnable, Separate for each head)
        self.c1 = nn.Parameter(torch.randn(n_heads, n_sub_keys, self.sub_key_dim))
        self.c2 = nn.Parameter(torch.randn(n_heads, n_sub_keys, self.sub_key_dim))

        # Memory Values (Learnable, Shared across heads)
        self.values = nn.Embedding(n_sub_keys * n_sub_keys, self.memory_dim)
        nn.init.normal_(self.values.weight, mean=0, std=input_dim**-0.5)

    def _get_knn_indices_gpu(self, queries, codebooks):
        # queries: (batch, head, dim)
        # codebooks: (head, num_sub_keys, dim)
        scores = torch.einsum("bhd,hnd->bhn", queries, codebooks)

        _, indices = torch.topk(scores, k=self.k, dim=-1, largest=True)

        return indices

    def _gather_keys(self, codebook, indices):
        # codebook: (n_heads, n_keys, dim)
        # indices: (batch, n_heads, k)
        # Output: (batch, n_heads, k, dim)

        # Expand codebook to batch size
        # (1, n_heads, n_keys, dim) -> (batch, n_heads, n_keys, dim)
        cb_exp = codebook.unsqueeze(0).expand(indices.size(0), -1, -1, -1)

        # Expand indices to dim
        # (batch, n_heads, k, 1) -> (batch, n_heads, k, dim)
        idx_exp = indices.unsqueeze(-1).expand(-1, -1, -1, codebook.size(-1))

        return torch.gather(cb_exp, 2, idx_exp)

    def forward(self, x):
        bs, seq_len, input_dim = x.shape

        # Flatten batch and sequence for processing
        x_flat = x.view(-1, input_dim)  # (batch*seq, input_dim)
        q = self.query_proj(x_flat)
        q = self.query_bn(q)
        q = q.view(-1, self.n_heads, self.query_dim)

        q1, q2 = torch.chunk(q, 2, dim=-1)  # Each: (batch, n_heads, sub_dim)

        # Retrieve Sub-Key Indices
        idx1 = self._get_knn_indices_gpu(q1, self.c1)
        idx2 = self._get_knn_indices_gpu(q2, self.c2)

        k1_selected = self._gather_keys(self.c1, idx1)
        k2_selected = self._gather_keys(self.c2, idx2)

        # Compute Dot Products (Scores)
        # q1: (batch, n_heads, dim) -> unsqueeze -> (batch, n_heads, 1, dim)
        # Todo we've calculated both halves separately, we can reuse that
        scores1 = (q1.unsqueeze(2) * k1_selected).sum(dim=-1)  # (batch, n_heads, k)
        scores2 = (q2.unsqueeze(2) * k2_selected).sum(dim=-1)  # (batch, n_heads, k)

        # Cartesian Product Sum
        # (batch, n_heads, k, 1) + (batch, n_heads, 1, k) -> (batch, n_heads, k, k)
        all_scores = scores1.unsqueeze(3) + scores2.unsqueeze(2)

        # Flatten (k, k) to k^2 to find global top-k
        all_scores_flat = all_scores.view(
            all_scores.size(0), self.n_heads, -1
        )  # (batch, n_heads, k*k)

        # Select global top-k from the k^2 candidates
        top_scores, top_indices_flat = torch.topk(all_scores_flat, self.k, dim=-1)

        # 4. Map back to Global Memory Indices
        # We need to find which (i, j) pair in the k*k grid corresponded to the top scores

        # Create grid of local indices
        # idx1: (batch, n_heads, k)
        idx1_grid = (
            idx1.unsqueeze(3)
            .expand(-1, -1, -1, self.k)
            .reshape(idx1.size(0), self.n_heads, -1)
        )
        idx2_grid = (
            idx2.unsqueeze(2)
            .expand(-1, -1, self.k, -1)
            .reshape(idx2.size(0), self.n_heads, -1)
        )

        # Gather the actual codebook indices corresponding to the winners
        best_idx1 = torch.gather(idx1_grid, 2, top_indices_flat)
        best_idx2 = torch.gather(idx2_grid, 2, top_indices_flat)

        # Calculate global memory index: i * |C| + j
        global_indices = best_idx1 * self.n_sub_keys + best_idx2

        # 5. Read from Value Memory (Weighted Sum)
        # Softmax over the top-k scores
        attn_weights = F.softmax(top_scores, dim=-1)  # (batch, n_heads, k)

        # Fetch Values
        # self.values.weight: (total_keys, val_dim)
        # global_indices: (batch, n_heads, k)
        # Output: (batch, n_heads, k, val_dim)

        # Since embedding weight is 2D, we flatten indices to lookup
        flat_indices = global_indices.view(-1)
        values_selected = F.embedding(flat_indices, self.values.weight)
        values_selected = values_selected.view(*global_indices.shape, self.memory_dim)

        # Weighted Sum
        # (batch, n_heads, k, val_dim) * (batch, n_heads, k, 1) -> sum over k
        head_outputs = (values_selected * attn_weights.unsqueeze(-1)).sum(
            dim=2
        )  # (batch, n_heads, val_dim)

        # 6. Multi-Head Aggregation
        # "The memory simply sums the output m_i(x) of each head" [cite: 210]
        output = head_outputs.sum(dim=1)  # (batch, val_dim)

        # Restore sequence dimension
        output = output.view(bs, seq_len, self.memory_dim)

        return output
