import math
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.utils import logging
from transformers import LlamaConfig

from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb, repeat_kv
from typing import Optional, Tuple
import math


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


# class LlamaRotaryEmbedding(torch.nn.Module):
#     def __init__(self, dim, max_position_embeddings=2048, base=500000, device=None):
#         super().__init__()
#         inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
#         self.register_buffer("inv_freq", inv_freq)

#         # Build here to make `torch.jit.trace` work.
#         self.max_seq_len_cached = max_position_embeddings
#         t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
#         freqs = torch.einsum("i,j->ij", t, self.inv_freq)
#         # Different from paper, but it uses a different permutation in order to obtain the same calculation
#         emb = torch.cat((freqs, freqs), dim=-1)
#         self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
#         self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

#     def forward(self, x, seq_len=None):
#         # x: [bs, num_attention_heads, seq_len, head_size]
#         # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
#         if seq_len > self.max_seq_len_cached:
#             self.max_seq_len_cached = seq_len
#             t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
#             freqs = torch.einsum("i,j->ij", t, self.inv_freq)
#             # Different from paper, but it uses a different permutation in order to obtain the same calculation
#             emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
#             self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
#             self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
#         return (
#             self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
#             self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
#         )

# class Llama3RotaryEmbedding(nn.Module):
#     def __init__(self, dim, max_position_embeddings=131072, base=500000.0, device=None):
#         super().__init__()
#         self.dim = dim
#         self.max_position_embeddings = max_position_embeddings
#         self.base = base
        
#         # --- LLAMA 3.2 SCALING CONFIGURATION ---
#         # Hardcoded from standard Llama 3.2 1B/3B config
#         self.factor = 32.0
#         self.low_freq_factor = 1.0
#         self.high_freq_factor = 4.0
#         self.original_max_position_embeddings = 8192
        
#         # 1. Generate Standard Inverse Frequencies
#         # Use float32 for stability
#         inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32).to(device) / self.dim))
        
#         # 2. Apply "llama3" Scaling Math
#         # Calculate Wavelengths: 2 * pi / freq
#         wavelen = 2 * math.pi / inv_freq
        
#         # Calculate Thresholds
#         low_freq_wavelen = self.original_max_position_embeddings / self.low_freq_factor
#         high_freq_wavelen = self.original_max_position_embeddings / self.high_freq_factor
        
#         # Calculate Scaled Wavelengths
#         new_wavelen = wavelen * self.factor
        
#         # Determine which frequencies to scale (Smooth Interpolation)
#         # Ratio: 0 = High Freq (Keep Original), 1 = Low Freq (Scale Fully)
#         ratio = (wavelen - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen)
#         ratio = torch.clamp(ratio, 0.0, 1.0)
        
#         # Interpolate Inverse Frequencies
#         # (1 - ratio) * old + ratio * new
#         new_inv_freq = 1.0 / new_wavelen
#         original_inv_freq = 1.0 / wavelen
#         final_inv_freq = original_inv_freq * (1 - ratio) + new_inv_freq * ratio
        
#         self.register_buffer("inv_freq", final_inv_freq, persistent=False)
#         self._set_cos_sin_cache(seq_len=max_position_embeddings, device=device, dtype=torch.get_default_dtype())

#     def _set_cos_sin_cache(self, seq_len, device, dtype):
#         self.max_seq_len_cached = seq_len
#         t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        
#         # Outer product to generate freq matrix
#         freqs = torch.outer(t, self.inv_freq)
        
#         # Cat to match Llama format (cos, cos)
#         emb = torch.cat((freqs, freqs), dim=-1)
        
#         # Store as standard dtype (usually bf16/fp16 for the model)
#         self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
#         self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

#     def forward(self, x, seq_len=None):
#         # print("LOL CALLED ---------------------------------")
#         if seq_len > self.max_seq_len_cached:
#             self._set_cos_sin_cache(seq_len=seq_len + 1024, device=x.device, dtype=x.dtype)
#         return (
#             self.cos_cached[:seq_len].to(dtype=x.dtype),
#             self.sin_cached[:seq_len].to(dtype=x.dtype),
#         )
    

# # def rotate_half(x):
# #     """Rotates half the hidden dims of the input."""
# #     x1 = x[..., : x.shape[-1] // 2]
# #     x2 = x[..., x.shape[-1] // 2 :]
# #     return torch.cat((-x2, x1), dim=-1)


# # def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
# #     gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
# #     gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
# #     cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
# #     sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    
# #     q_embed = (q * cos) + (rotate_half(q) * sin)
# #     k_embed = (k * cos) + (rotate_half(k) * sin)
# #     return q_embed, k_embed

# # --- 1. HELPER: Rotate Half Function ---
# def rotate_half(x):
#     """Rotates half the hidden dims of the input."""
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)

# # --- 2. HELPER: Custom Apply RoPE (Replaces Library Function) ---
# def apply_custom_rope(q, k, cos, sin, position_ids):
#     # This function handles the exact shapes produced by our Llama3RotaryEmbedding
#     # q, k: [Batch, Heads, Seq, Dim]
#     # cos, sin: [MaxSeq, Dim] (Raw Cache)
#     # position_ids: [Batch, Seq]
    
#     # 1. Select embeddings for the specific positions in this batch
#     # Output shape: [Batch, Seq, Dim]
#     cos = cos[position_ids]
#     sin = sin[position_ids]
    
#     # 2. Unsqueeze to broadcast over heads
#     # Output shape: [Batch, 1, Seq, Dim]
#     cos = cos.unsqueeze(1)
#     sin = sin.unsqueeze(1)
    
#     # 3. Apply rotation
#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed


class SVD_LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        ratio=1
    ):
        super().__init__()
        self.ratio = ratio
        low_rank = int(intermediate_size * hidden_size * self.ratio / (intermediate_size + hidden_size))
        self.gate_u_proj = nn.Linear(low_rank, intermediate_size, bias=False)
        self.gate_v_proj = nn.Linear(hidden_size, low_rank, bias=False)
        
        self.down_u_proj = nn.Linear(low_rank, hidden_size, bias=False)
        self.down_v_proj = nn.Linear(intermediate_size, low_rank, bias=False)
        
        self.up_u_proj = nn.Linear(low_rank, intermediate_size, bias=False)
        self.up_v_proj = nn.Linear(hidden_size, low_rank, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        up = self.up_u_proj(self.up_v_proj(x))
        gate = self.gate_u_proj(self.gate_v_proj(x))
        return self.down_u_proj(self.down_v_proj(self.act_fn(gate) * up))

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from transformers.models.llama.configuration_llama import LlamaConfig

# --- 1. Helper: Repeat KV for GQA ---
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim)
    to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# --- 2. Your Provided RoPE Class (Llama 3.2 Logic) ---
class RoPE(nn.Module):
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
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, freqs / factor, freqs)
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
        # Ensure we slice to the current sequence length of x
        seq_len = x.shape[1] # Assumes [batch, seq, heads, dim]
        cos_scaler = self.cos[:seq_len, :].to(x.device, dtype=x.dtype)
        sin_scaler = self.sin[:seq_len, :].to(x.device, dtype=x.dtype)
        
        # Reshape scalers for broadcasting: [seq, dim] -> [1, seq, 1, dim]
        cos_scaler = cos_scaler.view(1, seq_len, 1, -1)
        sin_scaler = sin_scaler.view(1, seq_len, 1, -1)
        
        return x * cos_scaler + x_rotated * sin_scaler

# --- 3. SVD Attention for Llama 3 (GQA + New RoPE) ---
class SVD_LlamaAttention(nn.Module):
    """Llama 3 Attention with SVD Compression and GQA support"""

    def __init__(self, config: LlamaConfig, ratio=1):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # --- Llama 3 GQA Setup ---
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.ratio = ratio

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Calculate low rank dimension
        low_rank = int(self.hidden_size * self.ratio / 2)

        # --- Query Projections (Uses full num_heads) ---
        self.q_u_proj = nn.Linear(low_rank, self.num_heads * self.head_dim, bias=False)
        self.q_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)

        # --- Key/Value Projections (Uses num_key_value_heads for GQA) ---
        self.k_u_proj = nn.Linear(low_rank, self.num_key_value_heads * self.head_dim, bias=False)
        self.k_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)

        self.v_u_proj = nn.Linear(low_rank, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)

        # --- Output Projection (Uses full hidden size) ---
        self.o_u_proj = nn.Linear(low_rank, self.hidden_size, bias=False)
        self.o_v_proj = nn.Linear(self.num_heads * self.head_dim, low_rank, bias=False)

        # --- Initialize Your Custom RoPE ---
        # Note: Llama 3.2 defaults
        rope_base = getattr(config, "rope_theta", 500000.0)
        # Check config for scaling, default to True for Llama 3.1/3.2
        use_scaling = True 
        
        self.rope = RoPE(
            dhead=self.head_dim,
            length=self.max_position_embeddings,
            base=rope_base,
            apply_freq_scaling=use_scaling,
            factor=32.0, # Llama 3 standard
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_position_embeddings=8192,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()

        # 1. SVD Projections
        # Q: [bsz, q_len, num_heads, head_dim]
        query_states = self.q_u_proj(self.q_v_proj(hidden_states))
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)

        # K/V: [bsz, q_len, num_key_value_heads, head_dim] (Smaller due to GQA)
        key_states = self.k_u_proj(self.k_v_proj(hidden_states))
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        value_states = self.v_u_proj(self.v_v_proj(hidden_states))
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # 2. Apply RoPE (Your custom class handles Q and K rotation)
        # Note: Your RoPE expects [batch, seq, heads, dim], which matches our view above
        query_states = self.rope(query_states)
        key_states = self.rope(key_states)

        # [bsz, heads, seq, dim] for attention math
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # 3. Handle KV Cache
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # 4. Repeat KV for GQA (Expand K/V heads to match Q heads)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 5. Attention Mechanism
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, key_states.size(2)):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, key_states.size(2))}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            # Ensure mask broadcasting matches
            if attention_mask.size() != (bsz, 1, q_len, key_states.size(2)):
                # You might need logic here to handle different mask shapes depending on your pipeline
                pass 
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))

        # Upcast attention to fp32 for stability
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # 6. Output Projection (SVD)
        attn_output = self.o_u_proj(self.o_v_proj(attn_output))

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value