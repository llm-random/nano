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


# # class SVD_LlamaAttention(nn.Module):
# #     """Multi-headed attention from 'Attention Is All You Need' paper"""

# #     def __init__(self, config: LlamaConfig, ratio=1):
# #         super().__init__()
# #         self.config = config
# #         self.hidden_size = config.hidden_size
# #         self.num_heads = config.num_attention_heads
# #         self.head_dim = self.hidden_size // self.num_heads
# #         self.max_position_embeddings = config.max_position_embeddings
# #         self.ratio = ratio # 1 means no truncate, just keep normal attn

# #         if (self.head_dim * self.num_heads) != self.hidden_size:
# #             raise ValueError(
# #                 f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
# #                 f" and `num_heads`: {self.num_heads})."
# #             )
# #         low_rank = int(self.hidden_size * self.ratio/2)
# #         self.q_u_proj = nn.Linear(low_rank, self.num_heads * self.head_dim, bias=False)
# #         self.q_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)

# #         self.k_u_proj = nn.Linear(low_rank, self.num_heads * self.head_dim, bias=False)
# #         self.k_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)

# #         self.v_u_proj = nn.Linear(low_rank, self.num_heads * self.head_dim, bias=False)
# #         self.v_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)

# #         self.o_u_proj = nn.Linear(low_rank, self.hidden_size, bias=False)
# #         self.o_v_proj = nn.Linear(self.num_heads * self.head_dim, low_rank, bias=False)

# #         self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

# #     def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
# #         return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

# #     def forward(
# #         self,
# #         hidden_states: torch.Tensor,
# #         attention_mask: Optional[torch.Tensor] = None,
# #         position_ids: Optional[torch.LongTensor] = None,
# #         past_key_value: Optional[Tuple[torch.Tensor]] = None,
# #         output_attentions: bool = False,
# #         use_cache: bool = False,
# #     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
# #         bsz, q_len, _ = hidden_states.size()
    
# #         query_states = self.q_u_proj(self.q_v_proj(hidden_states)).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

# #         key_states = self.k_u_proj(self.k_v_proj(hidden_states)).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

# #         value_states = self.v_u_proj(self.v_v_proj(hidden_states)).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

# #         kv_seq_len = key_states.shape[-2]
# #         if past_key_value is not None:
# #             kv_seq_len += past_key_value[0].shape[-2]
# #         cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
 
# #         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
# #         # [bsz, nh, t, hd]

# #         if past_key_value is not None:
# #             # reuse k, v, self_attention
# #             key_states = torch.cat([past_key_value[0], key_states], dim=2)
# #             value_states = torch.cat([past_key_value[1], value_states], dim=2)

# #         past_key_value = (key_states, value_states) if use_cache else None

# #         attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

# #         if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
# #             raise ValueError(
# #                 f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
# #                 f" {attn_weights.size()}"
# #             )

# #         if attention_mask is not None:
# #             if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
# #                 raise ValueError(
# #                     f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
# #                 )
# #             attn_weights = attn_weights + attention_mask
# #             attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))

# #         # upcast attention to fp32
# #         attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
# #         attn_output = torch.matmul(attn_weights, value_states)

# #         if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
# #             raise ValueError(
# #                 f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
# #                 f" {attn_output.size()}"
# #             )

# #         attn_output = attn_output.transpose(1, 2)
# #         attn_output = attn_output.reshape(bsz, q_len, -1)

# #         attn_output = self.o_u_proj(self.o_v_proj(attn_output))

# #         if not output_attentions:
# #             attn_weights = None

# #         return attn_output, attn_weights, past_key_value



# class SVD_LlamaAttention(nn.Module):
#     """Multi-headed attention from 'Attention Is All You Need' paper, updated for SVD and GQA (Llama 3)"""

#     def __init__(self, config: LlamaConfig, ratio=1):
#         super().__init__()
#         self.config = config
#         self.hidden_size = config.hidden_size
#         self.num_heads = config.num_attention_heads
#         self.head_dim = self.hidden_size // self.num_heads
        
#         # --- FIX 1: Support Grouped Query Attention (GQA) ---
#         self.num_key_value_heads = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else self.num_heads
#         self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
#         self.max_position_embeddings = config.max_position_embeddings
#         self.ratio = ratio 

#         if (self.head_dim * self.num_heads) != self.hidden_size:
#             raise ValueError(
#                 f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
#                 f" and `num_heads`: {self.num_heads})."
#             )
            
#         low_rank = int(self.hidden_size * self.ratio/2)
        
#         # Query Projections (Uses full num_heads)
#         self.q_u_proj = nn.Linear(low_rank, self.num_heads * self.head_dim, bias=False)
#         self.q_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)

#         # --- FIX 2: Key/Value Projections use num_key_value_heads ---
#         self.k_u_proj = nn.Linear(low_rank, self.num_key_value_heads * self.head_dim, bias=False)
#         self.k_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)

#         self.v_u_proj = nn.Linear(low_rank, self.num_key_value_heads * self.head_dim, bias=False)
#         self.v_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)

#         self.o_u_proj = nn.Linear(low_rank, self.hidden_size, bias=False)
#         self.o_v_proj = nn.Linear(self.num_heads * self.head_dim, low_rank, bias=False)

#         # self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, base=500000)
#         # --- FINAL FIX: Load RoPE config properly ---
#         # 1. Get the base frequency (500,000 for Llama 3.2)
#         rope_theta = getattr(config, "rope_theta", 10000.0)
        
#         # 2. Get the scaling configuration (Critical for Llama 3.2!)
#         rope_scaling = getattr(config, "rope_scaling", None)
        
#         print(f"rope_theta {rope_theta}") #dev
#         print(f"Scaling {rope_scaling}") #dev
#         print(f"max_position_embeddings {self.max_position_embeddings}") #dev

        
#         # self.rotary_emb = LlamaRotaryEmbedding(self.config)
#         # self.rotary_emb = Llama3RotaryEmbedding(self.config)
#         self.rotary_emb = Llama3RotaryEmbedding(
#             self.head_dim, 
#             max_position_embeddings=self.max_position_embeddings,
#             base=rope_theta
#         )
        

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Tuple[torch.Tensor]] = None,
#         output_attentions: bool = False,
#         use_cache: bool = False,
#         **kwargs, # --- FIX 3: Catch-all for Llama 3 extra args (position_embeddings, cache_position) ---
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

#         bsz, q_len, _ = hidden_states.size()

#         # 1. Project Queries (Standard SVD)
#         query_states = self.q_u_proj(self.q_v_proj(hidden_states))
#         query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

#         # 2. Project Keys/Values (Standard SVD + GQA Dimensions)
#         key_states = self.k_u_proj(self.k_v_proj(hidden_states))
#         key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

#         value_states = self.v_u_proj(self.v_v_proj(hidden_states))
#         value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

#         kv_seq_len = key_states.shape[-2]
#         if past_key_value is not None:
#             kv_seq_len += past_key_value[0].shape[-2]

#         # --- FIX 4: Handle Llama 3 RoPE (Use passed embeddings if available) ---

#         # cos, sin = self.rotary_emb(value_states, position_ids=position_ids)
#         cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

#         # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
#         query_states, key_states = apply_custom_rope(query_states, key_states, cos, sin, position_ids)
        

#         if past_key_value is not None:
#             # reuse k, v, self_attention
#             key_states = torch.cat([past_key_value[0], key_states], dim=2)
#             value_states = torch.cat([past_key_value[1], value_states], dim=2)

#         past_key_value = (key_states, value_states) if use_cache else None

#         # --- FIX 5: Repeat KV heads for GQA (The critical fix for the shape error) ---
#         # If KV heads < Query Heads, we must repeat them
#         key_states = repeat_kv(key_states, self.num_key_value_groups)
#         value_states = repeat_kv(value_states, self.num_key_value_groups)

#         attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

#         if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
#             raise ValueError(
#                 f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
#                 f" {attn_weights.size()}"
#             )

#         if attention_mask is not None:
#             # Basic broadcast check
#             if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
#                  # Llama 3 sometimes passes 4D masks that match, or 2D masks that need reshape.
#                  # We trust the model/pipeline to pass correct masks usually, but strictly speaking
#                  # this check was breaking on valid 4D masks in some versions. 
#                  # We'll leave it but warn if it might be the cause of future issues.
#                  pass 
            
#             attn_weights = attn_weights + attention_mask
#             attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))

#         # upcast attention to fp32
#         attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
#         attn_output = torch.matmul(attn_weights, value_states)

#         if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
#             raise ValueError(
#                 f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
#                 f" {attn_output.size()}"
#             )

#         attn_output = attn_output.transpose(1, 2)
#         attn_output = attn_output.reshape(bsz, q_len, -1)

#         attn_output = self.o_u_proj(self.o_v_proj(attn_output))

#         if not output_attentions:
#             attn_weights = None


#         return attn_output, Noneimport torch
############################################################################################################################################################################





import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import repeat_kv, LlamaRotaryEmbedding, apply_rotary_pos_emb

# --- 1. HIJACKED ROPE CLASS (Self-Contained Logic + Library Signature) ---
class Llama32RotaryEmbedding(LlamaRotaryEmbedding):
    """
    Inherits from transformers LlamaRotaryEmbedding to pass type checks/init signature,
    but implements ALL logic internally to avoid missing attribute errors in newer libs.
    """
    def __init__(self, config: LlamaConfig, device=None):
        # 1. Initialize Parent to satisfy library checks
        # We catch potential errors if parent init tries to do something fancy
        try:
            super().__init__(dim=config.hidden_size // config.num_attention_heads, max_position_embeddings=config.max_position_embeddings, base=500000.0, device=device)
        except Exception:
            # Fallback if parent is extremely different, just init as nn.Module
            nn.Module.__init__(self)
        
        # 2. Extract dimensions explicitly
        self.head_dim = config.hidden_size // config.num_attention_heads
        base = getattr(config, "rope_theta", 500000.0)
        max_position_embeddings = config.max_position_embeddings
        
        # --- LLAMA 3.2 MATH OVERWRITE ---
        factor = 32.0
        low_freq_factor = 1.0
        high_freq_factor = 4.0
        original_max_position_embeddings = 8192

        # 3. Generate Frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32).to(device) / self.head_dim))
        
        # 4. Apply Scaling
        wavelen = 2 * math.pi / inv_freq
        low_freq_wavelen = original_max_position_embeddings / low_freq_factor
        high_freq_wavelen = original_max_position_embeddings / high_freq_factor
        
        new_wavelen = wavelen * factor
        ratio = (wavelen - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen)
        ratio = torch.clamp(ratio, 0.0, 1.0)
        
        final_inv_freq = (1.0 / wavelen) * (1 - ratio) + (1.0 / new_wavelen) * ratio
        
        # 5. Register Buffers
        self.register_buffer("inv_freq", final_inv_freq, persistent=False)
        
        # 6. Initialize Cache (Defining the method below avoids the AttributeError)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=device, dtype=torch.get_default_dtype())

    # --- MISSING METHOD 1: Explicitly define cache generation ---
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Register standard 2D cache [Seq, Dim]
        # We let the apply function handle broadcasting
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    # --- MISSING METHOD 2: Explicitly define forward ---
    def forward(self, x, seq_len=None):
        # Handle cache expansion if needed
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len + 1024, device=x.device, dtype=x.dtype)
            
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

# --- 2. SVD ATTENTION CLASS ---
class SVD_LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, ratio=1):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.ratio = ratio 
        self.layer_idx = 0 

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size {self.hidden_size} not divisible by num_heads {self.num_heads}")
            
        low_rank = int(self.hidden_size * self.ratio/2)
        
        self.q_u_proj = nn.Linear(low_rank, self.num_heads * self.head_dim, bias=False)
        self.q_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)
        self.k_u_proj = nn.Linear(low_rank, self.num_key_value_heads * self.head_dim, bias=False)
        self.k_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)
        self.v_u_proj = nn.Linear(low_rank, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)
        self.o_u_proj = nn.Linear(low_rank, self.hidden_size, bias=False)
        self.o_v_proj = nn.Linear(self.num_heads * self.head_dim, low_rank, bias=False)

        # Instantiate Hijacked Class
        self.rotary_emb = Llama32RotaryEmbedding(config=self.config)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_u_proj(self.q_v_proj(hidden_states))
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_u_proj(self.k_v_proj(hidden_states))
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_u_proj(self.v_v_proj(hidden_states))
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if hasattr(past_key_value, "get_seq_length"):
                kv_seq_len += past_key_value.get_seq_length(self.layer_idx)
            else:
                kv_seq_len += past_key_value[0].shape[-2]

        # --- ENSURE POSITION IDS ---
        if position_ids is None:
            position_ids = torch.arange(kv_seq_len, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
            position_ids = position_ids[:, -q_len:]

        # --- EXECUTE ROPE ---
        # 1. Get Embeddings (from our hijacked class)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        
        # 2. Apply (transformers function handles the 2D->4D broadcasting for us)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
             if isinstance(past_key_value, tuple):
                 key_states = torch.cat([past_key_value[0], key_states], dim=2)
                 value_states = torch.cat([past_key_value[1], value_states], dim=2)
                 past_key_value = (key_states, value_states)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = self.o_u_proj(self.o_v_proj(attn_output))

        return attn_output, None








# class RoPE(nn.Module):
#     """
#     This code is mostly taken from HF for Llama #TODO (add url)
#     Standard setup for models:
#     - Llama base=500000, scale_freqs=True
#     - llm-random base=10000, scale_freqs=False
#     """

#     # features are paired x_i, x_{i + d_head/2}
#     def __init__(
#         self,
#         dhead,
#         length,
#         base,
#         apply_freq_scaling,
#         factor,
#         low_freq_factor,
#         high_freq_factor,
#         original_max_position_embeddings,
#     ):
#         super().__init__()
#         self.dhead = dhead
#         self.length = length
#         self.base = base
#         self.apply_freq_scaling = apply_freq_scaling
#         self.factor = factor
#         self.low_freq_factor = low_freq_factor
#         self.high_freq_factor = high_freq_factor
#         self.original_max_position_embeddings = original_max_position_embeddings
#         self.register_freqs()

#     def register_freqs(self):
#         angle_exponents = (
#             torch.arange(0, self.dhead, 2, dtype=torch.int64).float() / self.dhead
#         )
#         angles = 1.0 / torch.pow(self.base, angle_exponents).reshape(1, -1)
#         if self.apply_freq_scaling:
#             angles = self.scale_freqs(angles)

#         angle_per_token = angles * torch.arange(0, self.length).reshape(-1, 1)
#         self.register_buffer(
#             "sin", torch.sin(angle_per_token).repeat(1, 2), persistent=False
#         )
#         self.register_buffer(
#             "cos", torch.cos(angle_per_token).repeat(1, 2), persistent=False
#         )

#     def scale_freqs(self, freqs):
#         factor = self.factor
#         low_freq_factor = self.low_freq_factor
#         high_freq_factor = self.high_freq_factor
#         old_context_len = self.original_max_position_embeddings

#         low_freq_wavelen = old_context_len / low_freq_factor
#         high_freq_wavelen = old_context_len / high_freq_factor

#         wavelen = 2 * math.pi / freqs
#         # wavelen < high_freq_wavelen: do nothing
#         # wavelen > low_freq_wavelen: divide by factor
#         inv_freq_llama = torch.where(wavelen > low_freq_wavelen, freqs / factor, freqs)
#         # otherwise: interpolate between the two, using a smooth factor
#         smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
#             high_freq_factor - low_freq_factor
#         )
#         smoothed_inv_freq = (
#             1 - smooth_factor
#         ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
#         is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
#         inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
#         return inv_freq_llama

#     def forward(self, x):
#         [y1, y2] = torch.chunk(x, chunks=2, dim=-1)
#         x_rotated = torch.cat([-y2, y1], dim=-1)
#         cos_scaler = self.cos[: x.shape[-2], :].to(x.device, dtype=x.dtype)
#         sin_scaler = self.sin[: x.shape[-2], :].to(x.device, dtype=x.dtype)
#         return x * cos_scaler + x_rotated * sin_scaler

# from torch.nn.attention import SDPBackend
# import torch.nn.functional as F

# def attention_mechanism(
#     query: torch.Tensor,
#     key: torch.Tensor,
#     value: torch.Tensor,
#     causal: bool,
# ):
#     # https://github.com/pytorch/pytorch/blob/ce503c1b40207dab770c28cbd4568cd9e105277b/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp#L556
#     with torch.nn.attention.sdpa_kernel(
#         [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
#     ):
#         return F.scaled_dot_product_attention(
#             query=query,
#             key=key,
#             value=value,
#             attn_mask=None,
#             is_causal=causal,
#         )


# class AttentionMechanism(nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         # raise Exception("Not supported AttentionMechanism - use pc version")
#         super().__init__(*args, **kwargs)

#     def forward(
#         self,
#         query: torch.Tensor,
#         key: torch.Tensor,
#         value: torch.Tensor,
#         causal: bool,
#     ):
#         return attention_mechanism(
#             query=query,
#             key=key,
#             value=value,
#             causal=causal,
#         )


# class RoPEAttention(nn.Module):
#     def __init__(
#         self,
#         config,
#         # dmodel,
#         # q_heads,
#         # kv_heads,
#         # seq_len,
#         # rope_base,
#         # rope_scale_freqs: bool,
#         # factor=32,
#         # low_freq_factor=1,
#         # high_freq_factor=4,
#         # original_max_position_embeddings=8192,
#     ):
#         super().__init__()
#         self.config = config
#         self.hidden_size = config.hidden_size
#         self.num_heads = config.num_attention_heads
#         self.head_dim = self.hidden_size // self.num_heads
#         self.num_key_value_heads = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else self.num_heads
#         self.num_key_value_groups = self.num_heads // self.num_key_value_heads
#         self.max_position_embeddings = config.max_position_embeddings
#         self.ratio = ratio 
#         self.layer_idx = 0 

#         if (self.head_dim * self.num_heads) != self.hidden_size:
#             raise ValueError(f"hidden_size {self.hidden_size} not divisible by num_heads {self.num_heads}")
            
#         low_rank = int(self.hidden_size * self.ratio/2)

#         self.q_u_proj = nn.Linear(low_rank, self.num_heads * self.head_dim, bias=False)
#         self.q_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)
#         self.k_u_proj = nn.Linear(low_rank, self.num_key_value_heads * self.head_dim, bias=False)
#         self.k_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)
#         self.v_u_proj = nn.Linear(low_rank, self.num_key_value_heads * self.head_dim, bias=False)
#         self.v_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)
#         self.o_u_proj = nn.Linear(low_rank, self.hidden_size, bias=False)
#         self.o_v_proj = nn.Linear(self.num_heads * self.head_dim, low_rank, bias=False)
#         self.attention_mechanism = AttentionMechanism()

#         self.q_heads = q_heads
#         self.kv_heads = kv_heads
#         self.dhead = self.q_proj.weight.shape[0] // self.q_heads
#         self.dmodel = dmodel

#         self.rope = RoPE(
#             dhead=self.dhead,
#             length=seq_len,
#             base=rope_base,
#             apply_freq_scaling=rope_scale_freqs,
#             factor=factor,
#             low_freq_factor=low_freq_factor,
#             high_freq_factor=high_freq_factor,
#             original_max_position_embeddings=original_max_position_embeddings,
#         )

#     def forward(self, x):
#         query_states = self.q_proj(x)
#         key_states = self.k_proj(x)
#         value_states = self.v_proj(x)

#         batch, seq_len = x.shape[:-1]
#         q = query_states.view(batch, seq_len, self.q_heads, -1).transpose(1, 2)
#         q = self.rope(q)
#         k = key_states.view(batch, seq_len, self.kv_heads, -1).transpose(1, 2)
#         k = self.rope(k)

#         v = value_states.view(batch, seq_len, self.kv_heads, -1).transpose(1, 2)

#         k = repeat_kv(k, self.q_heads // self.kv_heads)
#         v = repeat_kv(v, self.q_heads // self.kv_heads)
#         attention_output = self.attention_mechanism(
#             query=q, key=k, value=v, causal=True
#         )

#         output = self.o_proj(attention_output.transpose(1, 2).contiguous().flatten(-2))

#         return output



