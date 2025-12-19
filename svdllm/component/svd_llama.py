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

# --- 1. HELPER: Repeat KV for GQA ---
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# --- 2. YOUR WORKING ROPE CLASS (Llama 3.2 Logic) ---
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
        # x shape: [batch, heads, seq, dim]
        [y1, y2] = torch.chunk(x, chunks=2, dim=-1)
        x_rotated = torch.cat([-y2, y1], dim=-1)
        
        # We slice based on the -2 dimension (seq_len)
        cos_scaler = self.cos[: x.shape[-2], :].to(x.device, dtype=x.dtype)
        sin_scaler = self.sin[: x.shape[-2], :].to(x.device, dtype=x.dtype)
        
        return x * cos_scaler + x_rotated * sin_scaler

# --- 3. SVD ATTENTION CLASS (Using your exact flow) ---
class SVD_LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, ratio=1):
        print(f"init in SVD_LlamaAttention --------------------------------------------------------------") #dev
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # GQA Settings
        self.num_key_value_heads = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.ratio = ratio

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size {self.hidden_size} not divisible by num_heads {self.num_heads}")

        low_rank = int(self.hidden_size * self.ratio / 2)

        # 1. Initialize SVD Projections
        self.q_u_proj = nn.Linear(low_rank, self.num_heads * self.head_dim, bias=False)
        self.q_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)

        self.k_u_proj = nn.Linear(low_rank, self.num_key_value_heads * self.head_dim, bias=False)
        self.k_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)

        self.v_u_proj = nn.Linear(low_rank, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)

        self.o_u_proj = nn.Linear(low_rank, self.hidden_size, bias=False)
        self.o_v_proj = nn.Linear(self.num_heads * self.head_dim, low_rank, bias=False)

        # 2. Initialize RoPE with Llama 3.2 Defaults
        rope_base = getattr(config, "rope_theta", 500000.0)
        use_scaling = True 
        
        self.rope = RoPE(
            dhead=self.head_dim,
            length=self.max_position_embeddings,
            base=rope_base,
            apply_freq_scaling=use_scaling,
            factor=32.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_position_embeddings=8192,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        print(f"forward in SVD_LlamaAttention --------------------------------------------------------------") #dev
        print(f"kwargs {kwargs}") #dev
        print(f"attention_mask {attention_mask}") #dev
        
        bsz, q_len, _ = hidden_states.size()

        # 1. Projections (SVD)
        query_states = self.q_u_proj(self.q_v_proj(hidden_states))
        key_states = self.k_u_proj(self.k_v_proj(hidden_states))
        value_states = self.v_u_proj(self.v_v_proj(hidden_states))

        # 2. Reshape & Transpose: [batch, heads, seq, dim]
        q = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 3. Apply RoPE (In-Place on Q and K)
        q = self.rope(q)
        k = self.rope(k)

        # 5. Repeat KV for GQA
        k = repeat_kv(k, self.num_heads // self.num_key_value_heads)
        v = repeat_kv(v, self.num_heads // self.num_key_value_heads)

        # 6. Attention Mechanism (Standard Scaled Dot Product)
        # q, k, v are now all [bsz, num_heads, seq_len, head_dim]
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        # 7. Output Projection
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = self.o_u_proj(self.o_v_proj(attn_output))


        return attn_output, None