from collections import OrderedDict
import re
import math
from omegaconf import OmegaConf
import torch.nn as nn
import torch
from hydra.utils import instantiate

from .model import AttentionMechanism, Linear
from transformers import AutoModelForCausalLM


class LlamaRoPE(nn.Module):
    # features are paired x_i, x_{i + d_head/2}
    def __init__(self, dhead, length, base=10000):
        super().__init__()
        self.dhead = dhead
        self.length = length
        angle_exponents = torch.arange(0, dhead, 2, dtype=torch.int64).float() / dhead
        angles = 1.0 / torch.pow(base, angle_exponents).reshape(1, -1)
        angles = self.scale_freqs(angles)

        angle_per_token = angles * torch.arange(0, length).reshape(-1, 1)
        self.register_buffer("sin", torch.sin(angle_per_token).repeat(1, 2))
        self.register_buffer("cos", torch.cos(angle_per_token).repeat(1, 2))

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
        # x shape (batch, n_heads, seq_len, dhead)
        [y1, y2] = torch.chunk(x, chunks=2, dim=-1)
        x_rotated = torch.cat([-y2, y1], dim=-1)
        cos_scaler = self.cos[: x.shape[-2], :].to(x.device)
        sin_scaler = self.sin[: x.shape[-2], :].to(x.device)
        return x * cos_scaler + x_rotated * sin_scaler


class LLamaFeedForward(nn.Module):
    def __init__(
        self,
        dmodel,
        dff,
        init_type: str,
        init_scale: float,
    ):
        super().__init__()
        self.ff_pre_act = Linear(
            dmodel, dff, bias=False, init_type=init_type, init_scale=init_scale
        )
        self.silu = nn.SiLU()
        self.ff_post_act = Linear(
            dff, dmodel, bias=False, init_type=init_type, init_scale=init_scale
        )
        self.gate = Linear(
            dmodel, dff, bias=False, init_type=init_type, init_scale=init_scale
        )

    def forward(self, x):
        gated = self.gate(x)
        gated = self.silu(gated)
        x = self.ff_pre_act(x)
        x = x * gated
        x = self.ff_post_act(x)
        return x


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


class LlamaAttention(nn.Module):
    def __init__(
        self,
        dmodel,
        q_heads,
        kv_heads,
        seq_len,
        causal,
        init_type: str,
        init_scale: float,
    ):
        super().__init__()
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.causal = causal
        self.head_dim = dmodel // self.q_heads

        self.q_proj = Linear(
            dmodel, dmodel, bias=False, init_type=init_type, init_scale=init_scale
        )
        self.k_proj = Linear(
            dmodel,
            self.kv_heads * self.head_dim,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.v_proj = Linear(
            dmodel,
            self.kv_heads * self.head_dim,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )

        self.o_proj = Linear(
            dmodel,
            dmodel,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.attention_mechanism = AttentionMechanism()
        self.rope = LlamaRoPE(dhead=self.head_dim, length=seq_len, base=500000)

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
            query=q, key=k, value=v, causal=self.causal
        )

        output = self.o_proj(attention_output.transpose(1, 2).contiguous().flatten(-2))

        return output


def remap_llamahf_state_dict_to_nano(llama_state_dict):
    remapped = {}
    for key, value in llama_state_dict.items():
        new_key = key

        # Embedding
        new_key = new_key.replace(
            "model.embed_tokens.weight", "embedding.embedding.weight"
        )

        # Final norm and lm head
        new_key = new_key.replace("model.norm.weight", "head.norm.weight")
        new_key = new_key.replace("lm_head.weight", "head.linear.weight")

        # Layers
        layer_match = re.match(r"model\.layers\.(\d+)\.(.*)", new_key)
        if layer_match:
            layer_num = layer_match.group(1)
            sub_key = layer_match.group(2)

            # Attention projections
            sub_key = sub_key.replace(
                "self_attn.q_proj.weight", f"attention_layer.layer.q_proj.weight"
            )
            sub_key = sub_key.replace(
                "self_attn.k_proj.weight", f"attention_layer.layer.k_proj.weight"
            )
            sub_key = sub_key.replace(
                "self_attn.v_proj.weight", f"attention_layer.layer.v_proj.weight"
            )
            sub_key = sub_key.replace(
                "self_attn.o_proj.weight", f"attention_layer.layer.o_proj.weight"
            )

            # Attention norms
            sub_key = sub_key.replace(
                "input_layernorm.weight", "attention_layer.norm.weight"
            )
            sub_key = sub_key.replace(
                "post_attention_layernorm.weight", "ff_layer.norm.weight"
            )

            # MLP
            sub_key = sub_key.replace(
                "mlp.up_proj.weight", "ff_layer.layer.ff_pre_act.weight"
            )
            sub_key = sub_key.replace(
                "mlp.gate_proj.weight", "ff_layer.layer.gate.weight"
            )
            sub_key = sub_key.replace(
                "mlp.down_proj.weight", "ff_layer.layer.ff_post_act.weight"
            )

            new_key = f"encoder.blocks.{layer_num}.{sub_key}"

        remapped[new_key] = value

    return OrderedDict(remapped)


def copy_llama_model_weights_from_HF(model: nn.Module, path: str):

    hf_model = AutoModelForCausalLM.from_pretrained(path)

    llama_state_dict = hf_model.state_dict()

    remapped_state_dict = remap_llamahf_state_dict_to_nano(llama_state_dict)

    model.load_state_dict(remapped_state_dict)


def save_pretrained_llama_as_nano(cfg: OmegaConf, metric_logger=None):

    with torch.device("meta"):
        model = instantiate(cfg.model)

    hf_model = AutoModelForCausalLM.from_pretrained(cfg.trainer.checkpoint.load.path)
    nano_sd = remap_llamahf_state_dict_to_nano(hf_model.state_dict())
    model.load_state_dict(nano_sd, strict=False, assign=True)

    torch.save(model.state_dict(), cfg.trainer.checkpoint.save.path)

    return None, None, None, None, None
