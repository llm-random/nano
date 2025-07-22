from functools import partial
import os
import torch
import torch.nn as nn
from typing import Callable, Optional
from torch.nn import (
    LayerNorm as LayerNorm,
)  # used by FSDP, but it keeps getting removed during file formatting

from torch.nn.modules.normalization import RMSNorm as RMSNorm
from torchtune.modules.position_embeddings import RotaryPositionalEmbeddings as RotaryPositionalEmbeddings

from src.core.llama import LlamaRoPE, repeat_kv
from src.core.model import AttentionMechanism, Residual
from torch.nn.init import trunc_normal_
import torch.distributed as dist

def llm_random_weight_init(fan_in, scale):
    std = scale * (1 / fan_in) ** 0.5
    low = -2 * std
    high = 2 * std
    return partial(trunc_normal_, mean=0.0, std=std, a=low, b=high)

dummy_weight_init = lambda _: trunc_normal_

class TransformerEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int, 
        dmodel: int, 
        init_fn: Optional[Callable]
    ):
        super().__init__()
        weight = torch.empty(
                vocab_size, 
                dmodel,
                dtype=torch.float32
            )
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


class LlamaAttention(nn.Module):
    def __init__(
        self,
        dmodel,
        q_heads,
        kv_heads,
        seq_len,
        linear_fn
    ):
        super().__init__()
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = dmodel // self.q_heads

        self.q_proj = linear_fn(dmodel, dmodel)
        self.k_proj = linear_fn(dmodel, self.kv_heads * self.head_dim)
        self.v_proj = linear_fn(dmodel, self.kv_heads * self.head_dim)
        self.output_projection = linear_fn(dmodel, dmodel)
        self.attention_mechanism = AttentionMechanism()

        self.rope = LlamaRoPE(
            dhead=self.head_dim,
            length=seq_len,
            base=500000
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

        output = self.output_projection(attention_output.transpose(1, 2).contiguous().flatten(-2))

        return output



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
        self.ff_layer =  Residual(
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
        self.blocks = nn.ModuleList([ block_fn(i)
            for i in range(n_blocks)
        ])

    def forward(self, x):
        return self.blocks(x)


class TransformerHead(nn.Module):
    def __init__(
        self, dmodel, vocab_size, linear_fn: Callable, norm_fn: Callable
    ):
        super().__init__()
        self.norm = norm_fn()
        self.linear = linear_fn(dmodel, vocab_size)
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
        self.embedding_layer = embedding
        self.encoder = encoder
        self.head = head

    def forward(self, *args, **kwargs):
        x = self.embedding_layer(*args, **kwargs)
        x = self.encoder(x)
        x = self.head(x)
        return x