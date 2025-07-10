from collections import OrderedDict
import os
import re
import sys
import torch.nn as nn
from typing import Callable
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.attention import SDPBackend
from torch.nn.init import trunc_normal_
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from src.definitions import AttentionConfig, Common, TowerConfig
from torch.nn import (
    LayerNorm as LayerNorm,
)  # used by FSDP, but it keeps getting removed during file formatting
import logging


logger = logging.getLogger(__name__)



def init_kaiming_uniform(shape, fan_in, scale, dtype=torch.float32):
    range_ = scale * (3 / fan_in) ** 0.5
    return torch.zeros(shape, dtype=dtype).uniform_(-range_, range_)


def init_truncated_normal(shape, fan_in, scale, dtype=torch.float32):
    std = (scale / fan_in) ** 0.5
    low = -2 * scale
    high = 2 * scale
    t = torch.zeros(shape, dtype=dtype)
    return trunc_normal_(t, mean=0.0, std=std, a=low, b=high)


def init_truncated_normal_fixed(shape, fan_in, scale, dtype=torch.float32):
    std = scale * (1 / fan_in) ** 0.5
    low = -2 * std
    high = 2 * std
    t = torch.zeros(shape, dtype=dtype)
    return trunc_normal_(t, mean=0.0, std=std, a=low, b=high)

def init_zeros(shape, fan_in=None, scale=None, dtype=torch.float32):
    assert scale is None
    assert fan_in is None
    return torch.zeros(shape, dtype=dtype)


def get_init_weight(shape, fan_in, init_type: str, scale, dtype=torch.float32):
    init_types = {
        "kaiming_uniform": init_kaiming_uniform,
        "truncated_normal": init_truncated_normal,
        "truncated_normal_fixed": init_truncated_normal_fixed,
        "zeros": init_zeros, # for later smart-init - eg. prunned base weights or dim reduction projections
    }

    if init_type not in init_types:
        raise ValueError(f"Unknown init_type: {init_type}")

    return init_types[init_type](shape=shape, fan_in=fan_in, scale=scale, dtype=dtype)

class RMSNorm(nn.Module):
    def __init__(self, dmodel, eps=1e-5):
        super().__init__()
        self.eps = eps

        self.g = nn.Parameter(torch.ones(dmodel))
        self.b = nn.Parameter(torch.zeros(dmodel))

    def forward(self, x):
        norm = torch.mean(x**2, dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.g + self.b


class Residual(nn.Module):
    def __init__(self, layer):
        super(Residual, self).__init__()
        self.layer = layer
        self.metric_logger = None

    def set_metric_logger(self, metric_logger):
        self.metric_logger = metric_logger

    def forward(self, x):
        out = self.layer(x)
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


def PreNormBlock(dmodel, layer, name, norm_class=nn.LayerNorm):
    return Residual(
        nn.Sequential(
            OrderedDict(
                [
                    ("pre_norm", norm_class(dmodel)),
                    (f"{name}", layer),
                ]
            )
        )
    )


def PostNormBlock(dmodel, layer, name, norm_class=nn.LayerNorm):
    return nn.Sequential(
        OrderedDict(
            [
                (f"{name}", Residual(layer)),
                ("post_norm", norm_class(dmodel)),
            ]
        )
    )


def TokenEmbedding(
    vocab_size,
    embedding_dim,
    init_type: str,
    init_scale: float,
):
    weight = get_init_weight(
        shape=(vocab_size, embedding_dim),
        fan_in=1,
        init_type=init_type,
        scale=init_scale,
    )
    return nn.Embedding(vocab_size, embedding_dim, _weight=weight)


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        max_length,
        embedding_dim,
        init_type: str,
        init_scale: float,
    ):
        super(PositionalEmbedding, self).__init__()
        self.layer = nn.Embedding(max_length, embedding_dim)
        default_weight = self.layer.weight.data
        self.layer.weight.data = get_init_weight(
            shape=default_weight.shape,
            fan_in=1,
            init_type=init_type,
            scale=init_scale,
            dtype=default_weight.dtype,
        )
        # TODO(jaszczur): add initialization as positional encoding

    def forward(self, x):
        positions = torch.arange(0, x.shape[-1], device=x.device)
        positions = positions * torch.ones_like(x)
        embeddings = self.layer(positions)
        return embeddings



class Aggregate(nn.Module): #dev TODO duplicate class
    def __init__(self, function, *layers):
        super(Aggregate, self).__init__()
        self.function = function
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        result = None
        for layer in self.layers:
            if result is None:
                result = layer(x)
            else:
                result = self.function(result, layer(x))
        return result


class Aggregate(nn.Module): #dev TODO not used class
    def __init__(
        self,
        function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *layers: nn.Module,
    ):
        super(Aggregate, self).__init__()
        self.function = function
        self.layers = nn.ModuleList(layers)
        assert len(self.layers) > 0, "Aggregate must have at least one layer"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.layers[0](x)
        for layer in self.layers[1:]:
            result = self.function(result, layer(x))
        return result


class Linear(nn.Linear):
    def __init__(self, *args, init_type, init_scale, **kwargs):
        if "bias" not in kwargs:
            kwargs["bias"] = False
        super().__init__(*args, **kwargs)
        self.weight.data = get_init_weight(
            shape=self.weight.shape,
            fan_in=self.in_features,
            init_type=init_type,
            scale=init_scale,
            dtype=self.weight.dtype,
        )


class EmbeddingLayer(Aggregate):
    def __init__(self, *layers):
        super(EmbeddingLayer, self).__init__((lambda x, y: x + y), *layers)


class PredictionHead(nn.Module):
    def __init__(
        self, embedding_dim, output_size, init_type, init_scale, use_layer_norm: bool
    ):
        super(PredictionHead, self).__init__()

        layers = OrderedDict()
        if use_layer_norm:
            layers["head_norm"] = nn.LayerNorm(embedding_dim)
        layers["head"] = Linear(
            embedding_dim, output_size, init_type=init_type, init_scale=init_scale
        )

        self.unembedding = nn.Sequential(layers)

    def forward(self, x):
        return self.unembedding(x)


def FeedForward(
    dmodel,
    dff,
    init_type: str,
    init_scale: float,
):
    return nn.Sequential(
        OrderedDict(
            [
                (
                    "logging_ff_pre_relu",
                    Linear(
                        dmodel,
                        dff,
                        bias=True,
                        init_type=init_type,
                        init_scale=init_scale,
                    ),
                ),
                ("relu", nn.ReLU()),
                (
                    "logging_ff_post_relu",
                    Linear(
                        dff,
                        dmodel,
                        bias=True,
                        init_type=init_type,
                        init_scale=init_scale,
                    ),
                ),
            ]
        )
    )


def attention_mechanism(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool,
):
    # https://github.com/pytorch/pytorch/blob/ce503c1b40207dab770c28cbd4568cd9e105277b/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp#L556
    with torch.nn.attention.sdpa_kernel(
        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    ):
        return F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=None,
            is_causal=causal,
        )


class AttentionMechanism(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool,
    ):
        return attention_mechanism(
            query=query,
            key=key,
            value=value,
            causal=causal,
        )


class Attention(nn.Module):
    def __init__(
        self,
        dmodel,
        heads,
        causal,
        init_type: str,
        init_scale: float,
    ):
        super(Attention, self).__init__()

        self.heads = heads
        self.causal = causal

        self.input_projection = Linear(
            dmodel,
            3 * dmodel,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.output_projection = Linear(
            dmodel,
            dmodel,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.attention_mechanism = AttentionMechanism()

    def forward(self, x):
        projected = self.input_projection(x)

        batch, seq_len = x.shape[:-1]
        q_chunk, k_chunk, v_chunk = torch.chunk(projected, chunks=3, dim=-1)
        q = q_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)
        k = k_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)
        v = v_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)

        attention_output = self.attention_mechanism(
            query=q, key=k, value=v, causal=self.causal
        )

        output = self.output_projection(attention_output.transpose(1, 2).flatten(-2))

        return output
    
def get_vanilla_embedding(common):
    return EmbeddingLayer(
        TokenEmbedding(
            common.vocab_size,
            common.dmodel,
            init_type=common.init_type,
            init_scale=common.init_scale,
        ),
        PositionalEmbedding(
            common.sequence_length,
            common.dmodel,
            init_type=common.init_type,
            init_scale=common.init_scale,
        ),
    )
