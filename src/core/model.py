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
from torch.nn import (
    LayerNorm as LayerNorm,
)  # used by FSDP, but it keeps getting removed during file formatting

from torch.nn.modules.normalization import RMSNorm as RMSNorm
from torchtune.modules.position_embeddings import RotaryPositionalEmbeddings as RotaryPositionalEmbeddings
from torch.nn.parallel import DistributedDataParallel as DDP
import logging

from src.core.utils import find_layers
from src.projected_compression.pruning import generate_structured_prune_mask, generate_unstructured_prune_mask

logger = logging.getLogger(__name__)


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


class TransformerBlock(nn.Module):
    def __init__(
        self,
        residual_fn,
        attention_fn,
        ff_layer_fn,
    ):
        super(TransformerBlock, self).__init__()
        residual_layers = [
            (
                "residual_attention",
                residual_fn(layer=attention_fn(), name="attention"),
            ),
            (
                "residual_feedforward",
                residual_fn(layer=ff_layer_fn(), name="feedforward"),
            ),
        ]
        self.block = nn.Sequential(OrderedDict(residual_layers))

    def forward(self, x):
        return self.block(x)


class TransformerTower(nn.Module):
    def __init__(
        self,
        block_fn: Callable[[], nn.Module],
        n_blocks: int,
    ):
        super().__init__()
        blocks = [
            (
                f"block_{i}",
                block_fn()
            )
            for i in range(n_blocks)
        ]
        self.blocks = nn.Sequential(OrderedDict(blocks))

    def forward(self, x):
        return self.blocks(x)


class Aggregate(nn.Module):
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


class Aggregate(nn.Module):
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

    def prune_weights_structured(self, prune_n: int, prune_m: int):
        pruning_mask = generate_structured_prune_mask(self.weight, prune_n, prune_m)
        self.weight[pruning_mask] = 0

    def prune_weights_unstructured(self, sparsity_ratio: float):
        pruning_mask = generate_unstructured_prune_mask(self.weight, sparsity_ratio)
        self.weight[pruning_mask] = 0            


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
            layers["head_norm"] = RMSNorm(embedding_dim)
        layers["head"] = Linear(
            embedding_dim, output_size, init_type=init_type, init_scale=init_scale
        )

        self.unembedding = nn.Sequential(layers)

    def forward(self, x):
        return self.unembedding(x)


class LLM(nn.Module):
    def __init__(
        self,
        embedding: nn.Module,
        tower: nn.Module,
        head: nn.Module,
    ):
        super(LLM, self).__init__()

        self.embedding_layer = embedding
        self.encoder = tower
        self.head = head

        self._add_metric_log_names()

    def _add_metric_log_names(self):
        def _get_metric_log_name(name: str):
            meaningful_regex = ["block_\\d+", "attention", "feedforward", "residual"]
            module_names = name.split(".")
            meaningful_names = [
                module_name
                for module_name in module_names
                if any(re.search(pattern, module_name) for pattern in meaningful_regex)
            ]
            return "/".join(meaningful_names)

        for name, model in self.named_modules():
            model.log_name = _get_metric_log_name(name)

    def forward(self, *args, **kwargs):
        x = self.embedding_layer(*args, **kwargs)
        x = self.encoder(x)
        x = self.head(x)
        return x
    

    def prune(self):
        hehe = find_layers(self, layers=[Linear])
        print(hehe)
        for name, layer in hehe:
            print(f"Pruning {name}")
            layer.prune_weights_structured(1, 2)



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


def get_init_weight(shape, fan_in, init_type: str, scale, dtype=torch.float32):
    init_types = {
        "kaiming_uniform": init_kaiming_uniform,
        "truncated_normal": init_truncated_normal,
        "truncated_normal_fixed": init_truncated_normal_fixed,
    }

    if init_type not in init_types:
        raise ValueError(f"Unknown init_type: {init_type}")

    return init_types[init_type](shape=shape, fan_in=fan_in, scale=scale, dtype=dtype)

def get_vanilla_embedding(vocab_size, dmodel, init_type, init_scale, sequence_length):
    return EmbeddingLayer(
        TokenEmbedding(
            vocab_size,
            dmodel,
            init_type=init_type,
            init_scale=init_scale,
        ),
        PositionalEmbedding(
            sequence_length,
            dmodel,
            init_type=init_type,
            init_scale=init_scale,
        ),
    )


def get_classes_from_globals(names):
    return [globals().get(name) for name in names]


def wrap_model_fsdp(model, fsdp_config):

    classes_to_wrap = get_classes_from_globals(fsdp_config.modules_to_wrap)
    print(f"Wrapping model with classes: {classes_to_wrap}")
    igonore_mixed_precision_classes = get_classes_from_globals(
        fsdp_config.mixed_precision.ignored_classes
    )
    print(f"Ignoring mixed precision for classes: {igonore_mixed_precision_classes}")
    mixed_precision_dtype = getattr(
        sys.modules["torch"], fsdp_config.mixed_precision.dtype
    )
    print(f"Using mixed precision dtype: {mixed_precision_dtype}")

    wrapped_model = FSDP(
        model,
        device_id=int(os.environ["RANK"]),
        mixed_precision=MixedPrecision(
            param_dtype=mixed_precision_dtype,
            cast_forward_inputs=True,
            _module_classes_to_ignore=igonore_mixed_precision_classes,
        ),
        auto_wrap_policy=ModuleWrapPolicy(classes_to_wrap),
    )
    return wrapped_model


def wrap_model_distributed(model, distributed_config):
    if distributed_config is not None:
        if torch.cuda.is_available():
            model = wrap_model_fsdp(model, distributed_config.fsdp)
        else:
            logger.info("FSDP is not supported with CPU. Running DDP instead")
            model = DDP(model)
    return model