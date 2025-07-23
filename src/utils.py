
# those functions generate circular imports if places in core/, when they use model modules from /projected_compression/model.py and projected_compression/models.py ablate its classes from core modules  

from collections import OrderedDict
import os
import re
import sys
from typing import Callable, Union

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from src.core.model import Attention, FeedForward, PostNormBlock, PreNormBlock, PredictionHead, RMSNorm
from src.definitions import AttentionConfig, Common, CommonCompression, TowerConfig
from src.projected_compression.model import ProjectedAttention, ProjectedFeedForward


def get_norm_class_function(norm_class_mode: str):
    norm_classes = {
        "layer_norm": nn.LayerNorm,
        "rms_norm": RMSNorm,
    }

    if norm_class_mode not in norm_classes:
        raise NotImplementedError(
            f"Norm class {norm_class_mode} not implemented. Supported types are: {list(norm_classes.keys())}"
        )

    return norm_classes[norm_class_mode]


def get_residual_function(
    residual_mode: str, dmodel: int, norm_class_mode: str
) -> Callable[[], nn.Module]:
    norm_class = get_norm_class_function(norm_class_mode)
    residual_layers = {
        "pre_norm": lambda layer, name: PreNormBlock(
            dmodel, layer, name, norm_class=norm_class
        ),
        "post_norm": lambda: PostNormBlock(dmodel, norm_class=norm_class),
    }

    if residual_mode not in residual_layers:
        raise NotImplementedError(
            f"Unsupported residual_mode: {residual_mode}. Supported modes are: {list(residual_layers.keys())}"
        )

    return residual_layers[residual_mode]


def get_attention_function(
    common: Union[Common, CommonCompression],
    attention_config: AttentionConfig,
) -> Callable[[], nn.Module]:
    causal = common.model_type == "gpt"

    attention_functions = {
        "vanilla": lambda: Attention(
            dmodel=common.dmodel,
            heads=attention_config.n_heads,
            causal=causal,
            init_type=common.init_type,
            init_scale=common.init_scale,
        ),
        "pc_vanilla": lambda: ProjectedAttention(
            dmodel=common.dmodel,
            base_dmodel=common.base_dmodel,
            heads=attention_config.n_heads,
            causal=causal,
            init_type=common.init_type,
            init_scale=common.init_scale,
        ),
        
    }

    if attention_config.mode not in attention_functions:
        raise ValueError(
            f"Unsupported attention_mode: {attention_config.mode}. Supported modes are: {list(attention_functions.keys())}"
        )

    return attention_functions[attention_config.mode]


def get_ff_layer_function( 
    common: Union[Common, CommonCompression],
    ff_mode: str,
) -> Callable[[], nn.Module]:

    ff_functions = {
        "vanilla": lambda: FeedForward(
            common.dmodel,
            common.dff,
            init_type=common.init_type,
            init_scale=common.init_scale,
        ),
        "vanilla": lambda: ProjectedFeedForward(
            common.dmodel,
            common.dff,
            common.base_dmodel,
            common.base_dff,
            init_type=common.init_type,
            init_scale=common.init_scale,
        ),
    }

    if ff_mode not in ff_functions:
        raise ValueError(
            f"Unsupported ff_mode: {ff_mode}. Supported modes are: {list(ff_functions.keys())}"
        )

    return ff_functions[ff_mode]

def get_classes_from_globals(names):
    return [globals().get(name) for name in names]


def wrap_model(model, fsdp_config):
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


class TransformerBlock(nn.Module):
    def __init__(
        self,
        common,
        block_config,
    ):
        super(TransformerBlock, self).__init__()
        residual_fn = get_residual_function(
            block_config.residual_mode, common.dmodel, block_config.norm_class_mode
        )

        attention_function = get_attention_function(common, block_config.attention)

        ff_layer = get_ff_layer_function(
            common,
            block_config.feedforward.mode,
        )

        residual_layers = [
            (
                "residual_attention",
                residual_fn(layer=attention_function(), name="attention"),
            ),
            (
                "residual_feedforward",
                residual_fn(layer=ff_layer(), name="feedforward"),
            ),
        ]
        self.block = nn.Sequential(OrderedDict(residual_layers))

    def forward(self, x):
        return self.block(x)


class TransformerTower(nn.Module):
    def __init__(
        self,
        common: Common,
        tower_config: TowerConfig,
    ):
        super().__init__()
        blocks = [
            (
                f"block_{i}",
                TransformerBlock(
                    common,
                    tower_config.block_config,
                ),
            )
            for i in range(tower_config.n_blocks)
        ]
        self.blocks = nn.Sequential(OrderedDict(blocks))

    def forward(self, x):
        return self.blocks(x)


class LLM(nn.Module):
    def __init__(
        self,
        embedding,
        head,
        common: Common,
        tower_config: TowerConfig,
    ):
        super(LLM, self).__init__()

        self.embedding_layer = embedding

        self.encoder = TransformerTower(
            common=common,
            tower_config=tower_config,
        )

        # self.head = PredictionHead(
        #     common.dmodel,
        #     common.vocab_size,
        #     init_type=common.init_type,
        #     init_scale=common.init_scale,
        #     use_layer_norm=common.head_norm,
        # )
        
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