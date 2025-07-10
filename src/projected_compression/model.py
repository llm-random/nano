# Add new transformer modules for PC compressor model (with residual weights, projections and base model weights):
# - embedding
# - head (unembedding)
# - feed forward
# - attention
# -- each element shoud take into account layer norm weights and bias, layernorm weights and biases should have shared projections as weights they normaizes output from!

# modules for activation based weights importance assesment

from types import NoneType
from collections import OrderedDict

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

from core.model import AttentionMechanism, EmbeddingLayer, Linear, PositionalEmbedding, get_init_weight



class ProjectedLinear(nn.Module):
    __constants__ = ["in_features", "out_features", "base_in_features", "base_out_features"]
    in_features: int
    out_features: int
    base_in_features: int
    base_out_features: int
    __project_in: bool
    __project_out: bool
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        base_in_features: int,
        base_out_features: int,
        project_in: bool = True,
        project_out: bool = True,
        bias: bool = False, # Not implemented
        device=None,
        dtype=None,
    ) -> None:
        assert not bias, "Projected Linear does not support bias"
        self.register_parameter("bias", None)
        
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.base_in_features = base_in_features
        self.base_out_features = base_out_features
        self.__project_in = project_in
        self.__project_out = project_out
        if project_in: 
            self.projection_in_weight = nn.Parameter(
                torch.zeros((base_in_features, in_features), **factory_kwargs)
            )
        else:
            self.register_parameter("projection_in_weight", None)
        self.base_weight = nn.Parameter(
            torch.zeros((base_out_features, base_in_features), **factory_kwargs)
        )
        if project_out:
            self.projection_out_weight = nn.Parameter(
                torch.zeros((out_features, base_out_features), **factory_kwargs)
            )
        else:
            self.register_parameter("projection_out_weight", None)

    def reset_parameters(self) -> None:
        raise NotImplemented("Projected Linear does not need it - propably")

    def forward(self, input: Tensor) -> Tensor:
        # gradient magic happens here - PC optimization
        if self.__project_in:
            projected_base_weights = self.projection_in_weight @ self.base_weight
        else:
            projected_base_weights = self.base_weight
        
        if self.__project_out:
            projected_base_weights = projected_base_weights @ self.projection_out_weight

        return F.linear(input, projected_base_weights, self.bias) # bias always None

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, base_in_features = {self.__project_in}, {self.base_in_features}, base_out_features = {self.__project_out}, {self.base_out_features}" # no bias


def ProjectedFeedForward(
    dmodel,
    dff,
    base_dmodel,
    base_dff,
    init_type: str = "zeros",
    init_scale: NoneType = None,
):
    assert init_type == "zeros", "PC does not use do weights init in Linear init"
    assert init_scale is None, "PC does not use do weights init in Linear init"

    return nn.Sequential(
        OrderedDict(
            [
                (
                    "logging_ff_pre_relu",
                    ProjectedLinear(
                        dmodel,
                        dff,
                        base_dmodel,
                        base_dff,
                    ),
                ),
                ("relu", nn.ReLU()),
                (
                    "logging_ff_post_relu",
                    ProjectedLinear(
                        dff,
                        dmodel,
                        base_dff,
                        base_dmodel,
                    ),
                ),
            ]
        )
    )


class ProjectedAttention(nn.Module):
    def __init__(
        self,
        dmodel,
        base_dmodel,
        heads, # number does not change for base model*, heads are dropped in separate hard-pruning of base model 
        causal, # dev TODO - add description; docustring
        init_type: str = "zeros",
        init_scale: NoneType = None,
    ):
        assert init_type == "zeros", "PC does not use do weights init in Linear init"
        assert init_scale is None, "PC does not use do weights init in Linear init"

        super(ProjectedAttention, self).__init__()

        self.heads = heads
        self.causal = causal

        self.input_projection = ProjectedLinear(
            dmodel,
            3 * dmodel,
            base_dmodel,
            3 * base_dmodel,
            project_out=False, # hidden att pruned by pre-pruning base model
        )
        self.output_projection = ProjectedLinear(
            dmodel,
            dmodel,
            base_dmodel,
            base_dmodel,
            project_in=False, # hidden att pruned by pre-pruning base model
        )
        self.attention_mechanism = AttentionMechanism()

    def forward(self, x):
        projected = self.input_projection(x) # projected hidden dimension same as base model

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


class ProjectedPredictionHead(nn.Module):
    def __init__(
        self, 
        embedding_dim, 
        base_embedding_dim, 
        output_size, 
        use_layer_norm: bool, 
        init_scale: NoneType = None, 
        init_type: str = "zeros", 
    ):
        assert init_type == "zeros", "PC does not use do weights init in Linear init"
        assert init_scale is None, "PC does not use do weights init in Linear init"

        super(ProjectedPredictionHead, self).__init__()

        layers = OrderedDict()
        if use_layer_norm:
            layers["head_norm"] = nn.LayerNorm(embedding_dim)
        layers["head"] = ProjectedLinear(
            embedding_dim, 
            output_size, 
            base_embedding_dim, 
            output_size, 
            project_out=False
        )

        self.unembedding = nn.Sequential(layers)

    def forward(self, x):
        return self.unembedding(x)
    

class ProjectedTokenEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        base_embedding_dim,
        init_type: str = "zeros", 
        init_scale: NoneType = None, 
    ):
        assert init_type == "zeros", "PC does not use do weights init in Linear init"
        assert init_scale is None, "PC does not use do weights init in Linear init"

        super(ProjectedTokenEmbedding, self).__init__()

        layers = OrderedDict()
        layers["head"] = ProjectedLinear(
            vocab_size, 
            embedding_dim, 
            vocab_size, 
            base_embedding_dim, 
            project_in=False
        )

        self.embedding = nn.Sequential(layers)

    def forward(self, x):
        return self.embedding(x)


# GETTERS FOR CONFIGS


def get_projected_embedding(common): # dev define types pc_common ?
    return EmbeddingLayer(
        ProjectedTokenEmbedding(
            common.vocab_size,
            common.dmodel,
            common.base_dmodel,
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


def get_projected_head(common): # dev define types pc_common ?
    return  ProjectedPredictionHead(
        common.dmodel,
        common.base_dmodel,
        common.vocab_size,
        init_type=common.init_type,
        init_scale=common.init_scale,
        use_layer_norm=common.head_norm,
    )

