from functools import partial
import math
import os
import torch
import torch.nn as nn
from typing import Callable, Optional
from torch.nn import (
    LayerNorm as LayerNorm,
)  # used by FSDP, but it keeps getting removed during file formatting

from torch.nn.modules.normalization import RMSNorm as RMSNorm
from torchtune.modules.position_embeddings import (
    RotaryPositionalEmbeddings as RotaryPositionalEmbeddings,
)

from torch.nn import Embedding as Embedding

from src.core.llama import repeat_kv
from src.core.model import AttentionMechanism
from torch.nn.init import trunc_normal_
from torch import zeros as zeros
import torch.distributed as dist
import torch.nn.functional as F

# The generic transformer lives in src.core.model now; these names are re-exported
# so the remaining projected-compression code and its configs keep resolving until
# the package is removed.
from src.core.model import (  # noqa: F401
    LLM,
    Linear,
    Residual,
    RoPE,
    RoPEAttention,
    TransformerBlock,
    TransformerEmbedding,
    TransformerEncoder,
    TransformerHead,
    llm_random_weight_init,
)
from src.core.model import GatedFeedForward as ProjectedLlamaFeedForward  # noqa: F401


# linear takes partial function which returns init_fn upon giving fan_in as an input, but sometimes it does not depend on fan_in
dummy_weight_init = lambda _: trunc_normal_
dummy_zeros = lambda _: zeros


class FeedForward(nn.Module):
    def __init__(self, ff_pre_act_fn, ff_post_act_fn):
        super().__init__()
        self.relu = nn.ReLU()
        self.ff_pre_act = ff_pre_act_fn()
        self.ff_post_act = ff_post_act_fn()

    def forward(self, x):
        x = self.ff_pre_act(x)
        x = self.relu(x)
        x = self.ff_post_act(x)
        return x


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


class ProjectedLinear(nn.Module):
    __constants__ = [
        "result_in_features",
        "result_out_features",
        "base_in_features",
        "base_out_features",
    ]
    result_in_features: Optional[int]
    result_out_features: Optional[int]
    base_in_features: int
    base_out_features: int

    def __init__(
        self,
        result_in_features: Optional[int],
        result_out_features: Optional[int],
        base_in_features: int,
        base_out_features: int,
        device=None,
        dtype=None,
    ) -> nn.Module:

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.result_in_features = result_in_features
        self.result_out_features = result_out_features
        self.base_in_features = base_in_features
        self.base_out_features = base_out_features
        self.initialized_compression = False

        self.weight = nn.Parameter(
            torch.rand((base_out_features, base_in_features), **factory_kwargs)
        )
        self.projection_in_weight = None
        self.projection_out_weight = None
        self.auxiliary_weight = None

    def init_projections(
        self,
        proj_in_topk_indices: Optional[torch.Tensor],
        proj_out_topk_indices: Optional[torch.Tensor],
        factory_kwargs={},
    ):
        if proj_in_topk_indices is None:
            assert (
                self.projection_in_weight is None
            ), "Projection 'in' is decalred, but not passed."
        else:
            assert (
                len(proj_in_topk_indices) == self.result_in_features
            ), "projection 'in' dimension mismatch."

        if proj_out_topk_indices is None:
            assert (
                self.projection_out_weight is None
            ), "Projection 'out' is decalred, but not passed."
        else:
            assert (
                len(proj_out_topk_indices) == self.result_out_features
            ), "projection 'out' dimension mismatch."

        if self.result_in_features is not None:
            weight = torch.zeros(
                self.base_in_features, self.result_in_features, **factory_kwargs
            )
            weight[proj_in_topk_indices, torch.arange(self.result_in_features)] = 1
            self.projection_in_weight = nn.Parameter(weight, requires_grad=True)

        if self.result_out_features is not None:
            weight = torch.zeros(
                self.result_out_features, self.base_out_features, **factory_kwargs
            )
            weight[torch.arange(self.result_out_features), proj_out_topk_indices] = 1
            self.projection_out_weight = nn.Parameter(weight, requires_grad=True)

        if self.result_in_features is not None or self.result_out_features is not None:
            final_in_features = (
                self.result_in_features
                if self.result_in_features is not None
                else self.base_in_features
            )
            final_out_features = (
                self.result_out_features
                if self.result_out_features is not None
                else self.base_out_features
            )
            weight = torch.zeros(
                final_out_features, final_in_features, **factory_kwargs
            )
            self.auxiliary_weight = nn.Parameter(weight, requires_grad=True)

        self.initialized_compression = True

    def finalize(self):
        device = "cpu"
        weight = self.weight.full_tensor().float().to(device)

        if self.result_in_features is not None:
            weight = weight @ self.projection_in_weight.full_tensor().float().to(device)

        if self.result_out_features is not None:
            weight = (
                self.projection_out_weight.full_tensor().float().to(device) @ weight
            )

        if self.result_in_features is not None or self.result_out_features is not None:
            weight += self.auxiliary_weight.full_tensor().float().to(device)

        self.projection_in_weight = None
        self.projection_out_weight = None
        self.auxiliary_weight = None
        if os.environ["RANK"] == "0":
            self.weight = nn.Parameter(weight, requires_grad=False)
        else:
            self.weight = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # gradient magic happens here - PC optimization
        # TODO maybe order of projections matter for speed
        weight = self.weight

        if self.result_in_features is not None:
            weight = weight @ self.projection_in_weight

        if self.result_out_features is not None:
            weight = self.projection_out_weight @ weight

        if self.result_in_features is not None or self.result_out_features is not None:
            weight += self.auxiliary_weight

        return F.linear(input, weight, bias=None)

    def extra_repr(self) -> str:
        if self.result_in_features is not None:
            result = f"(projection_in_weight) ({self.base_in_features}, {self.result_in_features})\n"
        else:
            result = ""
        result += f"(weight) ({self.base_out_features}, {self.base_in_features})"
        if self.result_out_features is not None:
            result += f"\n(projection_out_weight) ({self.result_out_features}, {self.base_out_features})"
        return result


class ProjectedEmbedding(Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        result_out_features: int,
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.projection = None
        self.result_out_features = result_out_features
        self.initialized_compression = False

    def finalize(self):
        device = "cpu"
        pweight = (
            self.projection.full_tensor().float().to(device)
            @ self.weight.full_tensor().float().to(device).T
            + self.auxiliary_weight.weight.full_tensor().float().to(device).T
        ).T

        self.projection = None
        self.auxiliary_weight.weight = None

        if os.environ["RANK"] == "0":
            self.weight = nn.Parameter(pweight, requires_grad=False)
        else:
            self.weight = None

    def forward(self, x):
        result = super().forward(x)
        if self.initialized_compression:
            result = F.linear(result, self.projection, bias=None)
            result += self.auxiliary_weight(x)
        return result

    def init_projection(self, topk_dmodel_indices, **factory_kwargs):
        assert len(topk_dmodel_indices) == self.result_out_features
        vocab_size, dmodel = self.weight.shape
        weight = torch.zeros(self.result_out_features, dmodel, **factory_kwargs)
        weight[torch.arange(self.result_out_features), topk_dmodel_indices] = 1
        self.projection = nn.Parameter(weight, requires_grad=True)
        self.initialized_compression = True

        zeros = torch.zeros(vocab_size, self.result_out_features, **factory_kwargs)
        self.auxiliary_weight = nn.Embedding(
            vocab_size, self.result_out_features, _weight=zeros
        )
