# Adapted from TransformerCompression/src/slicegpt/{layernorm_fusion.py,modules.py}
# Copyright (c) Microsoft Corporation. Licensed under the MIT license.
import logging
import math
from typing import Iterable

import torch
from torch import nn

from src.core.model import RMSNorm
from src.projected_compression.slicegpt import nano_adapter as na

logger = logging.getLogger(__name__)


class RMSN(nn.Module):
    """Weightless RMSNorm with a FIXED divisor `mean_dim` (SliceGPT's `RMSN`).

    After slicing, the input has d < mean_dim features, but the sum of squares is still divided by the
    ORIGINAL width, so normalisation stays consistent with the unsliced model.
    """

    def __init__(self, mean_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)


@torch.no_grad()
def fuse_ln_linear(norm: nn.Module, linear_layers: Iterable[nn.Linear]) -> None:
    """Fold the norm's scale into the input dimension of the following linear layers."""
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * norm.weight.double()).to(linear_dtype)
        # nano Llama norms (torch RMSNorm) have no bias; LayerNorm models are not supported here.
        assert (
            getattr(norm, "bias", None) is None
        ), "only RMSNorm (no bias) is supported"


def _norm_eps(norm: nn.Module) -> float:
    eps = getattr(norm, "eps", None)
    return 1e-5 if eps is None else float(eps)


@torch.no_grad()
def fuse_norms(model: nn.Module, center_embeddings: bool = False) -> None:
    """Fold every RMSNorm weight into the next linear(s) and replace the norms with weightless RMSN(D).

    The model output is unchanged. `center_embeddings` reproduces SliceGPT's unconditional embedding
    mean-subtraction (only meant for LayerNorm models; it changes an RMSNorm model's function), kept
    off by default and exposed only to replicate their pipeline exactly in the reference cross-check.
    """
    D = na.get_embedding(model).weight.shape[1]

    head = na.get_lm_head(model)
    if head.weight is na.get_embedding(model).weight:  # untie if tied
        head.weight = nn.Parameter(head.weight.clone())

    if center_embeddings:
        W = na.get_embedding(model).weight
        W_ = W.data.double()
        W.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.dtype)

    for block in na.get_blocks(model):
        fuse_ln_linear(na.get_first_norm(block), na.get_attention_inputs(block))
        fuse_ln_linear(na.get_second_norm(block), na.get_mlp_inputs(block))
        block.attention_layer.norm = RMSN(D, _norm_eps(block.attention_layer.norm))
        block.ff_layer.norm = RMSN(D, _norm_eps(block.ff_layer.norm))

    fuse_ln_linear(na.get_pre_head_norm(model), [head])
    model.head.norm = RMSN(D, _norm_eps(model.head.norm))
    logger.info("Fused RMSNorm weights into adjacent linears.")


@torch.no_grad()
def finalize_norms(model: nn.Module) -> None:
    """Replace each weightless RMSN(D) by a standard RMSNorm(d) with weight sqrt(D/d).

    nano's RMSNorm divides by the (sliced) width d, SliceGPT's RMSN by the original D:
        sqrt(D/d) * x / sqrt(sum(x^2)/d + eps) = x / sqrt(sum(x^2)/D + eps*d/D)
    i.e. identical up to the (negligible) eps term. The weight stays trainable for recovery training,
    like the norm weights of HP / PC models.
    """

    def _convert(norm: RMSN, d: int, device, dtype) -> RMSNorm:
        new = RMSNorm(
            d, eps=norm.eps, elementwise_affine=True, device=device, dtype=dtype
        )
        new.weight.data.fill_(math.sqrt(norm.mean_dim / d))
        return new

    # each norm's width = input width of the linear(s) it feeds (d everywhere, except the head norm when
    # the head is left unsliced, cf. rotate_and_slice(slice_head=False))
    ref = na.get_embedding(model).weight
    for block in na.get_blocks(model):
        d_attn = na.get_attention_inputs(block)[0].in_features
        d_mlp = na.get_mlp_inputs(block)[0].in_features
        block.attention_layer.norm = _convert(
            block.attention_layer.norm, d_attn, ref.device, ref.dtype
        )
        block.ff_layer.norm = _convert(
            block.ff_layer.norm, d_mlp, ref.device, ref.dtype
        )
    d_head = na.get_lm_head(model).in_features
    model.head.norm = _convert(model.head.norm, d_head, ref.device, ref.dtype)
