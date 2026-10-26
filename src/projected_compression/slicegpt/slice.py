# Adapted from TransformerCompression/src/slicegpt/{rotate.py,model_utils.py}
# Copyright (c) Microsoft Corporation. Licensed under the MIT license.
import logging
from typing import Optional

import numpy as np
import torch
from torch import nn

from src.projected_compression.slicegpt import nano_adapter as na
from src.projected_compression.slicegpt.fusion import RMSN
from src.projected_compression.slicegpt.rotate import (
    pca_calc,
    random_orthogonal_upper_left,
    rotate_embedding,
    rotate_inputs,
    rotate_outputs,
    slice_embedding,
    slice_inputs,
    slice_outputs,
)

logger = logging.getLogger(__name__)


@torch.no_grad()
def prune_dff(
    model: nn.Module, dimensions_importances_path: str, target_dff: int
) -> None:
    """Prune the FFN hidden size exactly like HP (`src.projected_compression.pruning.prune`, dff branch):
    per layer keep the top-`target_dff` channels by activation importance. Residual width is untouched.
    """
    dff_importances = torch.load(dimensions_importances_path)["dff_importances"]
    blocks = na.get_blocks(model)
    assert len(dff_importances) == len(blocks)
    for block, importances in zip(blocks, dff_importances):
        ff = block.ff_layer.layer
        idx = torch.topk(importances, dim=0, largest=True, k=target_dff).indices.to(
            ff.ff_pre_act.weight.device
        )
        for lin in (ff.ff_pre_act, ff.gate):
            lin.weight.data = lin.weight.data[idx, :].contiguous()
            lin.out_features = target_dff
        ff.ff_post_act.weight.data = ff.ff_post_act.weight.data[:, idx].contiguous()
        ff.ff_post_act.in_features = target_dff
    logger.info(f"Pruned dff to {target_dff} (HP importances).")


@torch.no_grad()
def get_signals(
    block: nn.Module, inps: list[torch.Tensor], device: torch.device
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Run `block` on each input batch. Return (pre-MLP-norm signal, block output), both on CPU.

    nano blocks take only the hidden states (RoPE + causal mask are internal), so no args/kwargs
    plumbing is needed (cf. SliceGPT `get_signals`).
    """
    mlp_ln_inputs, outputs = [], []

    def hook_fn(_module, args, _output):
        mlp_ln_inputs.append(args[0].cpu())

    second_norm = na.get_second_norm(block)
    assert isinstance(
        second_norm, RMSN
    ), "norms must be fused (fuse_norms) before slicing"
    hook = second_norm.register_forward_hook(hook_fn)
    try:
        for inp in inps:
            outputs.append(block(inp.to(device)).cpu())
    finally:
        hook.remove()
    return mlp_ln_inputs, outputs


@torch.no_grad()
def rotate_and_slice(
    model: nn.Module,
    calibration_batches: list[torch.Tensor],
    d: int,
    orientation: str = "random",
    seed: int = 0,
    device: Optional[torch.device] = None,
    slice_head: bool = True,
    ignore_masks: Optional[list[torch.Tensor]] = None,
) -> dict:
    """SliceGPT `rotate_and_slice_sequential` for nano Llama (const slicing).

    Norms must already be fused (`fuse_norms`). Residual connections get `shortcut_Q` parameters
    (d x d after slicing). `orientation`: "pca" keeps the PCA basis, "random" (SliceGPT default)
    applies a random orthogonal rotation inside the kept subspace (function-preserving).
    `slice_head`: our baseline slices the LM head (True); SliceGPT's CLI default (`do_slice_head=False`)
    keeps the last block's output and the head at full width D -- used only for the reference cross-check.
    `ignore_masks`: optional per-batch [batch, seq] masks (0 = padding) excluded from the PCA (right padding only:
    with causal attention, padded positions cannot affect real ones). Not needed for packed FineWeb-Edu batches.
    Returns per-layer diagnostics (kept fraction of the signal variance).
    """
    assert orientation in ("random", "pca")
    device = device or next(model.parameters()).device
    model.eval()
    dtype = na.get_embedding(model).weight.dtype
    D = na.get_embedding(model).weight.shape[1]
    assert d <= D
    rng = np.random.default_rng(seed)
    diagnostics = {"kept_variance": []}

    def orient(Q: torch.Tensor, k: int) -> torch.Tensor:
        if orientation == "random":
            Q = Q @ random_orthogonal_upper_left(Q.shape[0], k, rng).to(Q)
        return Q

    def record(name: str, eig_val: torch.Tensor) -> None:
        diagnostics["kept_variance"].append(
            (name, (eig_val[:d].sum() / eig_val.sum()).item())
        )

    # inputs of the first block = embedding outputs (pre-norm residual stream)
    inps = [
        na.get_embedding(model)(ids.to(device)).cpu() for ids in calibration_batches
    ]

    # rotate and slice embeddings
    eig_val, Q = pca_calc(inps, device, ignore_masks)
    record("embedding", eig_val)
    Q = orient(Q, d)
    rotate_embedding(na.get_embedding(model), Q)
    slice_embedding(na.get_embedding(model), d)

    blocks = na.get_blocks(model)
    for idx, block in enumerate(blocks):
        attn_res, ff_res = block.attention_layer, block.ff_layer
        # output width of this block: d, except the last block when the head is not sliced
        d_out = D if (idx == len(blocks) - 1 and not slice_head) else d

        # rotate and slice the attention inputs to match the previous layer
        attn_res.shortcut_Q = nn.Parameter(Q.T.clone().to(dtype=dtype))
        for W in na.get_attention_inputs(block):
            rotate_inputs(W, Q)
            slice_inputs(W, d)
        attn_res.shortcut_Q = nn.Parameter(attn_res.shortcut_Q[:d, :])
        inps = [
            torch.matmul(inp.to(device), Q.to(dtype=dtype))[:, :, :d].cpu()
            for inp in inps
        ]

        # signal between attention and MLP -> PCA -> rotate/slice attention output and MLP inputs
        mlp_ln_inputs, _ = get_signals(block, inps, device)
        eig_val, Q = pca_calc(mlp_ln_inputs, device, ignore_masks)
        record(f"block{idx}.mid", eig_val)
        Q = orient(Q, d)
        attn_res.shortcut_Q = nn.Parameter(
            torch.matmul(attn_res.shortcut_Q, Q.to(dtype=dtype)[:, :d])
        )
        rotate_outputs(na.get_attention_output(block), Q)
        slice_outputs(na.get_attention_output(block), d)

        ff_res.shortcut_Q = nn.Parameter(Q.T.clone().to(dtype=dtype)[:d, :])
        for W in na.get_mlp_inputs(block):
            rotate_inputs(W, Q)
            slice_inputs(W, d)
        del mlp_ln_inputs

        # block outputs = inputs of the next block -> PCA -> rotate/slice MLP output
        _, inps = get_signals(block, inps, device)
        eig_val, Q = pca_calc(inps, device, ignore_masks)
        record(f"block{idx}.out", eig_val)
        Q = orient(Q, d_out)
        ff_res.shortcut_Q = nn.Parameter(
            torch.matmul(ff_res.shortcut_Q, Q.to(dtype=dtype))
        )
        rotate_outputs(na.get_mlp_output(block), Q)
        slice_outputs(na.get_mlp_output(block), d_out)
        ff_res.shortcut_Q = nn.Parameter(ff_res.shortcut_Q[:, :d_out])

        block.attention_layer.layer.dmodel = d
        logger.info(f"Rotated and sliced block {idx} to d={d}")

    # rotate (and, for our baseline, slice) the head: an unsliced V x D head would exceed our param budgets
    rotate_inputs(na.get_lm_head(model), Q)
    if slice_head:
        slice_inputs(na.get_lm_head(model), d)
    return diagnostics
