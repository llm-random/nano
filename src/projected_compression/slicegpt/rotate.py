# Adapted from TransformerCompression/src/slicegpt/rotate.py
# Copyright (c) Microsoft Corporation. Licensed under the MIT license.
#
# Per-weight rotate / slice operations and the PCA. nano Linear/Embedding weights use the torch layout
# (out_features, in_features) / (vocab, dmodel), same as Hugging Face, so the math is unchanged.
from typing import Optional

import numpy as np
import torch
from torch import nn


@torch.no_grad()
def rotate_inputs(linear: nn.Linear, Q: torch.Tensor) -> None:
    """W <- W Q  (rotate the input / residual-side dimension)."""
    dtype, device = linear.weight.dtype, linear.weight.device
    W_ = linear.weight.data.to(device=Q.device, dtype=torch.float64)
    linear.weight.data = torch.matmul(W_, Q).to(device=device, dtype=dtype)


@torch.no_grad()
def slice_inputs(linear: nn.Linear, d: int) -> None:
    linear.weight.data = linear.weight.data[:, :d].contiguous()
    linear.in_features = d


@torch.no_grad()
def rotate_outputs(linear: nn.Linear, Q: torch.Tensor) -> None:
    """W <- Q^T W  (rotate the output / residual-side dimension)."""
    dtype, device = linear.weight.dtype, linear.weight.device
    W_ = linear.weight.data.to(device=Q.device, dtype=torch.float64)
    linear.weight.data = torch.matmul(Q.T, W_).to(device=device, dtype=dtype)
    assert linear.bias is None, "nano Llama linears have no bias"


@torch.no_grad()
def slice_outputs(linear: nn.Linear, d: int) -> None:
    linear.weight.data = linear.weight.data[:d, :].contiguous()
    linear.out_features = d


@torch.no_grad()
def rotate_embedding(embedding: nn.Embedding, Q: torch.Tensor) -> None:
    dtype, device = embedding.weight.dtype, embedding.weight.device
    W_ = embedding.weight.data.to(device=Q.device, dtype=torch.float64)
    embedding.weight.data = torch.matmul(W_, Q).to(device=device, dtype=dtype)


@torch.no_grad()
def slice_embedding(embedding: nn.Embedding, d: int) -> None:
    embedding.weight.data = embedding.weight.data[:, :d].contiguous()
    embedding.embedding_dim = d


def random_orthogonal_upper_left(
    total_dim: int, upper_block_dim: int, rng: np.random.Generator
) -> torch.Tensor:
    """Square matrix whose upper-left block is a random orthogonal matrix and the rest is the identity.

    Same as SliceGPT's helper, but with an explicit (seeded) generator for reproducibility.
    """
    A = rng.random((upper_block_dim, upper_block_dim))
    Q, _ = np.linalg.qr(A)
    R = np.eye(total_dim)
    R[:upper_block_dim, :upper_block_dim] = Q
    return torch.from_numpy(R)


@torch.no_grad()
def pca_calc(
    X: list[torch.Tensor],
    device: torch.device,
    ignore_masks: Optional[list[torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PCA over a list of [batch, seq, D] signals. Returns (eigenvalues, eigenvectors), descending, fp64.

    Identical to SliceGPT's `pca_calc` (fp64 covariance, 1% mean-diagonal damping, eigh). `ignore_masks`
    ([batch, seq], 0 = padding) zeroes padded positions like theirs; our FineWeb-Edu calibration is packed
    to full length (no padding), masks are only needed for the reference cross-check (their WikiText loader).
    """
    H = None
    for idx, X_batch in enumerate(X):
        X_batch = X_batch.to(device=device, dtype=torch.float64)
        if ignore_masks is not None:
            X_batch = X_batch * ignore_masks[idx].to(
                device=device, dtype=torch.float64
            ).unsqueeze(-1)
        H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)  # sum over the batch dimension
        H = H_batch if H is None else H + H_batch

    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1], device=device)
    H[diag, diag] = H[diag, diag] + damp
    X_eig = torch.linalg.eigh(H)
    del H
    index = torch.argsort(X_eig[0], descending=True)
    eig_val = X_eig[0][index]
    eigen_vec = X_eig[1][:, index]
    return eig_val, eigen_vec
