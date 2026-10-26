# SliceGPT (Ashkboos et al., ICLR 2024) ported to nano, used as a compression baseline.
# Algorithm code is adapted from https://github.com/microsoft/TransformerCompression
# (Copyright (c) Microsoft Corporation, MIT license).
#
# Pipeline (see slicegpt.md in the repo root):
#   1. optionally prune dff exactly like HP (`prune_dff`)        -> hybrid "SliceGPT + FFN pruning"
#   2. fold RMSNorm weights into the following linears (`fuse_norms`)
#   3. per-layer PCA rotation + slicing of the residual width (`rotate_and_slice`)
#   4. turn the weightless norms back into standard RMSNorm(d) (`finalize_norms`)
# The result is a standard nano Llama with dmodel=d plus a `shortcut_Q` (d x d) in every residual.

from src.projected_compression.slicegpt.fusion import RMSN, finalize_norms, fuse_norms
from src.projected_compression.slicegpt.slice import prune_dff, rotate_and_slice

__all__ = ["RMSN", "finalize_norms", "fuse_norms", "prune_dff", "rotate_and_slice"]
