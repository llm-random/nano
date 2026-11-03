"""Model FLOPs Utilization: how much of the GPU's peak the training loop uses.

Conventions (fixed in documents/notes/training_efficiency.md so every number
means the same thing):

- FLOPs/token (training) = 6 * P + score_flops, the PaLM convention: a matmul of P
  parameters costs 2P per token forward and 4P backward; attention scores add
  12 * s * datt per layer (QK^T and AV each cost 2 * s * datt per token forward),
  with no discount for causal masking.
- P is *measured*, not assumed: every 2-D weight in the live model except the
  lookups (norms are 1-D and drop out naturally; the untied head counts). A
  lookup is identified by its module being an nn.Embedding, which is what the
  rule has always meant: a gather costs no matmul FLOPs. That covers the token
  embedding and, since AME's Q/K banks are nn.Embeddings too, the banks -- which
  are ~4x the backbone in parameters and would otherwise be charged 6*P they
  never spend.
- Attention score FLOPs come from the attention modules themselves, through an
  `attention_score_flops_per_token` method, so a new type states its own cost
  instead of being invisible (0) or misread as something it is not. A module may
  likewise name 2-D parameters of its own that are not matmul weights, through
  `non_matmul_parameters`.
- Peaks are dense BF16 tensor-core numbers, because the production single-GPU path
  computes in bf16 (FSDP2's MixedPrecisionPolicy). An unknown GPU gives None: the
  caller logs tokens/s and no MFU rather than a wrong one.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Dense BF16 peak FLOPs/s. Ordered: first substring match wins.
_PEAK_BF16 = (
    ("H100 80GB HBM3", 989e12),  # SXM
    ("H100 PCIe", 756e12),
    ("H100 NVL", 835e12),
    ("GH200", 989e12),
    ("A100", 312e12),
)


def peak_flops_bf16(device_name):
    for needle, peak in _PEAK_BF16:
        if needle in device_name:
            return peak
    logger.warning(
        f"No BF16 peak known for {device_name!r}; tokens/s will be logged, MFU not."
    )
    return None


def model_flops_per_token(model, seq_len):
    """Training FLOPs per token for the live (possibly wrapped) model.

    Works through FSDP wrappers because module classes and global DTensor shapes
    survive fully_shard. Attention-score FLOPs are collected per block from the
    modules that actually do attention, so a mixed stack reports its true mix;
    an attention type that is not score-based simply declines to define
    `attention_score_flops_per_token` and contributes only its parameter matmuls.
    """
    lookups = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
        for parameter in module.parameters(recurse=False)
    }
    # "2-D means matmul weight" is a good heuristic and not a law: AME's per-head
    # shift coefficients are [H, 2] and its sink key is [H, d_head], neither of
    # which is a matmul. A module names such parameters and they drop out.
    for module in model.modules():
        for name in getattr(module, "non_matmul_parameters", ()):
            parameter = getattr(module, name, None)
            if isinstance(parameter, torch.Tensor):
                lookups.add(id(parameter))
    matmul_params = 0
    for _, parameter in model.named_parameters():
        if parameter.dim() == 2 and id(parameter) not in lookups:
            matmul_params += parameter.numel()

    score_flops = 0
    for module in model.modules():
        score_fn = getattr(module, "attention_score_flops_per_token", None)
        if score_fn is not None:
            score_flops += score_fn(seq_len)

    return 6 * matmul_params + score_flops
