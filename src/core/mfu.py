"""Model FLOPs Utilization: how much of the GPU's peak the training loop uses.

Conventions (fixed in documents/notes/pretraining_stage3_mfu_plan.md so every number
means the same thing):

- FLOPs/token (training) = 6 * P + score_flops, the PaLM convention: a matmul of P
  parameters costs 2P per token forward and 4P backward; attention scores add
  12 * s * datt per layer (QK^T and AV each cost 2 * s * datt per token forward),
  with no discount for causal masking.
- P is *measured*, not assumed: every 2-D weight in the live model except the
  embedding lookup (norms are 1-D and drop out naturally; the untied head counts).
  This way mha/mqa/gqa and future attention types report their true cost.
- Peaks are dense BF16 tensor-core numbers, because the production single-GPU path
  computes in bf16 (FSDP2's MixedPrecisionPolicy). An unknown GPU gives None: the
  caller logs tokens/s and no MFU rather than a wrong one.
"""

import logging

from src.core.model import RoPEAttention

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
    a future attention type that is not score-based simply contributes its
    parameter matmuls until it registers its own term here.
    """
    matmul_params = 0
    for name, parameter in model.named_parameters():
        if parameter.dim() == 2 and "embedding" not in name:
            matmul_params += parameter.numel()

    score_flops = 0
    for module in model.modules():
        if isinstance(module, RoPEAttention):
            score_flops += 12 * seq_len * module.q_heads * module.dhead

    return 6 * matmul_params + score_flops
