"""Model FLOPs Utilization: how much of the GPU's peak the training loop uses.

Conventions (fixed in documents/notes/training_efficiency.md so every number
means the same thing):

- FLOPs/token (training) = 6 * P + score_flops, the PaLM convention: a matmul of P
  parameters costs 2P per token forward and 4P backward; attention scores add
  12 * s * datt per layer (QK^T and AV each cost 2 * s * datt per token forward),
  with no discount for causal masking.
- P is *measured*, not assumed: every 2-D weight in the live model except the
  embedding lookup (norms are 1-D and drop out naturally; the untied head counts).
  This way mha/mqa/gqa and future attention types report their true cost.
  An MoE layer counts its router and the topk / num_experts share of its experts.
- A looped encoder's core runs n_loops times per forward, so its weights and its
  attention scores count n_loops times: FLOPs follow the effective depth.
- Peaks are dense BF16 tensor-core numbers, because the production single-GPU path
  computes in bf16 (FSDP2's MixedPrecisionPolicy). An unknown GPU gives None: the
  caller logs tokens/s and no MFU rather than a wrong one.
"""

import logging

from src.core.model import RoPEAttention, TransformerEncoder
from src.core.moe import MoE

logger = logging.getLogger(__name__)

# Dense BF16 peak FLOPs/s. Ordered: first substring match wins.
_PEAK_BF16 = (
    ("H100 80GB HBM3", 989e12),  # SXM
    ("H100 PCIe", 756e12),
    ("H100 NVL", 835e12),
    ("GH200", 989e12),
    ("A100", 312e12),
    ("NVIDIA H100", 989e12),  # SXM5 94GB (lem) reports the bare name
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
    runs = _runs_per_forward(model)
    matmul_params = 0
    for name, parameter in model.named_parameters():
        if parameter.dim() == 2 and "embedding" not in name:
            matmul_params += runs.get(id(parameter), 1) * parameter.numel()

    score_flops = 0
    for module in model.modules():
        n_runs = runs.get(id(module), 1)
        if isinstance(module, RoPEAttention):
            score_flops += n_runs * 12 * seq_len * module.q_heads * module.dhead
        if isinstance(module, MoE):
            # 3D expert weights are skipped above; a token runs topk of the experts
            experts = sum(p.numel() for p in module.expert_weights())
            matmul_params += n_runs * (experts // module.num_experts * module.topk)

    return 6 * matmul_params + score_flops


def _runs_per_forward(model):
    """id -> passes per forward, for every module and parameter a looped encoder reuses."""
    runs = {}
    for encoder in model.modules():
        if isinstance(encoder, TransformerEncoder):
            for looped in encoder.looped_modules():
                for item in (*looped.modules(), *looped.parameters()):
                    runs[id(item)] = encoder.n_loops
    return runs
