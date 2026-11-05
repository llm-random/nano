# AME configs

**Moved 2026-11-05:** every AME config that used to live here — benchmarks, A/B
twins, ablations, the h20 smoke run — is now in `configs/ame_tests/` under the same
name. Run one with `--config-path configs/ame_tests --config-name <name>`.

This directory holds six configs, one per (model size, GPU count): the best MFU
measured for that cell, copied from the run that measured it. Each file's header
names its source config and job; the table is
`documents/notes/ame_optimization_report.md` §6.1.

| config | GPUs | micro-batch × grad-accum | MFU/GPU | source, in `configs/ame_tests/` |
|---|---:|---:|---:|---|
| `h12_1gpu.yaml` | 1 | 32 × 16 | 34.46% | `h12_1gpu_mb32.yaml` |
| `h12_4gpu.yaml` | 4 | 32 × 4 | 30.00% | `h12_4gpu_mb16_o29.yaml` + `trainer.gradient_accumulation_steps=4` |
| `h16_1gpu.yaml` | 1 | 16 × 32 | 37.96% | `h16_1gpu_mb16.yaml` |
| `h16_4gpu.yaml` | 4 | 16 × 8 | 33.30% | `h16_4gpu_mb16_o29.yaml` |
| `h20_1gpu.yaml` | 1 | 4 × 128 | 35.59% | `h20_1gpu_mb4_nofsdp.yaml` |
| `h20_4gpu.yaml` | 4 | 16 × 8 | 35.02% | `h20_4gpu_mb16_mp4_o29.yaml` |

The training settings in these files (30 steps, no eval, no checkpoints, entropy,
1 h) are the measurement's, not a training recipe.

Only `h20_1gpu.yaml` runs without FSDP2, so only it logs AME's alpha/beta/tau/entropy
readouts. Under FSDP2, at any GPU count, `instrumentation()` hits sharded parameters
and the trainer drops it.
