# Glossary — PC project vocabulary

**Status:** established. Written 2026-10-20. Each entry says where it is defined in the code or the
configs; check there if an entry looks stale.

## Methods

| Term | Meaning |
|---|---|
| HP | Hard pruning. Structured channel pruning of `dmodel` and `dff` by importance score, then recovery training. `src.projected_compression.pruning.prune`, configs `llama_{1B,8B}_prune.yaml`. |
| HPD | HP + distillation from the uncompressed teacher. Configs `llama_{1B,8B}_prune_distillation.yaml`, `src.projected_compression.trainer_distillation`. |
| PC | Projected Compression — the method of the paper. Keeps the pretrained weights frozen and trains projection matrices around them; at finalization the projections are merged into the weights, giving a smaller *standard* transformer: `W_c = P₁ᵀ W P₂ᵀ`. `src/projected_compression/`. |
| PCD | PC + distillation. |
| PC_memeff | The memory-efficient PC implementation (`mem_eff.py`), the default for the paper's runs. |
| SVD | Baseline: SVD-LLM compression followed by full-parameter recovery training (not LoRA). |
| SliceGPT / SG-H | Baseline: rotation (PCA) + slicing of the residual stream. `SG-H` is our hybrid — SliceGPT on `dmodel`, HP-style channel pruning on `dff`. See [notes/slicegpt.md](../notes/slicegpt.md). |

## Compression ratio

**Trap — two conventions exist.** In the current configs, the paper tables and all 2026 notes,
**"50 / 70 / 90 %" is the fraction of parameters removed**: the 90 % model keeps ~12.6 % of the
parameters. Older material (and some wandb run names) uses the opposite reading, "30 % compression"
meaning the model *keeps* 30 %. When a number is ambiguous, resolve it with the dimensions:

| | 1B (`llama_3_1`, base 1498.5M) | 8B (`llama_3_8`, base 8030.3M) |
|---|---|---|
| base | dmodel 2048 / dff 8192 / datt 2048 | dmodel 4096 / dff 14336 / datt 4096 |
| 50 % | 1344 / 5376 → 801.7M (kept 0.535) | 3072 / 9216 → 4.51B (kept 0.562) |
| 70 % | 960 / 3840 → 501.9M (kept 0.335) | 2160 / 6480 → 2.61B (kept 0.324) |
| 90 % | 448 / 1728 → 188.8M (kept 0.126) | 1056 / 3168 → 0.94B (kept 0.117) |

Counted with untied embedding + head, as nano trains them. `datt`, `q_heads` and `kv_heads` are
**not** reduced by any of our methods — see [h-balanced-model.md](h-balanced-model.md).

## Models

- **llama1B** = Llama-3.2-1B, nano config [configs/_model/llama_3_1.yaml](../../configs/_model/llama_3_1.yaml)
  (16 blocks, dmodel 2048, dff 8192, 32 q heads / 8 kv heads, vocab 128256, RMSNorm, SwiGLU FF).
- **llama8B** = Llama-3.1-8B, [configs/_model/llama_3_8.yaml](../../configs/_model/llama_3_8.yaml)
  (32 blocks, dmodel 4096, dff 14336, head dim 128).
- **def_300 / def_800** = the llm-random family, [H-balanced](h-balanced-model.md) at H = 16 and
  H = 24, LayerNorm, vocab 50257. Their configs still use the ungated two-matrix FF; new models
  use a **gated FF** (`gate` + `up` + `down`), which is the assumption behind every parameter and
  FLOP count in `documents/`.
- Embeddings are trained **untied** in nano, including for models whose HF original ties them
  (Llama-3.2-1B). This matters for every parameter count.

## Training conventions

- **Token budgets**: 1BT / 2BT / 4BT / 6BT. 1B model: `n_steps` 2048 / 4096 / 8192 / 12288 at
  seq 1024 × batch 512. 8B model: 1024 / 2048 / 4096 / 6144 at seq 2048 × batch 512. Both ≈ 1.07 /
  2.1 / 4.3 / 6.4 B tokens; the paper rounds to 1/2/4/6 B.
- **Learning rate**: `trainer.learning_rate: n` with `n ≥ 1` means **lr = 2⁻ⁿ**; a value below 1 is
  used literally ([src/core/utils.py:45](../../src/core/utils.py#L45)). Tune only n ± 1. Known-good
  at 1B: HP 12/12/11, HPD 11/11/12 (50/70/90 %), PC 13.
- **Data**: FineWeb-Edu, deterministic dataloader keyed by seed — runs with the same seed, context
  length and batch size see the same samples in the same order, which is what makes PC and HP runs
  directly comparable.
- **Paper metric**: the final value of `steps/100/train/loss` in wandb.
- **Importance scores**: one *global* vector for `dmodel` (the residual stream is shared across
  layers) and a *per-layer* vector for `dff` (the FF layers are independent).
  `src/projected_compression/weight_importances.py`, precomputed tensors under
  `/storage_nvme_4/nano/j321m/pc/importances/`.

## Infrastructure

- **Pixi** is the environment manager (one env per cluster); `update_pixi.py` refreshes it remotely.
- **Running**: `python main.py --config-name <cfg>` locally; `python run_exp.py --config-name <cfg>`
  submits to a cluster (`infrastructure.server`, SLURM block in the config).
- **FSDP2** is the default sharding for PC — it behaves better with frozen weights.
- **wandb**: `ideas_cv/llm-random-test` holds the paper's Table 1 runs; recent work logs to
  `pc_v2`; the timing sweeps to `pc_time`.
