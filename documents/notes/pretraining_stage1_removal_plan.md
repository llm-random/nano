# Stage 1: content removal plan (`pretraining` branch)

Goal: strip compression, pruning, distillation and every other past-project feature
from the `pretraining` branch, leaving a minimal, runnable, test-green pretraining
framework. Nothing is lost — `main` keeps all of it.

**Status: PLAN — not executed. Awaiting approval.**

## Ground rules

- Work in `/home/maciej/code/nano-pretraining`, commits directly on `pretraining`.
- One phase = one or more commits; **pytest must be green after every commit**
  (`pytest -m "not gpu and not slow and not network"`, same filter as CI, plus
  `black --check`).
- Before anything: record the baseline (test count, and the loss of a seeded
  `tiny_local` run) to verify behavior is unchanged where it must be.
- `git rm` only; no history rewriting. Deleted things are one
  `git checkout main -- <path>` away.

## Phase A — relocate the live model stack (the enabler, do first)

**Problem:** the generic transformer (`LLM`, `TransformerEncoder`, `TransformerBlock`,
`RoPEAttention`, `RoPE`, `TransformerEmbedding`, `TransformerHead`, `Linear`,
`Residual`, `FeedForward`, `LlamaFeedForward`, `llm_random_weight_init`) lives in
`src/projected_compression/model.py`, and every base llama config targets it.
`src/core/model.py` holds a *dead* parallel stack (own `LLM`, `TransformerTower`,
`RoPEAttention`, `MLP`, `SwiGLU`, `PositionalEmbedding`, `Aggregate`…) that **no
config and no test instantiates**. Deleting the PC package first would break every
runnable config.

**Steps (single commit, mechanical):**

1. In `src/core/model.py`: delete the dead stack, **keeping** what the live code
   actually uses from core: `RMSNorm`, `AttentionMechanism`, `Residual` (check
   `main.py`'s use for activation-checkpoint wrapping — there are currently *two*
   `Residual` classes, core's and PC's; after the move exactly one remains),
   plus the init helpers (`get_init_weight` family) if anything kept references them.
2. Move the generic classes from `src/projected_compression/model.py` into
   `src/core/model.py` verbatim (no refactors — refactoring is stage 2).
   `Projected*` classes stay behind in the PC package and import the generics from
   core (they die in Phase B anyway; this keeps the repo green between phases).
3. Re-point Hydra targets `src.projected_compression.model.X` →
   `src.core.model.X` in the **kept** configs only:
   `configs/_model/llama/base_model.yaml`, `configs/_model/tiny.yaml`,
   `configs/ff_layer/dense.yaml` (see Phase C.2 — dense gets inlined, group deleted),
   `configs/tiny_local.yaml`, `configs/tiny_remote.yaml`,
   `configs/tiny_remote_wandb.yaml`, `configs/smoketest_fineweb_final_eval_local.yaml`.
   Doomed configs (pc_project etc.) are not touched — they are deleted in Phase B.
4. **Verification:** pytest green; seeded `python main.py --config-name tiny_local`
   (and the fineweb smoketest config) produce **bit-identical loss** to the Phase-0
   baseline.

## Phase B — compression / pruning / distillation / SliceGPT

| What | Paths |
|---|---|
| PC package (incl. pruning, mem_eff, slicegpt, both PC trainers, initialization, conversions) | `src/projected_compression/` (all) |
| Distillation in core | `src/core/trainer_distillation.py` |
| PC conversion in core | `src/core/conversion_from_finalized_pc.py` |
| `main.py` | drop checkpoint-load branches `sliced_nano`, `finalized_pc` + their imports |
| Configs | `configs/pc_project/` (44), `configs/pc_project_mas/` (37), `configs/slicegpt/` (4), `configs/_model/llama_3_1_pc.yaml`, `llama_3_8_pc.yaml`, `llama_3_1_sliced.yaml`, `llama_3_8_sliced.yaml`, `llama_tiny_compressor.yaml`, `configs/_model/llama/base_pc_model.yaml`, `base_pc_mem_eff.yaml`, `projected_*.yaml` (4), `configs/_trainer/llama_distillation.yaml`, `llama3_1B_distillation.yaml`, `llama3_8B_distillation.yaml` |
| Tests | delete `tests/test_projected_compression.py`, `tests/test_slicegpt.py`, `tests/test_convert_memeff_to_hf.py`; **rewrite** the PC-importing parts of `tests/test_distributed.py` (FSDP cases on plain `Linear`) and `tests/test_trainer_save_types.py` (contract test against a local stub subclass instead of `PCTrainer`) |
| SliceGPT tooling | `smoke/` (whole dir), `scripts/slicegpt_crosscheck.py`, `scripts/slicegpt_dump_ref_tokens.py`, `scripts/slicegpt_perturb_diag.py` |
| Hygiene test upkeep | `test_core_does_not_depend_on_projected_compression` is deleted (its job is done — the package no longer exists); prune `KNOWN_ORPHAN_MODEL_CONFIGS` and any constants naming deleted files |

## Phase C — other past-project features

1. **Product Keys** (other team's; lives on in `main`): `src/product_keys/`,
   `configs/product_keys/`.
2. **MoE** (reimplemented later with granularity CUDA kernels): `src/core/moe.py`,
   `configs/ff_layer/moe.yaml`, `configs/_model/llama/small_moe.yaml`,
   `configs/moe_example_run.yaml`. With both `ff_layer` variants gone, **inline the
   dense (gated/SwiGLU `LlamaFeedForward`) block into `base_model.yaml`** and delete
   the `configs/ff_layer/` group. (Gated FF itself is a stage-2 requirement — it stays,
   as `LlamaFeedForward`.)
3. **SmolLM family**: `configs/_model/smollm/` (all 8).
4. **llm-random legacy**: `src/core/conversion_from_llmrandom.py`,
   `configs/_model/llmrandom/` (5), the `llm-random` checkpoint-load branch in
   `main.py`. (The *cemetery* clone flow in `run_exp.py` is job-submission infra,
   not llm-random model legacy — it stays.)
5. **HF→nano import**: in `src/core/llama.py` delete
   `remap_llamahf_state_dict_to_nano`, `copy_llama_model_weights_from_HF`,
   `save_pretrained_llama_as_nano`; delete `src/core/conversion_to_hf.py`;
   `main.py` drops the HF-weights load branch + imports;
   `configs/_checkpoints/mas/` (3 conversion configs — doubles as personal scratch).
   **Keep** `repeat_kv`, `LlamaRoPE`, `LlamaAttention`, `LLamaFeedForward` — generic
   attention code the GQA/MHA/MQA switching work builds on. Keep their tests.
6. **Model-size lineage** (replaced by H-balanced in stage 2):
   `configs/_model/llama_3_1.yaml`, `llama_3_8.yaml`, `configs/_model/llama/1B.yaml`,
   `8B.yaml`, `small.yaml`. **Keep `base_model.yaml` + `tiny.yaml` + `tiny_local` /
   `tiny_remote*` / smoketest configs as temporary scaffolding** — they are what keeps
   the repo runnable and the config-composition tests meaningful until the H-balanced
   lineage lands.
7. **PC paper artifacts**: `evals/` (111 files), `benchmarks/` (11),
   `influences_figs/` (12).
8. **Personal scratch**: `configs/_mas/` (5), `configs/mastest/` (tracked `old/`
   files). Untracked root clutter never reaches the branch — no action.
9. **Downstream eval harness** (rewritten from scratch later): `run_eval.py`,
   `evaluate_hf_models.py`, `src/core/eval.py`, `configs/_eval/`,
   `generate_eval_sbatch_script` in `grid_generator/sbatch_builder.py`; update
   `ENTRY_POINTS` in `tests/test_repo_hygiene.py`. **Not** touching `pixi.toml`
   deps (lm_eval) — lock re-solve is a separate, later step (see log).
10. **Docs**: delete PC-specific notes — `documents/notes/no_p.md`, `pc_time.md`,
    `slicegpt.md`, `table1_runs.md`, `merge_to_main.md`, `review_findings_plan.md`,
    `remaining_issues_plan.md`, `documents/scripts/no_p_90_loss_compare.ipynb`,
    `documents/scripts/pc_notebooks/`. **Keep**: `concepts/` (glossary, h-balanced —
    stage-2 spec, sam-model), `long-context-attention.md` (attention project!),
    `main_baseline_audit.md` (documents core behavior), `cluster_fixes_lem_helios.md`,
    `claude-code-machine-setup.md`, `issues.md` (prune PC-only entries),
    `documents/scripts/h_balanced_*` (stage-2 material).

## What stays after stage 1 (the skeleton)

`main.py` (fresh-init + own nano checkpoint save/resume only) · `run_exp.py` +
`grid_generator/` + `resolver` · `src/core/`: trainer, datasets, schedulers (cosine ✓),
checkpointing, distributed_training, metric_loggers, model (relocated live stack),
llama (attention/rope only), utils, definitions · configs: `_cluster/`, `_dataset/`
(fineweb ✓), `_trainer/llama.yaml`, `_checkpoints/none.yaml`, `_misc/`,
`_model/{base_model,tiny}`, `tiny_*`, smoketest, `exp.yaml`, `clusters.yaml` · tests
(hygiene, core framework, distributed, smoke, infra) · scripts: sync_cluster_env,
install_hooks, watch_jobs, expand_config, prepare_test_dataset, create_tiny_llama
(check: if only SliceGPT used it, it goes too — verify at execution) · `.githooks`,
CI workflows, pixi files untouched.

## Execution order & gates

0. Baseline: pytest count + seeded `tiny_local` loss recorded in the log.
1. Phase A (relocation) — gate: pytest green **and** bit-identical loss.
2. Phase B — gate: pytest green.
3. Phase C, one numbered item ≈ one commit — gate: pytest green each time.
4. Close: `black --check`, full pytest, re-run seeded `tiny_local` + fineweb
   smoketest, update `README.md` to describe the pretraining framework (drop
   PC/eval sections), update the log, push.
5. Optional (cheap, recommended): one tiny 1-GPU smoke run on entropy via
   `run_exp.py` to prove the submission path survived the deletions.

## Risks / open points

- **Phase A is the only risky step** — everything else is deletion. The bit-identical
  gate catches mistakes; the move is verbatim, no refactors.
- `tests/test_distributed.py` rewrite touches FSDP wrapping tests — must keep testing
  the *same* wrap/shard behavior, just on a plain module.
- `main.py` imports `Residual` from core while configs build PC's `Residual` — resolve
  to the single relocated class in Phase A and confirm activation checkpointing still
  wraps it (covered by the seeded-loss gate + smoketest).
- `README.md` likely references PC — rewrite is included in the close step.
- Approval decision: commits land directly on `pretraining` (no PR), since the branch
  is itself the new project's baseline. Say the word if you want a PR instead.
