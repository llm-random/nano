# Standard models: build, test and merge

**Status (2026-11-21 16:40):** in progress. Branch `standard_models` holds every
approved feature and the six standard configs (steps 1–6 done). Tests, smoke runs and the
h12 reproductions are next. Plan: [standard_model_improvements.md](standard_model_improvements.md#implementation-plan).

## The branch

Worktree `~/code/nano-standard`, branch `standard_models`, from `ame_main` at `7f4df31f`.

| Commit | What |
|---|---|
| `957a5349` | Merge of `ame_improve`: `62a59d2f` (`n_tokens_b`), `7762da98` (QK-norm), `e64a466c` (`bank_lr_ratio`), `7345eebc` (dense KV shift) |
| `fcd9ae21` | Cherry-pick of `83ebaa85`, the `run_exp.py` missing-key fix |
| `c491b57c` | The `bank_lr_ratio` comment in `configs/_trainer/llama.yaml`: old runs need 1 |
| `8b78b805` | The FSDP2 readout fix, with its test |
| `db1cdda2` | `configs/standard/`: the six standard configs |

- **The merge** had the one conflict the plan expected: `src/core/trainer.py`, where both
  branches add `Trainer` fields. All three kept (`count_token_updates`, `n_tokens_b`,
  `bank_lr_ratio`).
- **The cherry-pick** had a conflict the trial merge could not show, in `run_exp.py`:
  `ame_main` had added the `sparse_exclude` check where the fix replaces the missing-key
  check. Both kept: `check_missing_keys(cfg)`, then `sparse_exclude_names(...)`.
- **`bank_lr_ratio`** is 4.0 in all four places after the merge (`DEFAULT_BANK_LR_RATIO`
  and the schema in `src/definitions.py`, the `Trainer` field, `llama.yaml`).
  `bank_init_std` keeps its default of 1.0.
- **The readout fix** (`src/core/ame.py`, `AMEAttention.instrumentation`): the attention
  diagnostics run only when no parameter of the layer has `full_tensor`. The new test in
  `tests/test_distributed.py` builds an FSDP2-wrapped AME layer on a one-rank gloo mesh;
  before the fix it raised the runs' error (`aten.embedding.default: got mixed
  torch.Tensor and DTensor`), after it the readout has `alpha_2`, `beta_2`, `tau` and the
  horizons. A sharded run still logs no attention entropy or top-k mass.
- **The six configs** compose and pass `check_point`. Composed, `ame_h12` and `dense_h12`
  equal the configs of `zdmv3qci` and `chne4w91` composed on `settle_runs_improve`, apart
  from the run name, tags, job name, LEM's `mem_per_gpu` (228G against 220G),
  `sparse_exclude` and `count_token_updates: False` (off, `ame_main`'s default). Every file
  excludes r16-11; the h16 and h20 sources did that on the command line.
- **σ = 1e-4 reaches the column-sharded banks** of `ame_h20` (`mem_parallel_size: 4`):
  `src/core/memory_parallel.py` slices each rank's columns from the full bank, drawn
  before at `bank_init_std`. No model is built on the meta device.

## Tests

| Suite | Where | Result |
|---|---|---|
| CPU gate, `pytest -m "not gpu and not slow and not network"` | local | running |
| `pytest -m gpu` (13 tests, all in `tests/test_ame.py`) | LEM interactive node | running |
| `tests/test_distributed.py` on CUDA with NCCL | LEM interactive node | running |

## Smoke runs

## Reproductions at h12

## Log

| Time (11-21) | Event |
|---|---|
| 16:25–16:33 | Steps 1–6: worktree, merge, cherry-pick, the `llama.yaml` comment, the readout fix, the six configs |
| 16:34 | CPU suite started locally |
| 16:36 | LEM: `sbatch --test-only` forecasts 11-25 for 1- and 4-GPU jobs; 166 jobs pending, 20 nodes idle and powered down. The settle runs started 7–24 h after submission, sooner than forecast |
| 16:40 | GPU tests started under `srun --pty` on the interactive partition (`sbatch` is refused there), from `standard_gputest_db1cdda2` in the project dir, with the `.pylib` pytest copy |
