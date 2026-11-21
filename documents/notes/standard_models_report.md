# Standard models: build, test and merge

**Status (2026-11-21 16:51):** in progress. Branch `standard_models` holds every
approved feature and the six standard configs (steps 1–6 done), and the tests pass
(step 7). The six smoke runs and the two h12 reproductions are queued on LEM (steps 8–9). Plan: [standard_model_improvements.md](standard_model_improvements.md#implementation-plan).

## The branch

Worktree `~/code/nano-standard`, branch `standard_models`, from `ame_main` at `7f4df31f`.

| Commit | What |
|---|---|
| `957a5349` | Merge of `ame_improve`: `62a59d2f` (`n_tokens_b`), `7762da98` (QK-norm), `e64a466c` (`bank_lr_ratio`), `7345eebc` (dense KV shift) |
| `fcd9ae21` | Cherry-pick of `83ebaa85`, the `run_exp.py` missing-key fix |
| `c491b57c` | The `bank_lr_ratio` comment in `configs/_trainer/llama.yaml`: old runs need 1 |
| `8b78b805` | The FSDP2 readout fix, with its test |
| `db1cdda2` | `configs/standard/`: the six standard configs |
| `41b041db` | The ST gate's init draws on the CPU: a GPU test that also fails on `ame_main` ([Tests](#tests)) |

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

GPU tests on LEM's interactive node r18-15 (H100, torch 2.7.1+cu128), under `srun --pty`
(`sbatch` is refused on that partition), with pytest from a `.pylib` copy.

| Suite | Tree | Result |
|---|---|---|
| CPU gate, `pytest -m "not gpu and not slow and not network"` | `db1cdda2` | **494 passed**, 18 skipped, 0 failed (4 min 29 s) |
| `pytest -m gpu` (13 tests, all in `tests/test_ame.py`) | `db1cdda2` | 12 passed, 1 failed: `test_on_cuda_every_gate_trains_under_compile_blocks_full[stem]` (job 6383495) |
| The same ST test | `ame_main` (`eeae4448`) | **Fails the same way**: it predates the merge (job 6383514) |
| `pytest -m gpu` | `41b041db`, with the ST fix | **13 passed** (job 6383514) |
| `tests/test_distributed.py`, on CUDA with NCCL there | `db1cdda2`, `41b041db` | **4 passed** both times, the readout test included |

**The ST failure and its fix (`41b041db`).** The ST gate's init made its draw with
`torch.empty(datt, dmodel)`, which follows the default device, so under
`torch.device("cuda")` the draw landed on CUDA while `_private_generator()` is a CPU
generator: "Expected a 'cuda' device type for generator but found 'cpu'". The draw is now
made on the CPU and copied to the gate's device. On the CPU path, which every ST run used,
nothing changes. The ST gate stays rejected and off by default; the one-line fix only makes
the GPU suite pass.

## Submission

All eight jobs come from `41b041db`, in the two halves of
[settle_runs_report.md](settle_runs_report.md#submission): `run_exp.py`'s own steps
locally with the secrets unset, then on LEM a sparse clone of each cemetery branch
(`run_exp.clone_command`, without `tests/`, `scripts/` and `documents/`), checked, and
`sbatch exp.job` from a shell that read the W&B key from stdin.

The smokes and the reproductions go in together. The settle runs waited 7–24 h in LEM's
queue, so holding the reproductions until the smokes pass could cost a day. If a smoke
fails, the reproductions are mine to cancel before they start, at no GPU cost (as in
[stage_1c_report.md](stage_1c_report.md), decision 3).

## Smoke runs

Each config with `trainer.n_steps=60 trainer.eval_interval=20`, no checkpoint
(`trainer.checkpoint.save.path=null`), heavy metrics every 20 steps (so AME's readouts
log at steps 0, 20 and 40), a 1.5 h limit, named `smoke_std_<config>` and tagged
`smoke`. Warmup is then 0 steps, as in the Stage 1c smokes.

| Config | Cemetery branch | Job | W&B | Result |
|---|---|---|---|---|
| `ame_h12` | `smoke_std_ame_h12_2026-11-21_16-45-38` | 6383541 | | pending |
| `dense_h12` | `smoke_std_dense_h12_2026-11-21_16-45-55` | 6383542 | | pending |
| `ame_h16` | `smoke_std_ame_h16_2026-11-21_16-46-09` | 6383543 | | pending |
| `dense_h16` | `smoke_std_dense_h16_2026-11-21_16-46-24` | 6383544 | | pending |
| `ame_h20` | `smoke_std_ame_h20_2026-11-21_16-46-39` | 6383545 | | pending |
| `dense_h20` | `smoke_std_dense_h20_2026-11-21_16-46-53` | 6383546 | | pending |

## Reproductions at h12

The configs as committed. References: `zdmv3qci` (AME; step 0 loss 11.25, 3.762 s/step,
`final_ce` 3.0312) and `chne4w91` (dense; 11.1875, 3.444 s/step, 2.9561). Both references
show `failed` in W&B: they failed at the end-of-run checkpoint save, over the file quota of
2026-11-19, after `final_ce` was logged ([settle_runs_report.md](settle_runs_report.md)).

| Config | Cemetery branch | Job | W&B | Step 0 | Last-500 paired gap | `final_ce` |
|---|---|---|---|---|---|---|
| `ame_h12` | `std_ame_h12_lr8_2026-11-21_16-47-08` | 6383547 | | | | |
| `dense_h12` | `std_dense_h12_lr8_2026-11-21_16-47-23` | 6383548 | | | | |

## Log

| Time (11-21) | Event |
|---|---|
| 16:25–16:33 | Steps 1–6: worktree, merge, cherry-pick, the `llama.yaml` comment, the readout fix, the six configs |
| 16:34–16:38 | CPU gate on `db1cdda2`: 494 passed, 0 failed |
| 16:36 | LEM: `sbatch --test-only` forecasts 11-25 for 1- and 4-GPU jobs; 166 jobs pending, 20 nodes idle and powered down. The settle runs started 7–24 h after submission, sooner than forecast |
| 16:37–16:39 | GPU tests under `srun --pty` on the interactive partition (`sbatch` is refused there), from `standard_gputest_db1cdda2` in the project dir, with the `.pylib` pytest copy: 12 of 13, the ST test fails; `test_distributed.py` 4 of 4 on CUDA |
| 16:40–16:42 | ST draw fixed (`41b041db`). One `srun`: the ST test fails on `ame_main`'s tree too; on `41b041db` all 13 GPU tests and `test_distributed.py` pass. The three test dirs (11.5k files) deleted afterwards |
| 16:41–16:45 | CPU gate on `41b041db`: 494 passed, 0 failed. **Step 7 passes** |
| 16:45–16:47 | Local halves of all eight jobs; the cemetery branches pushed, `standard_models` restored at `41b041db` |
| 16:48 | Clone check: GitHub refused port 22 from LEM's login node after two clones. Retried with sparse clones, 20 s apart; the first two made sparse after the fact |
| 16:48–16:51 | All eight clones checked: sparse (138 files each), `#SBATCH` lines and configs as intended, no live secret export, both fixes in `src/core/ame.py`. Submitted at 16:51, the W&B key on stdin: smokes 6383541–6383546, reproductions 6383547 (AME) and 6383548 (dense). All pending. A watcher polls `sacct` every 5 min |
