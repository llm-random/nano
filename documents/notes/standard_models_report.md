# Standard models: build, test and merge

**Status (2026-11-22 13:35): not merged; an AME rerun is queued on entropy.** Steps 1–8
pass, and so does the dense reproduction (gap +0.0075). The AME reproduction spiked at
steps 330–360 and ends 0.077 above `zdmv3qci`, outside the 0.022 band, so under the
plan's step 10 the merge waited for Maciej. For this config the merged code computes what
the reference's code did, so a chance spike is the likely cause, but one run cannot prove
it. At 13:2x Maciej answered "continue", taken as option 1 of [For Maciej](#for-maciej):
one rerun, and a merge if it ties `zdmv3qci` ([Rerun](#rerun-on-entropy)). Plan: [standard_model_improvements.md](standard_model_improvements.md#implementation-plan).

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
| `5684bea6` | The readout test's comment: alpha/beta are drawn, only tau starts at a fixed value (after the jobs were submitted; test only) |

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
| `ame_h12` | `smoke_std_ame_h12_2026-11-21_16-45-38` | 6383541 | `16jv7t55` | **Passes.** Step 0 11.25 (= `zdmv3qci`); 3.77 s/step (reference 3.762); peak 70.4 GiB; `final_ce` 7.164 logged; no NaN. Logs α/β, τ, the horizons, entropy and top-8 mass; `bank_lr` = 4 × `lr`. 9.5 min on r18-1 |
| `dense_h12` | `smoke_std_dense_h12_2026-11-21_16-45-55` | 6383542 | `tjhhov4d` | **Passes.** Step 0 11.1875 (= `chne4w91`); 3.45 s/step (reference 3.444); peak 53.3 GiB; `final_ce` 7.166 logged; no NaN. 7 min on r14-3 |
| `ame_h16` | `smoke_std_ame_h16_2026-11-21_16-46-09` | 6383543 | `40al2ex5` | **Passes.** Step 0 11.25; 6.77 s/step (a full run: 14.4 h of its 24 h limit); peak 72.2 GiB; `final_ce` 7.559 logged; no NaN. Logs all the readouts in all 16 layers; `bank_lr` = 4 × `lr`. 10.8 min on r18-1 |
| `dense_h16` | `smoke_std_dense_h16_2026-11-21_16-46-24` | 6383544 | `ckw39rr4` | **Passes.** Step 0 11.25; 6.27 s/step (a full run: 13.3 h of its 24 h limit); peak 43.3 GiB; `final_ce` 7.125 logged; no NaN; the final LR is 10% of 2^-9. 9.9 min on r14-3 |
| `ame_h20` | `smoke_std_ame_h20_2026-11-21_16-46-39` | 6383545 | `ad7fdm5h` | **Passes.** Under FSDP2 with the banks column-sharded over 4 GPUs it logs `alpha_2/mean`, `beta_2/mean`, `tau/mean` (≈ 8 = √64) and the horizons in all 20 layers, and no attention entropy, as intended: **the readout fix works**. Step 0 11.1875; 2.96 s/step (a full run: 12.6 h of its 24 h limit); peak 64.7 GiB, so `gradient_accumulation_steps` stays 8; `final_ce` 7.201 logged; no NaN; `bank_lr` = 4 × `lr`. 7.4 min on r14-6 |
| `dense_h20` | `smoke_std_dense_h20_2026-11-21_16-46-53` | 6383546 | `8r3012vt` | **Passes.** Step 0 11.25; 2.52 s/step on 4 GPUs (a full run: 10.7 h of its 24 h limit); peak 50.9 GiB; `final_ce` 7.406 logged; no NaN; the final LR is 10% of 2^-9. 5.9 min on r14-6 |

**Step 8 passes:** all six configs train, log `final_ce` and stay finite, and the h20 AME
run logs `alpha_2/mean` and `tau/mean` under FSDP2. No run ran out of memory, so no
`gradient_accumulation_steps` changed.

## Reproductions at h12

The configs as committed. References: `zdmv3qci` (AME; step 0 loss 11.25, 3.762 s/step,
`final_ce` 3.0312) and `chne4w91` (dense; 11.1875, 3.444 s/step, 2.9561). Both references
show `failed` in W&B: they failed at the end-of-run checkpoint save, over the file quota of
2026-11-19, after `final_ce` was logged ([settle_runs_report.md](settle_runs_report.md)).

| Config | Cemetery branch | Job | W&B | Step 0 | Last-500 paired gap | `final_ce` |
|---|---|---|---|---|---|---|
| `ame_h12` | `std_ame_h12_lr8_2026-11-21_16-47-08` | 6383547 | `zs1bceoy` | 11.25: **matches**, and so do steps 1–7 | **+0.0771: fails** (band 0.022); a spike from step 335, [below](#why-the-ame-reproduction-fails) | 3.1035 (reference 3.0312) |
| `dense_h12` | `std_dense_h12_lr8_2026-11-21_16-47-23` | 6383548 | `q2yrtd9t` | 11.1875: **matches**, and so do steps 1–7 | **+0.0075: passes** | 2.9629 (reference 2.9561) |

Both jobs completed with no error in either log, and both saved their final checkpoints:
AME ran 4 h 15 min on r18-14, dense 3 h 53 min on r18-4. GPU use for the whole plan is
about 9.8 GPU-h of the 10 budgeted.

### Why the AME reproduction fails

| Check | Finding |
|---|---|
| Where the gap opens | Steps 300–329: +0.003, a tie. Then the grad norm climbs from 0.12 at step 300 to 0.38 at step 342 and 3.0 at step 356, where the loss peaks at 7.47 (reference 4.03). After step 100 the reference's grad norm never passes 0.39. 63 steps between 335 and 714 meet `paired_gap.py`'s spike rule (0.1 above the median of the 50 steps before); the reference has none |
| Recovery | Gap +0.25 at steps 500–599, +0.18 at 1000–1099, +0.10 at 2000–2099, +0.08 at 3000–3099, +0.077 over the last 500. It is still closing at the end, too slowly |
| Readouts | At step 300, every layer's τ and α₂ equal the reference's to within 0.02. At step 400, after the spike, τ is up by about 0.2 and α₂ by 0.1–0.15 in layers 1–11. They are logged every 100 steps, so they show the spike's aftermath, not its start |
| Config | Equal in every training key: seed 123, `distributed: null`, clipping 1, `bank_lr_ratio` 4, σ 1e-4 and the compile flags. The diff is the start check's list ([Log](#log), 23:35) |
| Code | The run commits are `482aaf56` (reference: `settle_runs_improve`, `08b75753`) and `43872cbc` (`41b041db`). The training files differ by four things. First, the bank gates: their default `none` is the old forward, op for op, and draws nothing from the global RNG. Second, the `clip_factor` and `grad_norm/banks` readouts, computed after the clip. Third, the token counts, which are off. Fourth, the FSDP2 readout fix, idle without sharding (the ST init fix is `stem`-only and part of the gate code). `pixi.toml` and `pixi.lock` are unchanged |
| Determinism | None to expect: the two AME runs' losses first differ at step 20, the dense pair's at step 12 |

**Reading:** the merge most likely did not cause this. Two runs of the same computation take
different paths, and this one hit a spike the reference missed. The recipe (h12, σ 1e-4,
`bank_lr_ratio` 4, 2⁻⁸) has now spiked in 1 of its 2 runs. With one run on each side, a
regression cannot be ruled out, and the plan says not to merge after a failure.

### For Maciej

1. **Rerun `ame_h12` from `standard_models` once** (recommended), about 4.3 GPU-h. If it
   ties `zdmv3qci`, merge. If it spikes again, run the reference's code (`482aaf56`) on the
   same cluster to tell the code from the recipe. Where to run it:
   - On LEM, the budget is spent (about 9.8 of 10 GPU-h), so this takes it to about 14.
   - On entropy, the 8-GPU cap is full of the rebuttal's E3X jobs (1 running, 10 pending,
     deadline 11-23). A rerun would wait behind them or slow them, and it would compare
     across clusters.
2. **Merge on this evidence:** for this config, the code that differs never runs or never
   changes what is computed.

Nothing has been launched. The worktree `~/code/nano-standard` stays for the merge.

### Rerun on entropy

Maciej's "continue" (11-22, about 13:25) is taken as option 1. The rerun goes to entropy:
LEM's budget is spent, and Maciej asked on 11-21 for tests to run there.

- **Job:** 44202, from `standard_models` at `5684bea6` (its only change since `41b041db` is
  a test comment). Cemetery branch `std_ame_h12_lr8_rerun_2026-11-22_13-27-26`, run
  `std_ame_h12_lr8_rerun`, one H100, a 7 h limit.
- **Config:** a copy of `ame_h12.yaml`, because the cluster can't be switched on the command
  line. Composed, it differs only in cluster keys: data paths (entropy's FineWeb has the
  same 1,279,107,432 rows), cemetery, slurm settings, name and tags. Every model and trainer
  key is equal. The copy is in the cemetery commit only, never in `standard_models`.
- **The rebuttal keeps its GPUs:** `--nice=10000`, so the job ranks below every E3X job
  (priority 1 against 3875). One E3X job holds 4 of h100a's GPUs. `pzal`'s 2 GPUs can't be
  preempted, so E3X can't use more than 6, and a 1-GPU rerun fits beside it. Only if
  `pzal`'s jobs end during the rerun would the 8-GPU cap make a second E3X job wait for it.
- **Pass rule:** as in step 9, step 0 equal to the reference's and a last-500 paired gap
  under 0.022 against `zdmv3qci`. A tie means merge. A second spike means rerunning the
  reference's code (`482aaf56`) on entropy to separate code from recipe. The comparison
  crosses clusters (LEM reference, entropy rerun); earlier cross-cluster PC pairs differed
  by about ±0.002.

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
| 21:43–21:53 | Smoke `ame_h12` (6383541) on r18-1, after 4 h 52 min in the queue: passes ([Smoke runs](#smoke-runs)). The other seven still pending |
| 21:54 | The readout test's comment fixed on `standard_models` (`5684bea6`): alpha/beta are drawn (c1 ~ U(0, 1)), not (1, 0). Test only; the jobs run `41b041db` |
| 21:54–22:01 | Smoke `dense_h12` (6383542) on r14-3: passes. Smokes `ame_h16` and `dense_h16` started at 21:55 and 22:02 |
| 21:55–22:06 | Smoke `ame_h16` (6383543) on r18-1: passes |
| 22:02–22:12 | Smoke `dense_h16` (6383544) on r14-3: passes. The four 1-GPU smokes all pass; the h20 smokes and the reproductions still pending |
| 23:29 | Reproduction `ame_h12` (6383547, `zs1bceoy`) started on r18-14. Start check at 23:35: log clean; the W&B config differs from `zdmv3qci`'s only in the run name, tags, job name, `mem_per_gpu`, `sparse_exclude`, `count_token_updates: False` and the run paths; losses at steps 0–7 equal the reference's (11.25, 10.0625, 9.625, 9.5, 8.75, 8.5, 8.0625, 7.8125); 3.77 s/step (reference 3.762). About 4 h to go |
| 23:35–23:42 | Smoke `ame_h20` (6383545) on r14-6: passes, with the FSDP2 readouts. Smoke `dense_h20` started at 23:43 |
| 23:43–23:49 | Smoke `dense_h20` (6383546) on r14-6: passes. **Step 8 passes** |
| 23:48 | Maciej: use entropy for tests too, and move anything still pending on LEM there. Only the dense reproduction was pending. Prepared its move: entropy's FineWeb has the same 1,279,107,432 rows, GitHub answers, h100a's 8 GPUs held by 6 preemptible `lukaskoz` jobs and 2 `pzal` jobs. The cluster cannot be switched on the command line (Hydra's grammar rejects the `../_cluster@_global_` key), so the move would have used a copy of `dense_h12.yaml` |
| 23:51 | Before the move, LEM started the dense reproduction (6383548) on r18-4: nothing pending is left, so nothing moved. It stays on the cluster of its reference |
| 23:59 | Start check of the dense reproduction (`q2yrtd9t`): log clean; the W&B config differs from `chne4w91`'s only in the run name, tags, job name, `mem_per_gpu`, `sparse_exclude`, `count_token_updates: False` and the run paths; losses at steps 0–7 equal the reference's (11.1875, 9.875, 9.375, 9.25, 8.75, 8.4375, 8.0625, 7.90625); 3.443 s/step (reference 3.444). Both reproductions should end at about 03:35 |
| 11-22 03:43–03:44 | Both reproductions end (COMPLETED), with final checkpoints and no error in either log: dense (6383548) after 3 h 53 min, AME (6383547) after 4 h 15 min |
| 11-22 03:50 | Paired gaps: dense +0.0075, **passes**; AME +0.0771, **fails**, with a spike from step 335 |
| 11-22 03:52–04:00 | Diagnosis ([Why the AME reproduction fails](#why-the-ame-reproduction-fails)). No rerun: LEM's budget is spent, and the rebuttal's jobs fill entropy's cap. **Not merged**; the options go to Maciej |
| 11-22 13:2x | Maciej: "continue", taken as option 1 (rerun once, merge on a tie) |
| 11-22 13:27–13:29 | Entropy copy of `ame_h12.yaml` composed and diffed (cluster keys only); local half from `5684bea6`; sparse clone checked on entropy (139 files, `--nice=10000`, `gpu:h100:1`, no live secret); submitted as 44202 with the W&B key on stdin. Pending at priority 1, behind the E3X jobs. The copy deleted from the worktree. A watcher polls entropy's `sacct` every 5 min |
