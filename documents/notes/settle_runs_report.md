# Settle runs: the open items of the standard-model digest

**Status:** done, 2026-11-20: all six items have results.
- All eight jobs of 11-18 hit LEM's file quota on 11-19.
  - Six had already trained to their last step and logged `final_ce`, so their results
    stand ([Results](#results)).
  - The two single-switch runs died at their start. They were resubmitted on 11-20 at
    10:40 as jobs 6370656 and 6370657, and completed at 21:28 and 21:33.
- The plan below was written before any result; the log and the results follow it.

This note settles the "Runs that would settle what is left" table of
[standard_model_improvements.md](standard_model_improvements.md). Maciej set the task on
2026-11-18 with a `/goal`:
- Run each open item on LEM: 1 GPU, r16-11 excluded, seed 123.
- Start from the reference's logged W&B config and change only the listed setting, so
  each run pairs step by step with its reference.
- Check step 0 against the reference, and submit every job at once from clean worktrees.
- Judge by the last-500-step paired gap and `final_ce`; a gap under 0.022 is a tie.
- Record the results here and in the digest.

## The runs

| # | Question | Run (W&B name) | The change against its reference | References (`final_ce`) |
|---|---|---|---|---|
| 1 | Do the two bank levers stack? | `ai_h12_ps_bank4_std1e4_lr8` | `bank_lr_ratio` 4 at σ = 1e-4; 3,815 steps | `7te9jv9u` (σ 1e-4, ratio 1: 3.0400), `kf4q4t4s` (σ 0.01, ratio 1: 3.0352) |
| 2 | The small init's own gain at h16, and how much of ST's h16 lead it explains | `bank_gates_h16_none_std001` | σ = 0.01, no gate | `hjw7znbz` (σ 1, no gate: 2.8594); the gate arms at σ 0.01, `ug8g2i2x` (ST: 2.7520) and `3yc7haqu` (B3: 2.8105) |
| 3 | Where the σ grid's lower edge lies (σ = 1e-8) | done by the σ-grid session, see below | — | — |
| 4 | Is the document mask's ≈0.02 real? | `dm_ame_h12_ps_lr8_rep`, `dm_dense_h12_lr8_rep` | none: same-seed replicates, on LEM instead of entropy | masked `j6wxx6h2` (3.1084), `k3xx6w71` (2.9639); unmasked `doi36p8k` (3.1250), `x0xlqroj` (2.9824) |
| 5 | Do QK-norm and the KV shift help dense h12, alone or together, and what do they cost per step? | `ai_dense_h12_qkn_shift_lr8` (both), `ai_dense_h12_qkn_lr8` (QK-norm only), `ai_dense_h12_shift_lr8` (the shift only) | `attention.qk_norm: true`, `attention.shift: true`, or both. With `x0xlqroj` (neither), the full 2 × 2 (Maciej, 2026-11-18) | `x0xlqroj` (2.9824) |
| 6 | Dense h16's best learning rate, at the grid's edge | `s1b_d_h16_lr8` | n = 8 | `eh2sjsh7` (n = 9: 2.7441) |

All runs are h12 at 4B tokens (3,815 steps) or h16 at 8B (7,630), batch 512 × 2,048,
LR 2⁻⁸, seed 123, one LEM H100 each, no FSDP2 (so AME's readouts are logged).

### Configs and code

| # | Branch (worktree) | Commit | Config |
|---|---|---|---|
| 1, 5 | `settle_runs_improve` (`~/code/nano-settle-improve`), from `ame_improve_runs` `43a030e0` | `08b75753`; `01650535` adds the single-switch configs | `configs/ame_improve/ame_h12_ps_bank4_std1e4.yaml`; `configs/ame_improve/dense_h12_qknorm_shift.yaml`, `dense_h12_qknorm.yaml`, `dense_h12_shift.yaml` |
| 2, 6 | `settle_runs_main` (`~/code/nano-settle-main`), from `ame_main` `4110b129` | `6929902a` | `configs/bank_gates/h16_none_std001.yaml` (on `h16_base.yaml`); `configs/stage_1b/dense_h16_lr_grid.yaml` with its list narrowed to `[8]` |
| 4 | `settle_runs_doc_mask` (`~/code/nano-settle-docmask`), from `doc_mask` `ef68bc1d` | `b2dc57f8` | `configs/doc_mask/ame_h12_ps_docmask_rep.yaml`, `configs/doc_mask/dense_h12_docmask_rep.yaml` |

Each config was composed as `run_exp.py` composes it. Every grid point passed
`check_point`, and its resolved config was diffed against the references' logged W&B
configs. Apart from runtime keys (`run_env`, `infrastructure`, checkpoint paths), only
these differ:
- **1** against `7te9jv9u`: `bank_lr_ratio` 4. Other diffs:
  - schema keys `ame_improve_runs` has and the σ-grid tree lacks. They are all off or
    None: `compile_ce`, `compile_loss`, `fused_linear_ce`, `fused_ce_chunk_size`,
    `kernel_profile`, `profile_timers`, `skip_grad_sync_during_accumulation`, the
    `qk_norm` and `shift` switches, and `n_tokens_b`;
  - `bank_gate` absent: the `ame_improve_runs` tree has no gates, and the gates commit
    `157c3122` is bit-identical at `none`;
  - a final checkpoint saved, as `y5ogvt3k` did.

  Against `y5ogvt3k` the only changes are `bank_init_std` 1e-4, and `n_steps` 3,815
  where it had 3,812 from `n_tokens_b: 4`.
- **2** against `ug8g2i2x` and `3yc7haqu`: `bank_gate` (`none` against `stem` and
  `backoff`) and the arm name. Against `hjw7znbz` the diffs are `bank_init_std` 0.01,
  `bank_gate: none` and `count_token_updates: true`. No file under `src/` or `main.py`
  has changed since `bank_gates_lem`, where the gate runs were submitted.
- **4** against `j6wxx6h2` and `k3xx6w71`: the dataset paths only. They point at LEM's
  FineWeb, of which entropy's is a copy. The code is `3974addd`'s, since `doc_mask` has
  no code commit after it.
- **5** against `x0xlqroj`: `qk_norm: true`, `shift: true` and `bank_lr_ratio` 4.0. The
  single-switch runs set only their own switch; the other stays at the gqa default, off.
  - The ratio is a no-op without banks (`parameter_groups`, `src/core/model.py`).
  - The schema keys listed under 1 are all off.
  - The dense τ (QK-norm's learned temperature) and the shift's coefficients sit in a
    no-decay group, by design.
- **6** against `eh2sjsh7`: `learning_rate` 8 against 9. Also `count_token_updates:
  false` and the schema keys listed under 1, all off. The name is pinned to
  `s1b_d_h16_lr8`; the grid's template would have given `lr8.0`.

### Item 3 was done before this goal ran

The σ-grid session ran σ = 1e-8 as round 5: LEM job 6348965, W&B `n14uvt1n`, finished
2026-11-18 04:57.
- Paired gap +0.1828 against σ = 0.01 (`plabns2v`), `final_ce` 3.2178.
- The chart was redrawn with 15 points.
- It is recorded in [ame_bank_sigma_grid.md](ame_bank_sigma_grid.md) (`a8efa353`), so
  this goal submitted nothing for it.
- That session has since queued round 6 (σ = 1e-6, job 6353662), which is not part of
  this goal.

## How each run is judged

- **Paired gap:** the mean of (run − reference) `steps/train/loss` over the last 500
  steps, from `scripts/paired_gap.py` (branch `bank_sigma_grid`). Same seed means the
  same batches at the same step, so the gap is finer than `final_ce`.
- **`final_ce`:** the held-out loss, bf16-quantized, so ±0.008 per run.
- **Tie:** a gap under 0.022 in size (3 × AME's same-seed sd of 0.0072 at h12). Dense
  same-seed spread is 6–24× larger, so a dense gap is weaker evidence. No noise floor
  has been measured at h16, so the h12 one stands in.
- **The step-0 check:** step 0's train loss must equal the reference's.
  - AME 11.25, at h12 and h16 alike: under the hard QK-norm, σ only scales the bank
    rows, so the init forward is the same at any σ.
  - Dense h12 11.1875, dense h16 11.25.
  - The three run-5 corners are the exception: their step 0 is a new number.
    - QK-norm changes the init forward: unit q and k, scaled by τ = √d_head.
    - The shift's coefficients start at c1 ~ U(0, 1) and c2 = 1 − c1, not at the
      identity. They are drawn from the global RNG after the layer's own projections,
      so the later layers' weights may differ from `x0xlqroj`'s as well.

What each outcome means:
1. **Run 1 against `7te9jv9u`** (they differ only in the ratio):
   - a gap ≤ −0.022 means the levers stack;
   - a tie means ratio 4 adds nothing measurable on top of σ = 1e-4;
   - a gap ≥ +0.022 means it hurts there.
2. **Run 2:**
   - against `hjw7znbz`, it is the small init's own gain at h16;
   - ST against it is ST's gain beyond the init;
   - B3 against it is B3's gain beyond the init. At h12 that was +0.0001.
4. **Run 4:**
   - every run here uses seed 123, so a replicate has its original's init and sees the
     same batches in the same order. Only GPU nondeterminism differs, and here the
     cluster too. That nondeterminism is the 0.0072 sd behind the 0.022 tie band, so a
     replicate tests whether the ≈0.02 survives it. It does not test another seed: that
     would need an unmasked baseline at the new seed as well;
   - each replicate against its unmasked LEM baseline is the mask's gain measured on
     one cluster;
   - each replicate against its masked original is the same-seed spread across
     clusters. Entropy runs of `doi36p8k`'s config paired at +0.0157, +0.0002 and
     +0.0053 against it (`reference_final_ce_precision`);
   - the ≈0.02 is confirmed if both replicates come out negative again, at about −0.02.
5. **Run 5, the 2 × 2 against `x0xlqroj`:** each corner's paired gap, and its s/step
   against 3.272.
   - Each single switch against `x0xlqroj` is that switch's effect alone.
   - Both together against the better single switch shows whether the second switch
     adds anything.
   - The interaction is (both − neither) − (QK-norm − neither) − (shift − neither).
   - Dense same-seed runs spread 6–24× more than AME's, so only a large gap is
     evidence.
6. **Run 6 against `eh2sjsh7`:**
   - if n = 8 is lower, dense h16's best n is 8 or below, and n = 7 would be next;
   - if it is higher, n = 9 is an interior optimum and the grid is done.

**Baselines for the health checks** (the references' logs): AME h12 runs about 3.75
s/step and 4.2–4.4 h; dense h12 3.27 s/step and 3.7 h (3.32 with the mask); AME h16
15.0–15.6 h; dense h16 13.2 h.

## Submission

Two halves, as in [stage_1c_report.md](stage_1c_report.md) ("Submission"). LEM's
cemetery is world-readable, so the W&B key never goes into `exp.job`.
- **Local half:** `run_exp.py`'s own steps, called by a scratch script in each
  worktree, with the secrets unset. In order: compose, the missing-key check,
  `create_grid_config`, `check_point`, `dump_grid_configs` into a cleared
  `generated_configs/`, `generate_sbatch_script`, and `version_code`, which pushes the
  cemetery branch. `exp.job` then carries the two secret export lines commented out.
- **Remote half:** clone each branch into
  `/lustre/pd03/plgrid/plgllmefficont3/nano/plgmstefaniak/llmrandom_cemetery`. Check
  each clone: its configs, the `#SBATCH` lines, no live secret export and no Slurm log.
  Then submit all six with `sbatch exp.job` from one shell that read the key from stdin.

## Runs

"Done, then quota" means the job trained to its last step and logged `final_ce`, then
failed at the end-of-run checkpoint save, over the file quota. The result stands.

| # | Job | Cemetery clone | W&B | Node | Started | Ended | State | `final_ce` | Gap, last 500 |
|---|---|---|---|---|---|---|---|---:|---:|
| 1 | 6353710 | `ai_h12_ps_bank4_std1e4_lr8_2026-11-18_14-14-15` | `zdmv3qci` | r14-1 | 11-19 12:22 | 16:38 | Done, then quota | 3.0312 | −0.0083 against `7te9jv9u` |
| 5, both | 6353711 | `ai_dense_h12_qkn_shift_lr8_2026-11-18_14-14-43` | `chne4w91` | r18-1 | 11-19 13:12 | 17:05 | Done, then quota | 2.9561 | −0.0271 against `x0xlqroj` |
| 5, QK-norm only | 6353892 | `ai_dense_h12_qkn_lr8_2026-11-18_14-38-16` | — | r10-8 | 11-19 17:05 | 17:05 | FAILED at start: 0 s, no log (quota) | | |
| 5, shift only | 6353893 | `ai_dense_h12_shift_lr8_2026-11-18_14-38-39` | — | r10-8 | 11-19 17:06 | 17:06 | FAILED at start: 0 s, no log (quota) | | |
| 5, QK-norm only | 6370656 | the same clone | `8lqp6pk3` | r18-1 | 11-20 17:40 | 21:28 | COMPLETED: resubmitted 11-20 10:40, 5 h limit | 2.9941 | +0.0132 against `x0xlqroj` |
| 5, shift only | 6370657 | the same clone | `aak55dd1` | r10-2 | 11-20 17:45 | 21:33 | COMPLETED: resubmitted 11-20 10:40, 5 h limit | 2.9629 | −0.0216 against `x0xlqroj` |
| 2 | 6353712 | `bank_gates_h16_none_std001_2026-11-18_14-15-12` | `eeaz5uy1` | r10-9 | 11-19 13:13 | 11-20 04:09 | Done, then quota | 2.7959 | −0.0798 against `hjw7znbz` |
| 6 | 6353713 | `s1b_d_h16_lr8_2026-11-18_14-15-34` | `ezzyvpn8` | r14-3 | 11-19 13:22 | 11-20 02:34 | Done, then quota | 3.4346 | +0.7544 against `eh2sjsh7` |
| 4 AME | 6353714 | `dm_ame_h12_ps_lr8_rep_2026-11-18_14-16-02` | `h53srjtk` | r14-3 | 11-19 13:23 | 18:04 | Done, then quota | 3.1084 | −0.0014 against `j6wxx6h2`; −0.0177 against `doi36p8k` |
| 4 dense | 6353715 | `dm_dense_h12_lr8_rep_2026-11-18_14-16-22` | `igfkre78` | r14-12 | 11-19 13:37 | 17:45 | Done, then quota | 2.9629 | −0.0019 against `k3xx6w71`; −0.0222 against `x0xlqroj` |

## Log

- **2026-11-18, 14:05–14:21.**
  - Three worktrees and branches were made, the configs written and committed, and each
    diffed against its references (above).
  - LEM reached GitHub in 2.4 s.
  - The six local halves ran 14:14–14:16; the clones were checked; all six went to
    `sbatch` at 14:21.
  - All six are PENDING. 161 jobs were pending in the partition and 72 running.
  - Next: each job's start check about 2 minutes after it starts, then step 0. After
    that, nothing until the end of training unless a job fails or logs a NaN.
- **14:30–14:40.** Maciej asked for every combination of the two dense switches, not
  only both together.
  - Two configs with one switch each, `dense_h12_qknorm.yaml` and `dense_h12_shift.yaml`,
    committed as `01650535`.
  - Each differs from `x0xlqroj` in its own switch only, besides the no-op ratio and the
    off schema keys.
  - Submitted at 14:40 in the same two halves: QK-norm only as job 6353892, the shift
    only as 6353893. Both PENDING.
  - Item 5 is now a 2 × 2, with `x0xlqroj` as the corner with neither switch.
- **16:04, time limits (Maciej asked).** All eight jobs are still PENDING on Priority.
  Slurm forecasts their starts for 11-20 07:40 to 11-21 08:16.
  - **Limits as submitted:** 7 h for the six h12 jobs, 24 h for the two h16 jobs.
  - **Measured LEM wall times of the same recipes** (sacct, completed runs since
    11-05):

    | Recipe | Wall time | Runs |
    |---|---|---|
    | AME h12 | 4:12–4:26 | about 25 |
    | Dense h12 | 3:40–3:47 | |
    | AME h16 | 14:59–15:35; the ST gate is the slowest, ungated runs about 15:05 | |
    | Dense h16 | 13:10–13:15 | |

  - **Shorter limits don't move Slurm's forecast.** `sbatch --test-only` gives a new
    job the same start, 11-25 08:04, at 24 h, 18 h, 16 h, 7 h, 5 h 30 min and 5 h. The
    queue is bound by priority (jobs ahead of ours), so a shorter limit helps only when
    a backfill hole opens.
  - **Proposed, not applied:** AME h12 5 h 30 min, dense h12 5 h, AME h16 18 h, dense
    h16 16 h, at least 18% above the slowest comparable run.
    - It would be done in place with `scontrol update`, which keeps the queue position.
    - A user can only lower a limit, never raise it again.
- **The watcher died with the previous session** at about 14:45, while every job was
  still pending, so nothing was missed. Restarted at 16:10.

- **11-19, all eight jobs FAILED.**
  - Six ran their full length (wall 3:53 to 14:56) and failed at the end.
  - 6353892 (QK-norm only) and 6353893 (the shift only) died at their start: 0 s, node
    r10-8.
  - Maciej's diagnosis: the team went over LEM's file quota. That is Lustre project 649,
    the group directory, with a hard limit of 1,048,576 inodes.
  - **11-20, 10:10:** the project held 812,701 inodes, 192,345 of them Maciej's:

    | What | Inodes |
    |---|---|
    | The cemetery, 349 job clones | 104,678 |
    | The pixi env | 70,770 |
    | The rattler cache's unshared part | 7,468 |
    | Other | 9,429 |

  - **Next:** Maciej's call on cleanup and resubmission. Whether W&B holds the six
    full-length runs' results is unchecked.
- **11-20, 10:10–10:40.**
  - **W&B holds all six full-length runs.** Each logged every step (3,815 or 7,630) and
    `final_ce`. The jobs failed after that, at the end-of-run checkpoint save, when the
    project ran out of inodes. The results stand.
  - **Step 0 matched in all six:** 11.25 for AME at h12 and h16 and for dense h16;
    11.1875 for dense h12. That includes the run with both dense switches, which equals
    `x0xlqroj`'s at bf16 resolution. The replicates also match their originals at step 1.
  - **Maciej: the file quota is fixed, resubmit what broke.**
    - The two single-switch runs never started: no Slurm log, no W&B run.
    - Resubmitted at 10:40 from their existing clones, the key on stdin: QK-norm only as
      6370656, the shift only as 6370657.
    - Each has a 5 h limit, following Maciej's point that shorter limits schedule faster.
      The slowest dense h12 run took 3:53.
  - The watcher was restarted for the two. Next: their start check, then the end of
    training.

- **11-20, 11:18, freeing the quota (Maciej asked).** Deleted from all 349 job clones in
  `mstefaniak/llmrandom_cemetery`: `configs/`, `tests/`, `scripts/` and `documents/`.
  - 882 directories and 49,852 inodes.
  - Project 649 went from 812,792 to 762,940 inodes (72.8% of the limit).
  - A job never reads these folders: it runs with `--config-path=generated_configs`.
  - Each clone's `.git` can restore them with `git checkout`.
- **11-20, 17:40–17:50: both single-switch runs started.**
  - 6370656 started on r18-1 at 17:40 (W&B `8lqp6pk3`). At 2 minutes the log was clean:
    183,454,608 parameters, 12 blocks compiled, data loaded.
  - **Step 0 is 11.1875, the same as `x0xlqroj`'s.** Steps 1–3 equal its losses too
    (9.9375, 9.375, 9.3125), and the two first differ at step 4 (8.75 against 8.6875).
  - 6370657 (shift only) started on r10-2 at 17:45 (W&B `aak55dd1`). At 2 minutes the
    log was clean: 183,454,752 parameters. Over the 183,454,464 of the run without either
    switch, the shift adds 288: a (c1, c2) pair for keys and one for values per KV head,
    6 heads in each of 12 blocks. QK-norm adds 144, one τ per query head, and `chne4w91`
    carries both (183,454,896).
  - **Its step 0 is 11.1875 too.** Its steps 1–5 equal those of `chne4w91`, the run with
    both switches (9.875, 9.375, 9.25, 8.75, 8.4375). The shift sets the early losses, and
    QK-norm leaves them as in `x0xlqroj`.
  - Next: the end of training, about 21:30–21:40.
- **11-20, 21:28–21:45: both single-switch runs completed.**
  - 6370656 ended at 21:28 and 6370657 at 21:33, after 3:47 each. Both logged all 3,815
    steps and `final_ce`, saved their checkpoints and logged no error, so the file quota
    held.
  - Item 5's 2 × 2 is complete ([Results, 5](#5-dense-switches-the-kv-shift-carries-the-gain-qk-norm-adds-a-tie)).
  - With that, all six items have results.

## Results

Paired gap: the mean of (run − reference) train loss over the last 500 steps.
`final_ce` Δ is the difference of the bf16-rounded held-out losses. A tie is a gap under
0.022. Every health check below covers the steps after step 100.

### 1. The two bank levers barely stack

| Run | Against | Gap | `final_ce` Δ |
|---|---|---:|---:|
| `zdmv3qci`: ratio 4, σ 1e-4 | `7te9jv9u`: ratio 1, σ 1e-4 | −0.0083 | −0.0088 |
| | `kf4q4t4s`: ratio 1, σ 0.01 | −0.0051 | −0.0039 |
| | `plabns2v`: ratio 1, σ 0.01, LEM | −0.0133 | −0.0117 |

- **Ratio 4 on top of σ = 1e-4 is a tie: −0.008,** the size of one rerun's noise (0.008).
  At σ = 1 it gave −0.034.
- So the two levers mostly overlap. A small σ already lets Adam turn the bank rows fast,
  and a 4× rate adds little to that.
- **It costs nothing:**
  - s/step 3.762 against 3.768, memory 70.4 GiB;
  - no spike;
  - a lower grad-norm maximum: 0.39 against 0.56.
- **Reading:** ratio 4 is harmless in the standard AME config, but buys nothing
  measurable once σ is small.

### 2. At h16 the small init gains 0.08, and ST adds 0.05 beyond it

| Run | Against | Gap | `final_ce` Δ |
|---|---|---:|---:|
| `eeaz5uy1`: σ 0.01, no gate | `hjw7znbz`: σ 1, no gate | −0.0798 | −0.0635 |
| `ug8g2i2x`: ST, σ 0.01 | `eeaz5uy1` | −0.0478 | −0.0439 |
| `3yc7haqu`: B3, σ 0.01 | `eeaz5uy1` | +0.0170 | +0.0146 |

- **The init's own gain at h16 is −0.080,** about as large as at h12 (−0.0895 on the
  common evaluator).
- **ST's lead splits in two.** ST led `hjw7znbz` by 0.107 in `final_ce`.
  - The init explains 0.064 of that.
  - ST adds 0.044 on top (paired −0.048), twice the tie band. At h12 ST's gain beyond
    the init was a tie (−0.0167).
  - No h16 noise floor exists, so the h12 one stands in.
- **B3 adds nothing at h16 either:** +0.017, a tie. It also had a spike episode around
  steps 390–408 (grad-norm maximum 3.35).
- **ST's cost:** 7.02 s/step against 6.76 (+3.8%), and 1.3 GiB more memory.
- **The new run is healthy:** no spike, grad-norm maximum 0.56, 6.76 s/step (`hjw7znbz`
  6.81).

### 3. σ = 1e-8

+0.183 against σ = 0.01, from the σ-grid session's round 5 (above). Nothing was run for
it here.

### 4. The document mask's gain is real, but it costs 12% per step

| Run | Against | Gap | `final_ce` Δ |
|---|---|---:|---:|
| `h53srjtk`: AME masked, LEM | `j6wxx6h2`: masked original, entropy | −0.0014 | 0.0000 |
| | `doi36p8k`: unmasked, LEM | −0.0177 | −0.0166 |
| `igfkre78`: dense masked, LEM | `k3xx6w71`: masked original, entropy | −0.0019 | −0.0010 |
| | `x0xlqroj`: unmasked, LEM | −0.0222 | −0.0195 |

- **The replicates reproduce their originals almost exactly:** −0.0014 and −0.0019.
- **AME: the gain is resolved.**
  - The parity study holds three more unmasked runs of `doi36p8k`'s config, made on
    entropy (W&B `nano-ame-parity`).
  - Both masked runs come out below all four unmasked runs, in every one of the eight
    pairs: −0.016 to −0.033, mean −0.022.

  | Masked run | `doi36p8k` (LEM) | `u1zpx7lt` | `s65ebe41` | `w4hmw00o` |
  |---|---:|---:|---:|---:|
  | `j6wxx6h2` (entropy) | −0.0163 | −0.0319 | −0.0164 | −0.0215 |
  | `h53srjtk` (LEM) | −0.0177 | −0.0334 | −0.0179 | −0.0230 |

- **Dense: the gain reproduces,** −0.020 on entropy and −0.022 on LEM.
  - Both are against `x0xlqroj`, the only unmasked dense run of this config.
  - `x0xlqroj` had a spike episode at steps 194–203 that the masked runs lack. The gap
    shrinks from −0.15 in the first quarter to −0.022 at the end.
- **The cost, newly measured: about +12% s/step in both models,** within one cluster:

  | Model | Cluster | Unmasked s/step | Masked s/step |
  |---|---|---:|---:|
  | AME | LEM | 3.746 | 4.200 |
  | Dense | LEM | 3.272 | 3.681 |
  | AME | entropy | 3.36–3.38 (the parity runs) | 3.746 |

  - [doc_mask_report.md](doc_mask_report.md) reported AME "unchanged" and dense
    "+1.5%". Those numbers compared masked runs on entropy with unmasked runs on LEM,
    and entropy runs this workload about 10% faster.
- **A reading, untested.** The mask is a flex `score_mod`, a `where` per score, so it
  skips no work. A `BlockMask` built from the document boundaries could skip the
  cross-document blocks.

### 5. Dense switches: the KV shift carries the gain, QK-norm adds a tie

Each corner's paired gap against `x0xlqroj` (neither switch), with its `final_ce`:

| | No shift | Shift |
|---|---:|---:|
| **No QK-norm** | `x0xlqroj`: 0 (2.9824) | `aak55dd1`: −0.0216 (2.9629) |
| **QK-norm** | `8lqp6pk3`: +0.0132 (2.9941) | `chne4w91`: −0.0271 (2.9561) |

Each switch alone, and added to the other:

| Switch | Alone: gap (`final_ce` Δ) | Added to the other: gap (`final_ce` Δ) |
|---|---:|---:|
| Shift | −0.0216 (−0.0195) | −0.0403 (−0.0381), on QK-norm |
| QK-norm | +0.0132 (+0.0117) | −0.0055 (−0.0068), on the shift |

The paired gaps add up: −0.0216 − 0.0055 = +0.0132 − 0.0403 = −0.0271.

- **The shift carries the gain.**
  - Alone it gains −0.022, on the tie line (0.0216 against 0.022). Its gap is negative
    in every quarter: −0.090, −0.049, −0.028, −0.022.
  - On top of QK-norm it gains −0.040: −0.086, −0.069, −0.049, −0.041 by quarter.
- **QK-norm adds nothing measurable.**
  - Alone it is +0.013, a tie, and behind in every quarter: +0.059, +0.016, +0.014,
    +0.013.
  - On top of the shift it is −0.006, a tie: +0.063 in the first quarter, then −0.004 to
    −0.006.
  - So in both cases it costs about 0.06 early and then runs level.
- **The interaction is −0.019:** −0.0271 − 0.0132 + 0.0216. QK-norm helps only with the
  shift, and then only by a tie. This rests on four single-seed runs, so it is a hint.
- **Loss spikes:**
  - Neither run with the shift had one (grad-norm maxima 0.86 and 0.59).
  - Both runs without the shift had an episode: `x0xlqroj` at steps 194–203 (maximum
    2.58), and QK-norm only at steps 468–478 (20 spike steps, maximum 0.93).
  - **Caveat:** every pair that measures the shift has such an episode on its no-shift
    side. With one seed, it can't say whether the shift prevents the episodes or the two
    runs were unlucky. The one pair without an episode, both against the shift, is
    QK-norm's tie.
  - Dense same-seed runs spread more than AME's, so one corner alone is weak evidence;
    the pattern across the four is the stronger one.
- **Cost:** median s/step, one run each, on different LEM nodes; peak memory.

  | Run | s/step | Δ | Memory, GiB |
  |---|---:|---:|---:|
  | Neither (`x0xlqroj`) | 3.272 | | 47.0 |
  | QK-norm (`8lqp6pk3`) | 3.332 | +1.8% | 51.0 |
  | Shift (`aak55dd1`) | 3.362 | +2.7% | 49.3 |
  | Both (`chne4w91`) | 3.444 | +5.3% | 53.3 |

- **So the shift alone keeps −0.022 of the −0.027 at half the time cost.** QK-norm's
  share on top of it, −0.006, is a tie that costs +2.4% s/step and 4.0 GiB. Maciej keeps
  it for stability (2026-11-20; [the digest's conclusions](standard_model_improvements.md#conclusions-what-to-apply-to-ame-and-dense)).

### 6. Dense h16 at n = 8 is unstable: n = 9 is the optimum

| Run | Against | Gap | `final_ce` Δ |
|---|---|---:|---:|
| `ezzyvpn8`: n = 8 | `eh2sjsh7`: n = 9 | +0.7544 | +0.6904 |

- **It never recovered:**
  - 102 spike steps from step 271 on;
  - a grad-norm maximum of 27.5, at step 1975;
  - the gap was +1.60 in the first quarter and +0.75 at the end.
- **So dense h16's best n is 9, bracketed:** n = 8 is unstable, and n = 10 was worse
  ([stage_1b_report.md](stage_1b_report.md)). The "at the edge" caveat is settled.
