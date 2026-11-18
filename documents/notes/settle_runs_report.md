# Settle runs: the open items of the standard-model digest

**Status:** running, from 2026-11-18. Six jobs are on LEM, submitted at 14:21 CET. The
plan below was written before any result. The log and the results are appended as the
runs go.

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
| 5 | Do QK-norm and the KV shift help dense h12, and what do they cost per step? | `ai_dense_h12_qkn_shift_lr8` | `attention: {qk_norm: true, shift: true}` | `x0xlqroj` (2.9824) |
| 6 | Dense h16's best learning rate, at the grid's edge | `s1b_d_h16_lr8` | n = 8 | `eh2sjsh7` (n = 9: 2.7441) |

All runs are h12 at 4B tokens (3,815 steps) or h16 at 8B (7,630), batch 512 × 2,048,
LR 2⁻⁸, seed 123, one LEM H100 each, no FSDP2 (so AME's readouts are logged).

### Configs and code

| # | Branch (worktree) | Commit | Config |
|---|---|---|---|
| 1, 5 | `settle_runs_improve` (`~/code/nano-settle-improve`), from `ame_improve_runs` `43a030e0` | `08b75753` | `configs/ame_improve/ame_h12_ps_bank4_std1e4.yaml`, `configs/ame_improve/dense_h12_qknorm_shift.yaml` |
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
- **5** against `x0xlqroj`: `qk_norm: true`, `shift: true` and `bank_lr_ratio` 4.0.
  - The ratio is a no-op without banks (`parameter_groups`, `src/core/model.py`).
  - The schema keys listed under 1 are all off.
  - The dense τ (QK-norm's learned temperature) sits in a no-decay group, by design.
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
  - Run 5 is the exception: QK-norm and the shift change the init forward, so its
    step 0 is a new number.

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
   - each replicate against its unmasked LEM baseline is the mask's gain measured on
     one cluster;
   - each replicate against its masked original is the same-seed spread across
     clusters. Entropy runs of `doi36p8k`'s config paired at +0.0157, +0.0002 and
     +0.0053 against it (`reference_final_ce_precision`);
   - the ≈0.02 is confirmed if both replicates come out negative again, at about −0.02.
5. **Run 5 against `x0xlqroj`:** the paired gap, and s/step against 3.272.
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

| # | Job | Cemetery clone | W&B | Node | Started | Ended | State | `final_ce` | Gap, last 500 |
|---|---|---|---|---|---|---|---|---:|---:|
| 1 | 6353710 | `ai_h12_ps_bank4_std1e4_lr8_2026-11-18_14-14-15` | | | | | PENDING | | |
| 5 | 6353711 | `ai_dense_h12_qkn_shift_lr8_2026-11-18_14-14-43` | | | | | PENDING | | |
| 2 | 6353712 | `bank_gates_h16_none_std001_2026-11-18_14-15-12` | | | | | PENDING | | |
| 6 | 6353713 | `s1b_d_h16_lr8_2026-11-18_14-15-34` | | | | | PENDING | | |
| 4 AME | 6353714 | `dm_ame_h12_ps_lr8_rep_2026-11-18_14-16-02` | | | | | PENDING | | |
| 4 dense | 6353715 | `dm_dense_h12_lr8_rep_2026-11-18_14-16-22` | | | | | PENDING | | |

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

## Results

Pending.
