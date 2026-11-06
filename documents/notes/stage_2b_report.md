# Stage 2b — positional encoding and the shift at h16: report

**Status: wave 1 (S2b-P, S2b-S, S2b-PS, S2b-N at n = 9) submitted 2026-11-06 12:29 on LEM,
by Maciej's call, with Stage 1b concluded at n\* = 9.**

The plan is [ame_experiments.md, "Stage 2b — positional encoding and the shift at h16"](ame_experiments.md#stage-2b--positional-encoding-and-the-shift-at-h16).
This file is its record: every job, result and decision, each with its reason. Times are
CET. Stage 1b's record, which supplies the reference cell S1b-A0 and the dense baseline
S1b-D, is [stage_1b_report.md](stage_1b_report.md). Stage 2's record, the same question
at h12, is [stage_2_report.md](stage_2_report.md).

## Setup

| | |
|---|---|
| Question | Stage 2's question at h16 and 8B tokens: which AME variant is best, and how much does each component add? And do the answers match Stage 2's? |
| Design | The 2 × 2 of {no positional encoding, ALiBi} × {no shift, KV shift}. S1b-A0 (neither) comes from Stage 1b; S2b-P, S2b-S and S2b-PS are trained here; S2b-N repeats S1b-A0 with seed 124 as the noise floor |
| Positional encoding | The implemented ALiBi ladder, as in Stage 2: one fixed slope per head, horizons 16 to 8,192 tokens, here over 16 heads |
| Code and configs | Commit **`89975bbf`** on `ame_main`, for every Stage 2b run, submitted from its own clean worktree, `~/code/nano-stage2b` (branch `stage2b_lem`). Since Stage 1b's `31cbc268`, the only code change is the gqa NoPE option (`c4cb1ef6`, [stage_2_report.md](stage_2_report.md) decision 3). It does not touch AME: `src/core/ame.py`, `src/core/memory_parallel.py` and `_build_ame` are unchanged. So Stage 2b stays comparable with S1b-A0 without a re-run |
| Configs | [ame_h16_pe.yaml](../../configs/stage_2b/ame_h16_pe.yaml) (S2b-P), [ame_h16_shift.yaml](../../configs/stage_2b/ame_h16_shift.yaml) (S2b-S), [ame_h16_pe_shift.yaml](../../configs/stage_2b/ame_h16_pe_shift.yaml) (S2b-PS), [ame_h16_simplest_seed2.yaml](../../configs/stage_2b/ame_h16_simplest_seed2.yaml) (S2b-N) |
| Cluster, recipe | As in Stage 1b: LEM, 1 H100 SXM5 94 GB per run, no FSDP2, autocast bf16, micro-batch 16 × 32, 7,630 steps = 8.00B tokens, seed 123 (S2b-N: 124), 24 h limit, r16-11 excluded |

## Decisions

1. **n\* = 9, by Maciej's call (2026-11-06, about 12:00), concluding Stage 1b.** His
   words: "assume that optimal lr for h16 baseline and AME is 9 - that concludes stage 1b
   - dont run more experiemnts for stage 1b, submit now experiments from stage 2b for ame
   for lr = 9". The four files set `learning_rate: 9`, committed in `89975bbf`.

   Stage 1b's grid was then at step ~800 of 7,630 ([stage_1b_report.md](stage_1b_report.md),
   decision 3). n = 9 led dense; AME led at n = 10 and 11, where n = 9 had spiked early.
   It matches the h12 choice ([stage_1_report.md](stage_1_report.md), decision 9).

   The plan's learning-rate check still follows wave 1: the best arm at n\* ± 1 = 8 and 10.
2. **`sbatch` was run by hand**, as in Stage 2 ([stage_2_report.md](stage_2_report.md),
   decision 2). The run names read `s2b_<arm>_h16_lr9.0`, and `run_exp.py`'s tmux step
   fails on the dot. Each clone was complete, with the secrets in `exp.job`, so
   `cd <clone> && sbatch exp.job` on LEM is the step that was skipped.

## Wave 1 — the three arms and the noise floor, at n = 9

| Arm | Positional encoding | KV shift | Job | Cemetery commit | Cemetery clone (under `.../nano/plgmstefaniak/llmrandom_cemetery/`) |
|---|---|---|---|---|---|
| S2b-P | ALiBi | off | 6255245 | `9c9e37c3` | `s2b_p_h16_lr9.0_2026-11-06_12-27-59` |
| S2b-S | none (16 zero slopes) | on | 6255246 | `10b6a080` | `s2b_s_h16_lr9.0_2026-11-06_12-28-20` |
| S2b-PS | ALiBi | on | 6255247 | `aa0aa401` | `s2b_ps_h16_lr9.0_2026-11-06_12-28-40` |
| S2b-N | none | off, seed 124 | 6255248 | `d8a407e3` | `s2b_n_h16_lr9.0_seed124_2026-11-06_12-29-01` |

Every cemetery commit has `89975bbf` as its parent and adds only `exp.job` and
`generated_configs/config_0.yaml`. The plan's order (S2b-P, S2b-S, S2b-PS, then S2b-N)
was kept for submission; the runs do not depend on each other.

Results: pending; about 14.5–15.1 h each once running (Stage 1b smoke runs).

## For Maciej

Items that need your decision. None blocks a run.

1. **n\* = 9 was set while Stage 1b's grid was 10% in** (decision 1). If the finished
   grid (about 23:50) puts S1b-A0's best elsewhere, the plan's rule would re-run wave 1
   there, at about 62 GPU-h and 16 h. At h12, AME kept improving down to n = 6
   ([stage_1_report.md](stage_1_report.md), decision 10), which hints that h16 AME may
   also prefer a higher learning rate than n = 9.
2. **`run_exp.py` and run names with a dot**: see [stage_2_report.md](stage_2_report.md),
   "For Maciej" 2.
