# Stage 1b — Stage 1 at h16 and 8B tokens: report

**Status: the smoke runs passed (09:07); the grid is submitted once LEM's SSH answers again.**

The plan is [ame_experiments.md, "Stage 1b — Stage 1 at h16 and 8B tokens"](ame_experiments.md#stage-1b--stage-1-at-h16-and-8b-tokens).
This file is its record: every job, result and decision, each with its reason. Times are
CET. Stage 1b runs alongside Stages 1 and 2, whose record is
[stage_1_report.md](stage_1_report.md). Stage 1b shares their cluster and recipe, but not
their commit or worktree.

## Setup

| | |
|---|---|
| Question | With a tuned learning rate for each, how far is the simplest AME (S1b-A0) from a dense baseline (S1b-D) at h16 and 8B tokens? And does the gap shrink from Stage 1's? |
| Code and configs | Commit **`31cbc268`** on `ame_main`, for every Stage 1b run. Jobs are submitted from the clean worktree `~/code/nano-stage1b2b` (branch `stage1b2b_lem`, held at that commit), separate from Stage 1–2's `~/code/nano-stage12`. Between Stage 1's `8a240816` and `31cbc268`, nothing changed in `src/`, `main.py`, the submit path or the shared configs: only docs and these six configs. So Stages 1 and 1b run the same code |
| Configs | [configs/stage_1b/dense_h16_lr_grid.yaml](../../configs/stage_1b/dense_h16_lr_grid.yaml) (S1b-D), [configs/stage_1b/ame_h16_simplest_lr_grid.yaml](../../configs/stage_1b/ame_h16_simplest_lr_grid.yaml) (S1b-A0) |
| Cluster | LEM, as in Stage 1: partition `plgrid-lem-gpu-h100`, 1 H100 SXM5 94 GB per run, node r16-11 excluded ([stage_1_report.md](stage_1_report.md), decision 3). The partition allows 3 days, so the configs' 24 h limit is accepted |
| W&B | `ideas_cv/nano-ame`, tag `stage_1b` |
| Recipe | Stage 1's, with the plan's changes: h_balanced h16 (16 blocks, 16 heads, 8 KV in dense, d_model 1,024, d_head 64, d_ff 4,096); 7,630 steps × 1,048,576 tokens = 8.00B tokens; warmup 76 steps; micro-batch 16 × 32 accumulation in both arms; grid n ∈ {9, 10, 11}; 24 h limit |

**Pre-flight, 2026-11-06, before commit `31cbc268`.** All six Stage 1b–2b configs compose
on LEM, and every grid point passes `run_exp.py`'s schema check. The repo's config tests
pass (41 of 41). Every model builds at the plan's sizes, counted on the meta device:

| | S1b-D, dense | S1b-A0, S2b-P, S2b-N | S2b-S, S2b-PS |
|---|---:|---:|---:|
| Total parameters | 354,714,624 | 1,986,331,904 | 1,986,332,928 |
| Q/K banks | — | 1,648,361,472 | 1,648,361,472 |
| Training FLOPs per token, seq 2048 | 2.222 G | 2.121 G | 2.121 G |

The shift arms carry 1,024 more parameters: α and β, 2 + 2 per head over 16 heads and 16
layers. S2b-N trains on seed 124 and is scored on the eval set of seed 123, as in Stage 2.

What the comparison matches and what it does not (from the plan):

| Matched | Not matched |
|---|---|
| tokens | total parameters: AME has 5.6× |
| data order | FLOPs per token: AME does 0.95× |
| recipe | positional signal: RoPE vs none |

**Weight decay differs between the arms.** AME's Q/K banks are exempt from weight decay,
while the dense W_Q and W_K are not
([archive/ame_review_and_fixes.md](archive/ame_review_and_fixes.md) 2.6). Every AME-vs-dense
number in this file carries that difference.

## Smoke runs — LEM's h16 baseline

The plan's "Before the grid": `trainer.n_steps=60 trainer.eval_interval=20`, with n = 10
for the AME pair. All five smoke runs were submitted at 08:53 from `~/code/nano-stage1b2b`
at `31cbc268`, and all finished by 09:07. Every check passed.

| Job | Config | n | W&B run | Node | Result |
|---|---|---:|---|---|---|
| 6253231_0 | stage_1b/dense_h16_lr_grid | 9 | `x8fdvgvx` | r12-4 | completed |
| 6253231_1 | stage_1b/dense_h16_lr_grid | 10 | `hv43mznh` | r12-8 | completed |
| 6253231_2 | stage_1b/dense_h16_lr_grid | 11 | `fq0rxyo3` | r12-8 | completed |
| 6253235 | stage_2b/ame_h16_simplest_seed2 (the S1b-A0 code path, seed 124) | 10 | `p572pg6e` | r12-9 | completed |
| 6253237 | stage_2b/ame_h16_pe_shift (ALiBi and the shift) | 10 | `82b324oh` | — | completed |

| Check | Dense h16 | AME h16, S1b-A0 path | AME h16, S2b-PS |
|---|---|---|---|
| Micro-batch 16 fits without FSDP2 | yes, peak 37.50 GiB | yes, 67.92 GiB | yes, 71.92 GiB of 93.6 |
| FLOPs per token, as logged | 2.222 G | 2.121 G | 2.121 G |
| s/step on LEM, median over steps 15–59 | 5.920 / 5.950 / 5.971 (n = 9/10/11) | 6.480 | 6.765 |
| s/step in the plan's estimate | 6.2 | 6.6 | 6.9 |
| AME readouts in W&B | — | yes: α₂ = β₂ = 0, horizon = Infinity, τ = 8, attention entropy 0.926 (layer 0) | yes: α₂ = 0.557, β₂ = 0.501, horizons 16–8,196 (layer 0) |
| `final_ce` logged | yes | yes | yes |
| Checkpoint written | yes | yes, bank state included | yes, bank state included |

The checkpoint lines come from the job logs that W&B captured (`output.log`), because
LEM's login node stopped answering SSH at 08:56 (decision 2). The smoke checkpoints are
deleted once it answers again.

**Whole-run estimate on LEM:** 7,630 steps at the smoke s/step, plus about 76 evals and
startup:
- S1b-D: 12.6 h of steps, about 13.4 h in all;
- S1b-A0: 13.7 h, about 14.5 h;
- S2b-PS: 14.3 h, about 15.1 h.

All fit the 24 h limit with room to spare. MFU, offline at 989 TFLOP/s as in Stage 1, is
40.0% for dense, 33.9% for S1b-A0 and 32.5% for S2b-PS.

## Decisions

Stage 1's decisions carry over where they apply
([stage_1_report.md](stage_1_report.md), "Decisions"):
- the dense smoke runs its points at once;
- smoke runs are named `smoke_*` and tagged `smoke`, with a 1.5 h limit;
- node r16-11 is excluded.

Points specific to Stage 1b:

1. **The configs' 24 h limit stays.** The smoke runs put a whole run at 13.4–15.1 h, which
   leaves more than 8 h of margin.
2. **LEM's SSH went unreachable at 08:56.** The TCP connection to `ui.wcss.pl:22` timed
   out while the internet and W&B worked, and the jobs kept running and logging. The
   likely cause is a firewall rate limit on new connections from this machine. After a
   stuck ControlMaster had to be bypassed at 06:52, my watchers were opening a fresh SSH
   connection every minute. The watchers now share one persistent connection with
   keep-alives, poll every 5 minutes, and wait for the port with a bare TCP probe every 3
   minutes. Everything that needed LEM was moved to W&B meanwhile: the smoke results,
   and the checkpoint lines from the captured logs.
