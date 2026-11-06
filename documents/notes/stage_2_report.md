# Stage 2 — positional encoding and the shift: report

**Status: wave 1 (S2-P, S2-S, S2-PS, S2-N at n = 9) submitted 2026-11-06 09:33–09:35 on LEM,
by Maciej's call, before Stage 1 settled.**

The plan is [ame_experiments.md, "Stage 2 — positional encoding and the shift"](ame_experiments.md#stage-2--positional-encoding-and-the-shift).
This file is its record: every job, result and decision, each with its reason. Times are
CET. Stage 1's record, which supplies the reference cell S1-A0 and the dense baseline
S1-D, is [stage_1_report.md](stage_1_report.md).

## Setup

| | |
|---|---|
| Question | Which AME variant is best at h12 and 4B tokens: with positional encoding, with the KV shift, or with both? And how much does each one add? |
| Design | The 2 × 2 of {no positional encoding, ALiBi} × {no shift, KV shift}. S1-A0 (neither) comes from Stage 1; S2-P, S2-S and S2-PS are trained here; S2-N repeats S1-A0 with seed 124 as the noise floor |
| Positional encoding | The implemented ALiBi ladder: one fixed slope per head, horizons 16 to 8,192 tokens over 12 heads. No Stage 0 replacement was proposed before the start |
| Code and configs | Commit **`f9680709`** on `ame_main`, for every Stage 2 run. Its code is identical to Stage 1's `8a240816`: between them, only docs, the Stage 1b–2b configs and the four `learning_rate: 9` lines changed. Jobs are submitted from a clean worktree of their own, `~/code/nano-stage2` (branch `stage2_lem`). Stage 1's worktree stays at `8a240816` |
| Configs | [ame_h12_pe.yaml](../../configs/stage_2/ame_h12_pe.yaml) (S2-P), [ame_h12_shift.yaml](../../configs/stage_2/ame_h12_shift.yaml) (S2-S), [ame_h12_pe_shift.yaml](../../configs/stage_2/ame_h12_pe_shift.yaml) (S2-PS), [ame_h12_simplest_seed2.yaml](../../configs/stage_2/ame_h12_simplest_seed2.yaml) (S2-N) |
| Cluster, recipe | As in Stage 1: LEM, 1 H100 SXM5 94 GB per run, no FSDP2, autocast bf16, micro-batch 32 × 16, 3,815 steps = 4.00B tokens, seed 123 (S2-N: 124), r16-11 excluded, 7 h limit passed at submit ([stage_1_report.md](stage_1_report.md), decisions 3–4) |

## Decisions

1. **n\* = 9, by Maciej's call (2026-11-06, about 09:30), before Stage 1 settled.** His
   words: "Submit h12 experiments from stage 2, assume that LR 9 is the optimal one for
   both dense and ame … we can submit experiments from stage 2 and resubmit if we get
   different conclusions from stage 1." The four files set `learning_rate: 9`, committed
   in `f9680709`.

   At that moment Stage 1 stood like this
   ([stage_1_report.md](stage_1_report.md)):
   - n = 8 had beaten n = 9 in both arms: 2.9824 against 3.0244 for dense, 4.3711
     against 4.4316 for AME;
   - n = 7 (both arms) and n = 6 (AME) were still training.

   So Stage 1's final n\* may well differ from 9. If it does, the plan's rule decides the
   re-run: Stage 2's three arms and S2-N go again at Stage 1's n\*. Until then, the
   learning-rate check of Stage 2 itself (the best arm at n\* ± 1 = 8 and 10) still runs
   as the plan says. It measures directly whether n = 8 beats n = 9 for the winning arm.
2. **`run_exp.py` cannot finish a submit whose run name contains a dot, so `sbatch` was
   run by hand.** The W&B names read `s2_<arm>_h12_lr9.0`, as the plan predicts, because
   the schema types the learning rate as a float. `run_exp.py` uses the same string as a
   tmux session name. tmux turns the `.` into `_` when it creates the session, then reads
   the `.` in `tmux send -t <name>.0` as a pane separator, and fails with
   `can't find pane: 0_2026-11-06_09-33-22.0` before `sbatch` runs. Each clone was already
   complete, with the secrets substituted into `exp.job`. So `cd <clone> && sbatch exp.job`
   on LEM is exactly the step that was skipped. No code was changed. The fix is for
   Maciej ("For Maciej" below).

## Wave 1 — the three arms and the noise floor, at n = 9

| Arm | Positional encoding | KV shift | Job | Cemetery commit | Cemetery clone (under `.../nano/plgmstefaniak/llmrandom_cemetery/`) |
|---|---|---|---|---|---|
| S2-P | ALiBi | off | 6253416 | `95f210ea` | `s2_p_h12_lr9.0_2026-11-06_09-33-22` |
| S2-S | none (12 zero slopes) | on | 6253417 | `766c8311` | `s2_s_h12_lr9.0_2026-11-06_09-33-51` |
| S2-PS | ALiBi | on | 6253467 | `43f6bf5d` | `s2_ps_h12_lr9.0_2026-11-06_09-34-55` |
| S2-N | none | off, seed 124 | 6253468 | `5fe267bb` | `s2_n_h12_lr9.0_seed124_2026-11-06_09-35-28` |

| Arm | n | Job | W&B run | Node | Parameters, as logged |
|---|---:|---|---|---|---:|
| S2-P | 9 | 6253416_0 | `0rbp1bju` | r12-9 | 1,103,598,480 |
| S2-S | 9 | 6253417_0 | `r6lxbmtd` | r12-9 | 1,103,599,056 |
| S2-PS | 9 | 6253467_0 | — (starting at 09:37) | r12-12 | |
| S2-N | 9 | 6253468_0 | — (starting at 09:37) | r12-12 | |

Every cemetery commit has `f9680709` as its parent and adds only `exp.job` and
`generated_configs/config_0.yaml`. The plan's run order (S2-P, then S2-S, then S2-PS) was
kept for submission; the four runs do not depend on each other.

Results: pending; about 4.3 h each once running.

## For Maciej

Items that need your decision. None blocks a Stage 1–2 run.

1. **n\* = 9 against Stage 1's data.** Stage 2 runs at your n = 9. Stage 1 had n = 8 ahead
   of n = 9 in both arms when it started (decision 1), and n = 7 and 6 were still running.
   If Stage 1 ends with n\* ≠ 9, the plan re-runs Stage 2's wave at the new n\*. That costs
   about 17 GPU-h and one 4.5 h wave.
2. **`run_exp.py` and run names with a dot** (decision 2). A one-line fix would be to
   sanitize the tmux session name, e.g.
   `tmux_name = experiment_branch_name.replace(".", "_").replace(":", "_")`, and use it
   in `tmux new`, `tmux send` and `wait_for_job_id`. Until then, every submit of a config
   with a fixed float learning rate needs the manual `sbatch`.
3. **No MFU in W&B for LEM runs.** LEM's GPUs report as `NVIDIA H100` (H100 SXM5 94 GB),
   which `src/core/mfu.py`'s peak table does not match, so `train/mfu` is never logged
   there. Adding `("NVIDIA H100", 989e12)` after the more specific H100 entries would fix
   it. The stage reports compute MFU offline meanwhile.
4. **Node r16-11 on LEM has a bad GPU.** Two jobs died there within seconds with
   `CUDA-capable device(s) is/are busy or unavailable`, so every Stage 1–2 job excludes it.
   It may be worth reporting to WCSS.
