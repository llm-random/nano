# Stage 2b — positional encoding and the shift at h16: report

**Status: wave 2 finished (2026-11-07 18:43). The learning-rate check moved S2b-PS's best
n to 8: `final_ce` 2.8594, against 2.8965 at n = 9 and 2.9688 at n = 10. By the plan's
rule, S2b-P and S2b-S re-run at n = 8 before the winner is named. That is wave 3, running
since 08:17 (jobs 6261187, 6261188; decision 4), done about 23:00. At n = 9 the order of
the cells matched h12, with S2b-PS best. The noise-floor run S2b-N was cancelled at
Maciej's request.**

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
3. **S2b-N cancelled by Maciej (2026-11-06, about 12:40): "we dont need that S2b-N,
   cancel that one".** Job 6255248 was cancelled while still pending and never ran; its
   clone stays on disk, unused.
   - **Consequence.** Stage 2b has no noise floor of its own: no second seed of S1b-A0
     at h16, so a tie cannot be measured at this scale.
   - **How the analysis handles it.** It takes Stage 2's h12 seed gap, S1-A0 against S2-N
     at n = 9, as a stated proxy. It reports every Stage 2b difference with that proxy
     beside it, instead of calling ties at h16.
4. **S2b-P and S2b-S re-run at n = 8 before the learning-rate check finished
   (2026-11-07 08:16).** The plan re-runs the other two arms at the new n if the check
   moves S2b-PS's best n. I started them when the check was 30% done, not at its end:
   - **n = 8 leads n = 9 at matched steps, and the lead grows.** The two runs see the
     same data in the same order (seed 123), so the comparison is paired. n = 8 overtook
     n = 9 at about step 1,500 (20% of the run) and leads by 0.006 at steps 2,000–2,199
     (table in wave 2). n = 10 trails by 0.07.
   - **h12 took the same path and ended at n = 8.** There n = 8 overtook n = 9 at about
     step 950 of 3,815 (25%); its lead grew steadily, to 0.061 in `final_ce`. At h16 the
     crossover came earlier, but the lead grows more slowly so far: 0.006 against h12's
     0.011 at 26–29% of the run.
   - **Starting now saves about 10.5 h**: wave 3 finishes around 23:45 instead of about
     10:00 on 2026-11-08.
   - **Cost if n = 9 wins after all:** about 30 GPU-h, and the two runs still show how
     S2b-P and S2b-S respond to the learning rate. The LEM grant had 72,150 of its
     100,000 GPU-h left (`service-balance`, 08:15).

   The runs use the same commit, `89975bbf`, with `trainer.learning_rate=8`; `sbatch` was
   run by hand (decision 2).

## Wave 1 — the three arms and the noise floor, at n = 9

| Arm | Positional encoding | KV shift | Job | Cemetery commit | Cemetery clone (under `.../nano/plgmstefaniak/llmrandom_cemetery/`) |
|---|---|---|---|---|---|
| S2b-P | ALiBi | off | 6255245 | `9c9e37c3` | `s2b_p_h16_lr9.0_2026-11-06_12-27-59` |
| S2b-S | none (16 zero slopes) | on | 6255246 | `10b6a080` | `s2b_s_h16_lr9.0_2026-11-06_12-28-20` |
| S2b-PS | ALiBi | on | 6255247 | `aa0aa401` | `s2b_ps_h16_lr9.0_2026-11-06_12-28-40` |
| ~~S2b-N~~ | none | off, seed 124 | 6255248 — **cancelled before it ran** (decision 3) | `d8a407e3` | `s2b_n_h16_lr9.0_seed124_2026-11-06_12-29-01` |

Every cemetery commit has `89975bbf` as its parent and adds only `exp.job` and
`generated_configs/config_0.yaml`. The plan's order (S2b-P, S2b-S, S2b-PS, then S2b-N)
was kept for submission; the runs do not depend on each other.

| Arm | n | Job | W&B run | Node |
|---|---:|---|---|---|
| S2b-P | 9 | 6255245_0 | `eedqxi1j` | r12-16 |
| S2b-S | 9 | 6255246_0 | `c2tjvh5w` | r16-12 |
| S2b-PS | 9 | 6255247_0 | `rv4xqzap` | r16-12 |

### Results — wave 1 finished 2026-11-07 02:59–03:34

Every row has n = 9 and 7,630 steps × 1,048,576 tokens = 8.00B tokens. The AME arms have
1,986.33M parameters (234.9M backbone, 1,648.36M banks; the shift arms add 1,024) and
2.121 GFLOP/token. Dense has 354.71M parameters (251.7M backbone, no banks) and 2.222
GFLOP/token. Each run wrote its final checkpoint, 23 GB for AME.

| Cell | Positional encoding | Shift | `final_ce` | Loss spikes (train) | s/step | MFU | W&B run | Commit | Config |
|---|---|---|---:|---|---:|---:|---|---|---|
| S1b-A0 (Stage 1b) | none | — | 3.9746 | steps 380 → 6.66; 1,600–1,622 → 7.00 | 6.540 | 34.4% | `n15qni7a` | `31cbc268` | stage_1b/ame_h16_simplest_lr_grid |
| S2b-S | none | ✓ | 3.4658 | steps 1,937–1,962 → 9.56; 2,971–2,995 → 8.38; 4,050–4,066 → 4.84 | 6.782 | 33.2% | `c2tjvh5w` | `89975bbf` | stage_2b/ame_h16_shift |
| S2b-P | ALiBi | — | 2.9834 | none | 6.511 | 34.5% | `eedqxi1j` | `89975bbf` | stage_2b/ame_h16_pe |
| **S2b-PS** | ALiBi | ✓ | **2.8965** | none | 6.773 | 33.2% | `rv4xqzap` | `89975bbf` | stage_2b/ame_h16_pe_shift |
| S1b-D (Stage 1b) | RoPE | — | 2.7441 | none | 5.964 | 39.5% | `eh2sjsh7` | `31cbc268` | stage_1b/dense_h16_lr_grid |

![Stage 2b eval loss](stage_figures/stage_2b_eval_loss.png)

*Eval loss on 10 batches by step, from step 300, with the y-axis cut at 6. The end labels
are the last such eval, not `final_ce`. Made with
[documents/scripts/stage_results.py](../scripts/stage_results.py) `curves`.*

### Analysis at n = 9 — the plan's list

**The four cells:**

|  | no shift | KV shift |
|---|---:|---:|
| **no positional encoding** | S1b-A0: 3.9746 | S2b-S: 3.4658 |
| **ALiBi** | S2b-P: 2.9834 | **S2b-PS: 2.8965** |

**Main effects and interaction**, next to Stage 2's at the same n:

| | h16, 8B (here) | h12, 4B (Stage 2) |
|---|---:|---:|
| ALiBi without the shift (P − A0) | −0.991 | −1.134 |
| ALiBi with the shift (PS − S) | −0.569 | −0.752 |
| **ALiBi, averaged** | **−0.780** | **−0.943** |
| Shift without ALiBi (S − A0) | −0.509 | −0.493 |
| Shift with ALiBi (PS − P) | −0.087 | −0.111 |
| **Shift, averaged** | **−0.298** | **−0.302** |
| **Interaction** | **+0.422** | **+0.382** |

**No noise floor of its own.** S2b-N was cancelled (decision 3). The stated proxy is
Stage 2's h12 seed gap, 0.232, which one loss spike inflates. Against it:
- **ALiBi** is far above the floor in both rows.
- **The shift without ALiBi** (−0.51) is above it.
- **The shift on top of ALiBi** (−0.087) is below it.

S2b-PS against S2b-P is therefore a tie by the proxy rule. It has the same sign as at h12
(−0.111 at n = 9, −0.123 at n = 8), and neither ALiBi run spiked at either scale.

**Every cell against S1b-D**, dense h16 at n = 9 (2.7441):

| Cell | Gap at h16, 8B | Gap at h12, 4B (Stage 2, n = 9) | Change |
|---|---:|---:|---:|
| S1b-A0 | +1.231 | +1.407 | −0.177 |
| S2b-S | +0.722 | +0.914 | −0.192 |
| S2b-P | +0.239 | +0.274 | −0.034 |
| **S2b-PS** | **+0.152** | +0.162 | −0.010 |

The comparison matches tokens, data order and recipe. It does not match:
- total parameters: AME has 5.6×, mostly banks;
- FLOPs per token: AME is 0.95× dense;
- positional signal;
- weight decay: the AME banks are exempt, while the dense W_Q and W_K get 0.1.

**Against Stage 2, the plan's two questions:**
- **Do the four cells rank in the same order at both scales? Yes:** PS < P < S < A0 at h12
  (n = 8 and 9) and at h16 (n = 9).
- **How does each cell's gap to dense change from h12/4B to h16/8B? Every gap shrinks.**
  The weak cells shrink most: −0.18 for A0, −0.19 for S. The ALiBi cells shrink least:
  −0.03 for P, −0.01 for PS. So scale helps the position-free variants catch up. The best
  variant's distance to dense barely moves, about 0.15–0.16 nats.
- **What this measures.** Model size and tokens changed together, at almost the same
  tokens per dense parameter (21.8 and 22.6). So this is a scaling result, not the effect
  of either change alone.

**Readouts at the end of training**, averaged over the 16 layers:

| Cell | τ, mean | Shortest horizon | Attention entropy, normalized | Top-8 mass | α₂, mean \|·\| | β₂, mean \|·\| |
|---|---:|---:|---:|---:|---:|---:|
| S1b-A0 | 5.88 | ∞ | 0.930 | 0.113 | pinned 0 | pinned 0 |
| S2b-S | 8.20 | ∞ | 0.784 | 0.269 | 0.297 | 0.740 |
| S2b-P | 7.23 | 6.0 (4.1 in layers 3–4) | 0.527 | 0.574 | pinned 0 | pinned 0 |
| S2b-PS | 8.33 | 15.0 | 0.536 | 0.555 | 0.454 | 0.669 |

The h12 pattern repeats:
- **ALiBi sharpens attention** to about half the mass on 8 keys.
- **The shift is kept,** with β₂ > α₂.
- **Without the shift, τ falls and shortens ALiBi's horizons** (S2b-P: down to about 4
  tokens). With the shift, τ stays at about 8.

**The best AME variant at h16 and 8B tokens, provisionally: S2b-PS, ALiBi with the KV
shift**, at `final_ce` 2.8965, 0.152 behind dense. The plan names it only after the
learning-rate check below.

## Wave 2 — the learning-rate check on S2b-PS, at n = 8 and 10

The plan: the best arm trains at n\* ± 1. If its best n moves, the other two arms re-run at
the new n before a winner is named. The runs were submitted at 03:36 from
`~/code/nano-stage2b` at `89975bbf`, with `trainer.learning_rate=8` and `=10`; `sbatch`
was run by hand (decision 2).

| Arm | n | Job | Cemetery commit | Cemetery clone | W&B run | Node |
|---|---:|---|---|---|---|---|
| S2b-PS | 8 | 6260570 | `e51f51d4` | `s2b_ps_h16_lr8.0_2026-11-07_03-36-02` | `hjw7znbz` | r12-4 |
| S2b-PS | 10 | 6260571 | `c1df24a4` | `s2b_ps_h16_lr10.0_2026-11-07_03-36-23` | `kxt7oe48` | r12-4 |

Both started at about 03:38 and passed the startup check: 1,986,332,928 parameters and no
error. They finish around 18:45.

**At 30% (08:14, step ~2,300 of 7,630)**, train loss averaged over 200-step windows, with
wave 1's S2b-PS at n = 9 at the same steps. The runs share seed and data order, so each
row compares the same batches. `eval/loss` is logged in bf16, too coarse for gaps this
small: it reads 3.2031 for both n = 8 and n = 9 at step 2,200.

| Steps | Share of the run | n = 8 | n = 9 (wave 1) | n = 10 | n = 8 − n = 9 | h12, same share: n = 8 − n = 9 |
|---|---|---:|---:|---:|---:|---:|
| 1,000–1,199 | 13–16% | 3.5739 | 3.5566 | 3.6230 | +0.017 | +0.088 |
| 1,200–1,399 | 16–18% | 3.4941 | 3.4848 | 3.5557 | +0.009 | +0.059 |
| 1,400–1,599 | 18–21% | 3.4308 | 3.4298 | 3.4996 | +0.001 | +0.031 |
| 1,600–1,799 | 21–24% | 3.3811 | 3.3836 | 3.4546 | −0.002 | +0.011 |
| 1,800–1,999 | 24–26% | 3.3426 | 3.3453 | 3.4168 | −0.003 | +0.000 |
| 2,000–2,199 | 26–29% | 3.3074 | 3.3134 | 3.3831 | **−0.006** | −0.011 |

The h12 column is Stage 2's same check (`doi36p8k` against `ublx3ion`), in 100-step
windows at the same share of its 3,815 steps (7,630 = 2 × 3,815). Its gap kept growing:
−0.061 over the last 115 steps and in `final_ce`. At h16, n = 8 crossed earlier (20% of
the run, against 25%) but its lead grows more slowly so far. The direction is the same,
so wave 3 started now (decision 4).

**At 50% (11:14, step ~3,830)** n = 8's lead had grown, and n = 10 stayed behind. Train
loss over 200-step windows:

| Steps | n = 8 | n = 9 (wave 1) | n = 10 | n = 8 − n = 9 |
|---|---:|---:|---:|---:|
| 2,400–2,599 | 3.2512 | 3.2589 | 3.3274 | −0.008 |
| 3,000–3,199 | 3.1880 | 3.1969 | 3.2641 | −0.009 |
| 3,600–3,799 | 3.1319 | 3.1424 | 3.2091 | **−0.011** |

`eval/loss` now separates them by one bf16 step: 3.0312 (n = 8) against 3.0469 (n = 9)
at step 3,800. Neither run has spiked: no train loss above 5.41 after step 200.

**At 70% (14:06, step ~5,300)** the lead was still growing, as at h12 in the cosine decay:

| Steps | n = 8 | n = 9 (wave 1) | n = 10 | n = 8 − n = 9 |
|---|---:|---:|---:|---:|
| 4,200–4,399 | 3.0856 | 3.0994 | 3.1653 | −0.014 |
| 4,600–4,799 | 3.0598 | 3.0758 | 3.1430 | −0.016 |
| 5,000–5,199 | 3.0338 | 3.0532 | 3.1213 | −0.019 |
| 6,000–6,199 (16:54) | 2.9768 | 3.0044 | 3.0773 | −0.028 |
| 6,400–6,599 (16:54) | 2.9603 | 2.9910 | 3.0659 | **−0.031** |

Still no spike in either run at 16:54 (step ~6,750, 88%). `eval/loss` at step 6,700:
2.8750 (n = 8), 2.9062 (n = 9), 2.9844 (n = 10).

### Results — wave 2 finished 18:37 (n = 10) and 18:43 (n = 8)

Both jobs `COMPLETED` after 14 h 59 min and 15 h 06 min, 30.1 GPU-h together. Parameters,
FLOPs per token and code as S2b-PS in wave 1 (1,986.33M; 2.121 GFLOP/token; `89975bbf`,
config stage_2b/ame_h16_pe_shift with `trainer.learning_rate` set on the command line).
Readouts at the end of training, averaged over the 16 layers:

| n | `final_ce` | Last 10-batch eval | Loss spikes (train) | s/step | MFU | τ, mean | Attention entropy, normalized | Top-8 mass | α₂, mean \|·\| | β₂, mean \|·\| | Shortest horizon | Job | W&B run |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 8 | **2.8594** | 2.8594 | none | 6.810 | 33.0% | 8.53 | 0.502 | 0.594 | 0.550 | 0.725 | 14.8 | 6260570 | `hjw7znbz` |
| 9 (wave 1) | 2.8965 | 2.8906 | none | 6.773 | 33.2% | 8.33 | 0.536 | 0.555 | 0.454 | 0.669 | 15.0 | 6255247 | `rv4xqzap` |
| 10 | 2.9688 | 2.9688 | none | 6.757 | 33.3% | 8.04 | 0.595 | 0.481 | 0.350 | 0.690 | 15.2 | 6260571 | `kxt7oe48` |

**The best arm's best n moved to 8**, 0.037 below n = 9; n = 10 is 0.072 above. h12 did
the same: its check moved S2-PS to n = 8, by 0.062 ([stage_2_report.md](stage_2_report.md)).
- **What the plan's rule now asks.** S2b-P and S2b-S at n = 8 before a winner is named.
  That is wave 3, already running (decision 4).
- **No n = 7 run.** n = 8 is the edge of the check, but the plan's check is n\* ± 1 and
  has no edge rule; the edge rule belongs to the Stage 1 and 1b grids. At h12, Maciej's
  extra S2-PS runs at n = 7 and 6 did not beat n = 8. "For Maciej" 3 has the option.
- **The readouts move with the learning rate, as at h12.** The higher it is, the sharper
  the attention (normalized entropy 0.595 → 0.536 → 0.502 from n = 10 to 8; top-8 mass
  0.48 → 0.56 → 0.59) and the larger α₂ (0.35 → 0.45 → 0.55). β₂ and the shortest ALiBi
  horizon barely move.
- **No spikes at any of the three learning rates**, as for every ALiBi run at h12.

## Wave 3 — S2b-P and S2b-S at n = 8

Started early, while wave 2 runs (decision 4). Submitted at 08:16 from
`~/code/nano-stage2b` at `89975bbf`, with `trainer.learning_rate=8`; `sbatch` was run by
hand (decision 2). Each cemetery commit has `89975bbf` as its parent and adds only
`exp.job` and `generated_configs/config_0.yaml`.

| Arm | n | Job | Cemetery commit | Cemetery clone | W&B run | Node |
|---|---:|---|---|---|---|---|
| S2b-P | 8 | 6261187 | `def924ee` | `s2b_p_h16_lr8.0_2026-11-07_08-16-10` | `le2dr3r6` | r10-8 |
| S2b-S | 8 | 6261188 | `52b1d182` | `s2b_s_h16_lr8.0_2026-11-07_08-16-30` | `eeuebwsj` | r10-13 |

Both started at 08:17 and passed the startup check (08:27, step 50):
- **S2b-P:** 1,986,331,904 parameters, 6.52 s/step.
- **S2b-S:** 1,986,332,928 parameters (the shift's 1,024 more), 6.77 s/step.

Both match wave 1. They finish around 22:30 (S2b-P) and 23:00 (S2b-S).

**At 20% (11:14)**, against wave 1's runs at n = 9 on the same steps (train loss, 200-step
windows):

| Steps | S2b-P n = 8 | S2b-P n = 9 | n = 8 − n = 9 | S2b-S n = 8 | S2b-S n = 9 | n = 8 − n = 9 |
|---|---:|---:|---:|---:|---:|---:|
| 600–799 | 3.9935 | 3.9428 | +0.051 | 4.8758 | 5.0561 | −0.180 |
| 800–999 | 3.8241 | 3.7901 | +0.034 | 4.5780 | 4.8613 | −0.283 |
| 1,000–1,199 | 3.7076 | 3.6885 | +0.019 | 5.1825 | 4.5731 | +0.609 (spike) |
| 1,200–1,399 | 3.6250 | 3.6151 | +0.010 | 4.7512 | 4.4294 | +0.322 (recovering) |

- **S2b-P** follows S2b-PS's path: n = 8 starts behind and closes the gap. S2b-PS's own
  gap at steps 1,200–1,399 was +0.009.
- **S2b-S at n = 8 spiked** at steps 1,056–1,137, to 8.00 at step 1,064, and again at
  step 1,204, to 6.50. By step ~1,490 its loss was back at the pre-spike level, about
  4.4. The same arm spiked three times at n = 9 (wave 1), and twice or more at both n at
  h12 ([stage_2_report.md](stage_2_report.md)). So the spikes belong to the
  shift-without-ALiBi cell, not to this run or this learning rate. No action: the run
  continues, and its `final_ce` is the result.

**At 40% (14:06):**

| Steps | S2b-P n = 8 | S2b-P n = 9 | n = 8 − n = 9 | S2b-S n = 8 | S2b-S n = 9 | n = 8 − n = 9 |
|---|---:|---:|---:|---:|---:|---:|
| 1,800–1,999 | 3.4658 | 3.4685 | −0.003 | 4.1495 | 4.8453 | −0.696 (n = 9 spike) |
| 2,200–2,399 | 3.3954 | 3.4033 | −0.008 | 4.5134 | 4.1247 | +0.389 (n = 8 spike) |
| 2,600–2,799 | 3.3415 | 3.3528 | −0.011 | 3.9616 | 3.9640 | −0.002 |
| 2,800–2,999 | 3.3173 | 3.3280 | **−0.011** | — | — | — |

- **S2b-P at n = 8 overtook n = 9** at about step 1,900 (25%) and leads by 0.011 at
  steps 2,800–2,999. That is slightly ahead of S2b-PS's path: S2b-PS led by 0.008 on the
  same steps.
- **S2b-S at n = 8 spiked twice more**: steps 2,236–2,290, to 7.47, and steps
  2,903–2,924, to 5.03. That makes four so far, against three in the whole n = 9 run.
  Between spikes the two runs are level (2,600–2,799: −0.002).

**At 58% (16:54):**

| Steps | S2b-P n = 8 | S2b-P n = 9 | n = 8 − n = 9 | S2b-S n = 8 | S2b-S n = 9 | n = 8 − n = 9 |
|---|---:|---:|---:|---:|---:|---:|
| 3,400–3,599 | 3.2480 | 3.2630 | −0.015 | 3.8046 | 3.8648 | −0.060 |
| 3,800–3,999 | 3.2117 | 3.2296 | −0.018 | 3.7415 | 3.7932 | **−0.052** |
| 4,200–4,399 | 3.1800 | 3.2007 | **−0.021** | — | — | — |

- **S2b-P at n = 8** keeps widening its lead, on S2b-PS's path.
- **S2b-S at n = 8** has not spiked since step 2,924, and leads n = 9 by about 0.05.
  n = 9 spiked again at steps 4,050–4,066. If wave 2 still ends with n = 9 best,
wave 1 names the winner, and these two runs are reported as a learning-rate check of
S2b-P and S2b-S.

## For Maciej

Items that need your decision. None blocks a run.

1. **n\* = 9 sits at the high-LR edge of Stage 1b's grid.** You set it while the grid was
   10% in (decision 1). The finished grid has both arms' lowest `final_ce` at n = 9,
   its edge ([stage_1b_report.md](stage_1b_report.md)), and Stage 1b stopped there by
   your call. Two hints that h16 AME wants a higher learning rate than n = 9:
   - at h12, S1-A0 kept improving down to n = 6
     ([stage_1_report.md](stage_1_report.md), decision 10);
   - here, S2b-PS at n = 8 leads n = 9 at 30% of the run (wave 2).

   S1b-A0 at n = 8, about 15 GPU-h, would give the 2 × 2's reference cell at n = 8 and
   test whether the h16 optimum moves.
2. **`run_exp.py` and run names with a dot**: see [stage_2_report.md](stage_2_report.md),
   "For Maciej" 2.
3. **S2b-PS's best n, 8, is the edge of the plan's check (n\* ± 1).** The plan has no edge
   rule for the check, so no n = 7 run was made. At h12, S2-PS at n = 7 and 6 did not
   beat n = 8 (3.2031 after a spike, and 3.1348, against 3.1250). S2b-PS at n = 7 would
   cost about 15 GPU-h and 15 h.
   - **Dense at n = 8 is untested at h16**, because Stage 1b stopped at n = 9. At h12, dense
     gained 0.042 from n = 9 to n = 8. So the AME-to-dense gaps measured against S1b-D at
     n = 9 probably flatter AME by a few hundredths. S1b-D at n = 8 would cost about
     14 GPU-h.
