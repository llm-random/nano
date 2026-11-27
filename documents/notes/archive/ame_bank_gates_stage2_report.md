# AME bank gates, stage 2 — report

**Archived 2026-11-27 (historical).** Done on 2026-11-08: the σ = 0.01 bank init is all of
stage 1's gain. Its recommendation, σ = 0.01 with no gate, was refined by the σ grid: σ =
1e-4 is the standard, and the code default under the QK-norm since 2026-11-25. Its
replicates were never submitted. **Today:**
[standard_model_improvements.md](../standard_model_improvements.md) ("Bank init σ (AME)").

**Status (2026-11-08 19:20): done; no job running. The init is all of stage 1's gain.**
- **Q1, yes:** σ = 0.01 with no gate (`none_std001`, 3.035771) beats production by 0.0895,
  resolved. Its share of stage-1 B3's gain is 1.00.
- **Q2, no:** B3 adds nothing at either init (+0.0001 at σ = 0.01, −0.0079 at σ = 1; both
  ties).
- **Q3, no:** B1 costs 0.07-0.09 from either start (both resolved). Its trust stays where
  it starts.
- `none_std001` reproduces stage 1's rare-token gain: 2.962 at u = 101-1000, against
  production's 3.305.
- **Recommendation: σ = 0.01, no gate** ([Results](#results)). It rests on a tie with B3,
  so four replicates (about 17 GPU-hours) are proposed, not submitted.

The plan is [ame_bank_gates_stage2_plan.md](ame_bank_gates_stage2_plan.md). This file is
its record: every job, check, result and decision, each with its reason. Times are CET.
Stage 1: [plan](ame_bank_gates_plan.md), [report](ame_bank_gates_report.md). All twelve
W&B runs of both stages are listed in
[stage 1's report](ame_bank_gates_report.md#all-runs-stages-1-and-2).

## Setup

**Questions** (the plan's):
1. Does σ = 0.01 alone, with no gate, lower final CE against production?
2. Does B3 add anything beyond the init?
3. Can B1 help if its trust doesn't start at 0.05?

**The seven cells.** The four new runs are in bold; the other three exist. Final CE is the
common evaluator's (`ame_ref32`).

| σ | No gate | B1, trust starts at 0.05 | B1, trust starts at 0.95 | B3 |
|---|---|---|---|---|
| 1 (production's) | `doi36p8k`: 3.125227 | **`b1_std1`** | — | **`b3_std1`** |
| 0.01 (stage 1's) | **`none_std001`** | stage-1 B1 (`621sjn71`): 3.141958 | **`b1open_std001`** | stage-1 B3 (`5w6p64bv`): 3.035837 |

| | |
|---|---|
| Recipe | Stage 1's [base.yaml](../../../configs/bank_gates/base.yaml), unchanged: h12, 3,815 steps = 4.0B tokens, n = 8 (LR 2⁻⁸), warmup 38, seed 123, one LEM H100, micro-batch 32 × 16, Tier 2 banks, no FSDP2, autocast bf16, `compile_blocks_full`, 7 h limit |
| Configs | `configs/bank_gates/stage2_<X>.yaml`, each `defaults: [base, _self_]` plus the plan's options; W&B names `bank_gates_stage2_<X>` |
| Evaluator | Stage 1's, unchanged: `parity_numerics.py score` (`ame_ref32`, then `ame_prod`) and `bank_gates_eval.py`, through `bank_gates_lem_eval.sh` |
| Noise floor | ±0.019, stage 1's: the 95% half-width of one run against another. A gap inside it is a tie |

## Decisions

1. **`b1open_std001` is in (2026-11-08).** It is the plan's default, and Maciej's go kept
   it. It needs the stage's one code change, `confidence_init`.
2. **`confidence_init` refuses what it cannot mean (2026-11-08).**
   - A value outside (0, 1) has no logit, so the layer raises.
   - A non-default value on any gate but `confidence` raises too. Ignoring it would let
     a run believe it had changed B1.
   - The default, 0.05, stays legal everywhere, so no stage-1 config changes.
3. **The stage-2 plan stays uncommitted (2026-11-08).** It is Maciej's file, untracked in
   `~/code/nano` (last edited 14:08). The rules let me commit only my own edits. I
   updated its status line in place, as for stage 1's plan, and left the commit to
   Maciej. Until then, the link above works in the working tree only.

## Implementation and local checks

Committed as `f313760b` on `bank_gates_lem`, rebased onto `ame_main` as **`03317133`** and
fast-forwarded into `ame_main`. Between stage 1's last code (`69e31dc1`) and `03317133`,
only this commit touches code; everything else is other sessions' docs.

**The change** ([ame.py](../../../src/core/ame.py), [attention.py](../../../src/core/attention.py)):
- `AMEAttention(confidence_init=0.05)`, passed to `_with_trust_columns`, which writes
  `logit(confidence_init)` into the H trust columns of both banks.
- The logit is the expression the old module constant was, so the default writes the
  same fp32 value as stage 1's B1: −2.9444389 (0.95 gives +2.9444389).
- `confidence_init` is added to `_AME_OPTIONS`, so `attention.ame` accepts it.

**Tests:** 393 passed, 18 skipped (`-m "not gpu and not slow"`). That is stage 1's 381
plus 12 new cases:
- the main-build-path test (through the memory-parallel copy) and the at-init score test,
  both parametrized over 0.05 and 0.95. At 0.95 every score starts at 0.9025 of its
  ungated size;
- the default and an explicit 0.05 build the same state dict, bit for bit, and the trust
  columns hold logit(0.05) exactly;
- the readout and the layer's repr report 0.95;
- the rejections in decision 2.

The GPU-marked tests cannot run here (this machine's torch has no CUDA, which fails every
gate's GPU test alike); the runs themselves exercise that path.

**CPU smoke of `b1open_std001`'s path** through `main.py` (`smoke_local.yaml`, h4):
- every layer's trust read 0.9500 at step 0 and had moved by step 2;
- the run ended normally, with its final checkpoint and `token_counts.pt`.

**Composed as `run_exp.py` composes them**, each config passed `check_point` and was built
on the meta device. Each differs from its stage-1 counterpart only in the plan's options
and the names:

| Run | Parameters | = stage 1's | Differs from | In |
|---|---:|---|---|---|
| `none_std001` | 1,103,599,056 | none (`base.yaml`) | `base.yaml` | `bank_init_std: 0.01`, an explicit `bank_gate: none` |
| `b1_std1` | 1,118,086,608 | B1 | `b1.yaml` | `bank_init_std: 1.0` |
| `b3_std1` | 1,103,617,488 | B3 | `b3.yaml` | `bank_init_std: 1.0` |
| `b1open_std001` | 1,118,086,608 | B1 | `b1.yaml` | `confidence_init: 0.95` |

All four run 1.0529 GFLOP/token, as `doi36p8k` does.

## The wave — submitted 2026-11-08 14:32-14:35

All four were submitted with `run_exp.py --config-path configs/bank_gates --config-name
stage2_<X> +infrastructure.slurm.exclude=r16-11 < /dev/null`, one at a time, from the
worktree at `03317133` (= `ame_main`, clean). Every cemetery commit has `03317133` as its
parent and changes only `exp.job`, `generated_configs/config_0.yaml` and the two pixi
files.

| Run | Job | Cemetery commit | Cemetery clone (under `.../nano/plgmstefaniak/llmrandom_cemetery/`) | W&B run | Node | Parameters, as logged |
|---|---|---|---|---|---|---:|
| `none_std001` | 6272065_0 | `c5459c14` | `bank_gates_stage2_none_std001_2026-11-08_14-32-40` | `kf4q4t4s` | r14-5 | 1,103,599,056 |
| `b1_std1` | 6272075_0 | `430f9846` | `bank_gates_stage2_b1_std1_2026-11-08_14-33-32` | `1dmp8l8w` | r14-6 | 1,118,086,608 |
| `b3_std1` | 6272076_0 | `b1806db4` | `bank_gates_stage2_b3_std1_2026-11-08_14-33-56` | `5hzd2fte` | r16-1 | 1,103,617,488 |
| `b1open_std001` | 6272078_0 | `ff4100c5` | `bank_gates_stage2_b1open_std001_2026-11-08_14-34-18` | `jc9flzsz` | r16-1 | *pending* |

Each final checkpoint lands under `checkpoint/<job>/0/step_3814/` in its clone, with
`token_counts.pt` beside it.

### Checks

The plan's checks, at about 2 minutes after each start and at step 500:
- no NaN;
- s/step within 10% of stage 1's: 3.746 with no gate, 3.771 for B3, 3.880 for B1;
- B1's trust and B3's base norms moving.

**Start, 14:39-14:42.** All four were running within a minute of submission, logged the
composed parameter counts, and trained from 14:40. Taken at steps 30-34 (step-0 readouts;
the windows are steps 10-38, against `doi36p8k`'s same steps):

| Run | Step | loss | grad_norm | grad_norm/banks, steps 10-38 | clip factor | s/step | vs stage 1 |
|---|---:|---:|---:|---|---:|---:|---:|
| `none_std001` | 34 | 6.63 | 0.65 | 0.011-0.24 | 1.00 | 3.760 | +0.4% (3.746) |
| `b1_std1` | 30 | 7.16 | 1.32 | 9e-7 to 8e-5 | 0.76 | 3.901 | +0.5% (3.880) |
| `b3_std1` | 32 | 7.03 | 0.54 | 9e-5 to 0.035 | 1.00 | 3.787 | +0.4% (3.771) |
| `b1open_std001` | 30 | 7.09 | 1.28 | 0.010-4.8 | 0.78 | 3.900 | +0.5% (3.880) |

- **No NaN; s/step within 0.5% of stage 1's** for the same gate.
- **Loss tracks production's early steps:** a mean of 7.09-7.52 over steps 10-38, against
  `doi36p8k`'s 7.34. The warmup's `grad_norm` peaks (up to 46-50) are the backbone's, as in
  `doi36p8k` (46.5 in the same steps).
- **Every mechanism starts at its init** in every layer: trust 0.05 (`b1_std1`) and 0.95
  (`b1open_std001`), base norms 0 (`b3_std1`). Step 500 reads whether they move.
- **`b1_std1`'s banks get almost no gradient, as the plan feared.**
  - Its `grad_norm/banks` is 9e-7 to 8e-5, four orders of magnitude below
    `none_std001`'s. Stage-1 B1 (σ = 0.01) started at 4e-5.
  - Two factors shrink it. A σ = 1 row is 8 long per head, so the norm divides its
    gradient by 8 rather than 0.08. The trusts (0.05 × 0.05) scale the scores, and
    so the gradient, down by another 400.
  - The banks' AdamW uses the default ε = 1e-8. Spread over the hundreds of millions of
    bank entries a batch touches, a norm of 1e-5 leaves a typical entry's gradient far
    below ε. Adam's step is then about g/ε of the learning rate rather than the learning
    rate itself, so the rows and trusts may hardly train. This is a reading to check, not
    a fault: nothing is cancelled.

**`b1open_std001` spiked at steps 381-391, 15:05.** The step-500 watch flagged it at step
390, against a low of 4.13.
- **The loss:** 4.10-4.13 at steps 350-370, creeping to 4.34 by step 380. It then rose
  from step 381, peaked at 7.16 at step 391, and was back to 5.63 by step 398.
- **The gradient:** `grad_norm` peaked at 4.96 at step 388, and the clip was active
  (factor 0.20-0.80) at steps 382-394.
- **The backbone's, not the banks':** the banks were 0.06-0.21 of the norm throughout
  (at most 4.5% of the squared norm).
- **The banks' share before the spike was `none_std001`'s:** a median of 0.25 of the
  norm over steps 340-380, against 0.26. At trust 0.95, B1 passes nearly the whole score
  gradient, so its banks train like the ungated σ = 0.01 run's.
  - Stage-1 B1, at trust 0.05, had 0.014.
  - Both σ = 1 runs had about 0.01 (`b3_std1` 0.009, `b1_std1` 0.006).
- **Like stage 1's h16 spikes:** B2's at steps 345-359 (to 7.97) and B3's at 404-416 (to
  6.56) were of the same kind, and both runs recovered and finished normally.
- **Not a crash or a NaN,** so nothing is cancelled or resubmitted. Step 500 shows whether
  it recovered.

| Step | 370 | 380 | 382 | 385 | 388 | 391 | 394 | 398 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train loss | 4.13 | 4.34 | 4.88 | 5.88 | 6.63 | 7.16 | 6.19 | 5.63 |
| `grad_norm` | 0.18 | 0.40 | 1.33 | 2.54 | 4.96 | 2.45 | 1.25 | 0.65 |

**Step 500, 15:15.** Every check passes. `b1open_std001` has recovered from its spike.

| Run | Loss, steps 400-499 (min/mean/max) | vs `doi36p8k` (4.33) | `grad_norm/banks` | Clip factor | s/step | At step 500 (layer mean [range]) |
|---|---|---:|---:|---|---:|---|
| `none_std001` | 3.80/3.89/4.00 | −0.44 | 0.027 | 1.000 | 3.759 | entropy 0.49; τ 8.08 |
| `b1_std1` | 4.31/4.44/4.59 | +0.11 | 0.0015 | 1.000 | 3.900 | trust mean 0.0501 (q), p90 0.0505 (q) and 0.0501 (k); entropy 0.69; τ 8.72 [8.15, 9.95] |
| `b3_std1` | 4.16/4.30/4.44 | −0.03 | 0.0014 | 1.000 | 3.788 | base norms 1.87 [1.54, 2.24] (q) and 1.28 [1.11, 1.64] (k); entropy 0.51; τ 8.49 |
| `b1open_std001` | 4.16/4.48/5.41 (the spike's tail) | +0.15 | 0.032 | 1.000 | 3.900 | trust mean 0.950, p90 0.9522 (q and k); entropy 0.44; τ 8.24 |
| stage-1 B3 (`5w6p64bv`) | 3.78/3.88/3.98 | −0.45 | 0.025 | | | base norms 0.30 (q) and 0.29 (k) |
| stage-1 B1 (`621sjn71`) | 4.31/4.47/4.62 | +0.14 | 0.0027 | | | trust mean 0.0516, p90 0.0564 (q) |

- **No NaN; s/step steady** within 0.5% of stage 1's.
- **`b1open_std001` recovered.**
  - The clip was back at 1.0 from step 395.
  - The loss was 4.16 by step 520, against 4.10-4.13 before the spike.
  - Its 400-499 mean still carries the spike's tail. Before the spike it was already
    behind `none_std001`: 4.44 against 4.33 over steps 200-299.
- **B3's bases move fast.** 0 → 1.87 (q) and 1.28 (k) by step 500, six times stage-1
  B3's. That is the length they need to count against rows 8 long: a quarter of the
  row, where stage-1 B3's 0.3 already outweighed its 0.08-long rare rows.
- **B1's trust moves, but only in the third decimal place, from either start.**
  - `b1_std1`'s p90 went 0.0500 → 0.0505, stage-1 B1's 0.0500 → 0.0564, and
    `b1open_std001`'s 0.9500 → 0.9522.
  - At p90 that is a logit drift of +0.010, +0.127 and +0.047, in the same order.
    (Corrected at 19:05: the first version swapped the last two.) That is the order of
    lr × √steps ≈ 0.09, the drift of a gradient with no consistent sign.
  - Starting the trust open does not let it learn. It mostly stays where it is put.
- **`b1_std1`'s banks are no longer starved.** Its `grad_norm/banks` grew a
  thousandfold from the start, to 0.0015 over steps 400-499. That is level with
  `b3_std1`'s 0.0014, at the same σ, so Adam's ε matters for the first steps at most.
  Its trust is still the slowest of the three B1 runs.
- **Early ordering, not a result:** the init alone (`none_std001`, 3.89) is level with
  stage-1 B3 (3.88) at steps 400-499, 0.44 below production. Both σ = 1 runs stay near
  production: `b3_std1` −0.03, `b1_std1` +0.11. Stage 1's early leads shrank but held to
  the end, so only final CE answers the questions.

### Finish — 2026-11-08 18:50-19:00

All four completed (Slurm `COMPLETED`, exit 0) at step 3,814, after 4,000,317,440 tokens.
The run-to-finish watch read W&B and Slurm every 5 minutes from 15:17 to the end and
flagged nothing.

| Run | Job | Elapsed | s/step, median | vs `doi36p8k` | Peak memory, GiB | Train loss, last 100 steps | In-training `final_ce` (bf16) |
|---|---|---|---:|---:|---:|---:|---:|
| `none_std001` | 6272065_0 | 4:17:33 | 3.758 | +0.3% | 70.40 | 3.108 | 3.0352 |
| `b1_std1` | 6272075_0 | 4:26:01 | 3.900 | +4.1% | 70.66 | 3.295 | 3.2100 |
| `b3_std1` | 6272076_0 | 4:18:16 | 3.786 | +1.1% | 70.58 | 3.204 | 3.1182 |
| `b1open_std001` | 6272078_0 | 4:24:28 | 3.900 | +4.1% | 70.66 | 3.183 | 3.1084 |
| stage-1 B3 | 6264726_0 | 4:14:02 | 3.771 | +0.7% | 70.58 | 3.108 | 3.0361 |
| stage-1 B1 | 6264723_0 | 4:21:18 | 3.880 | +3.6% | 70.66 | 3.225 | 3.1406 |
| `doi36p8k` | 6256577_0 | 4:12:54 | 3.746 | | 70.40 | 3.212 | 3.125 |

- **Clean apart from `b1open_std001`'s spike,** over every step of the full W&B histories:
  - no NaN or inf in any run;
  - one loss spike in the four: `b1open_std001`'s, at steps 382-438, peak 7.16 at 391. It is
    the one recorded above. A spike here is a step more than 0.3 above the median of the
    50 before it; the loss stayed that far above its pre-spike 4.16 until step 438;
  - after the warmup the clip acted on 14 steps in all. Eleven were `b1open_std001`'s, all
    inside the spike (steps 382-394, minimum 0.20 at 388). Three were `b3_std1`'s 38, 39
    and 41, as the warmup ended (minimum 0.58). It never acted in `none_std001` or
    `b1_std1`.
- **Checkpoints:** each `step_3814/` holds three files:
  - the model, 2.12 GB;
  - the banks' sidecar, 11.13 GB (11.30 GB for the two B1 runs, wider by the trust
    columns);
  - `token_counts.pt`.
- **The four `token_counts.pt` are byte-identical with stage 1's** (sha256 `ab1df510…`; stage
  1 bucketed `doi36p8k` with these counts too). All seven cells saw the same tokens in the
  same steps.
- **Train-loss windows** (means). Same seed and batches, so the gaps are paired; final CE
  from the common evaluator still decides:

  | Steps | `none_std001` | stage-1 B3 | `b1open_std001` | `b3_std1` | `doi36p8k` | stage-1 B1 | `b1_std1` |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | 500-999 | 3.6460 | 3.6437 | 3.8755 | 3.8768 | 3.8963 | 4.0228 | 4.0411 |
  | 1000-1999 | 3.3787 | 3.3813 | 3.5099 | 3.5087 | 3.5198 | 3.5799 | 3.6453 |
  | 2000-3433 | 3.1900 | 3.1910 | 3.2758 | 3.2929 | 3.3027 | 3.3227 | 3.3966 |
  | 3434-3814 | 3.1140 | 3.1142 | 3.1908 | 3.2102 | 3.2192 | 3.2325 | 3.3024 |

  `none_std001` and stage-1 B3 agree within 0.003 in every window.
- **Readouts at step 3,800** (layer means; the range over layers in brackets):

  | Run | Readouts |
  |---|---|
  | `none_std001` | τ 8.04 [7.90, 8.29]; entropy 0.52 |
  | `b1_std1` | trust mean 0.051 (q and k), p90 0.053 (q) and 0.051 (k); 0.002-0.003% of (token, head) entries above 0.5; τ 10.5 [9.9, 11.2]; entropy 0.66 |
  | `b3_std1` | base norms 4.79 [2.98, 6.17] (q) and 2.24 [2.01, 2.50] (k); τ 8.53 [7.94, 9.70]; entropy 0.42 |
  | `b1open_std001` | trust mean 0.950 (q and k), p90 0.954 (q) and 0.955 (k); every entry above 0.5; τ 8.11 [7.93, 8.36]; entropy 0.52 |
  | stage-1 B1 | trust mean 0.055 (q) and 0.057 (k), p90 0.068 both; τ 11.3 [9.7, 12.8]; entropy 0.54 |
  | stage-1 B3 | base norms 0.47 [0.27, 0.58] (q) and 0.58 [0.40, 0.83] (k); τ 8.20; entropy 0.51 |

- **B1's trust ends where it started, from either start.**
  - Over the whole run, the p90 logit drifted +0.06 (q) and +0.03 (k) in `b1_std1`, and
    +0.10 and +0.11 in `b1open_std001`. Stage-1 B1's drifted +0.32 and +0.33.
  - lr × √steps is now 0.24, so all of these are the size of a drift with no consistent
    sign. The means moved less still: +0.02, −0.01 and +0.11.
  - No B1 run learned a per-token, per-head confidence. `b1_std1`'s scores end at
    s_q·s_k ≈ 0.0026 of their ungated size; `b1open_std001`'s end at 0.90.
- **`b1_std1`'s banks trained after a slow start.** `grad_norm/banks`, median per window:

  | Steps | `none_std001` | `b1open_std001` | `b3_std1` | `b1_std1` |
  |---|---:|---:|---:|---:|
  | 0-99 | 0.042 | 0.035 | 3.4e-4 | 1.0e-5 |
  | 100-499 | 0.037 | 0.040 | 1.3e-3 | 7.8e-4 |
  | 500-999 | 0.018 | 0.022 | 1.4e-3 | 2.0e-3 |
  | 1000-1999 | 0.012 | 0.014 | 1.3e-3 | 1.5e-3 |
  | 2000-3433 | 0.0083 | 0.0094 | 1.1e-3 | 1.1e-3 |
  | 3434-3814 | 0.0077 | 0.0083 | 9.8e-4 | 9.9e-4 |

  - From step 500 on, `b1_std1`'s bank gradient is level with `b3_std1`'s, at the same σ.
    Adam's ε could have held it back only in the first few hundred steps.
  - So its CE reads as B1's at σ = 1, not as a run whose banks never trained.
  - At σ = 1 both runs' bank gradients stay about 8 times below σ = 0.01's to the end.

### Scoring — submitted 2026-11-08 19:01

Four one-GPU jobs were run through `bank_gates_lem_eval.sh` from the worktree at
**`c6d46c64`** (= `ame_main`, clean; archived at `.../bank_gates/code/c6d46c64`).
- Since the runs' `03317133`, `ame_main` gained only other sessions' docs and Stage 1c/2c
  configs. Nothing under `src/`, `documents/scripts/`, `configs/bank_gates/` or the base
  configs changed.
- Since stage 1's h12 scoring at `8b07c983`, the only `src/` change is `confidence_init`,
  whose default is stage 1's B1 bit for bit. So the three existing cells' scores stand,
  and all seven cells share one evaluator.

Each job scores its run's `step_3814` checkpoint and its own `token_counts.pt`, in one go:
1. `parity_numerics.py score`: `ame_ref32`, then `ame_prod`, on the 8,192 × 2,049 held-out
   tokens;
2. `bank_gates_eval.py`: the buckets by u, with the readouts.

| Run | Eval job | Node |
|---|---|---|
| `none_std001` | 6275729 | r16-1 |
| `b1_std1` | 6275730 | r16-1 |
| `b3_std1` | 6275731 | r16-2 |
| `b1open_std001` | 6275732 | r16-2 |

All four started at 19:02.

All four completed at 19:11, in about 9.5 minutes each.

### Results

**Final CE, common evaluator** (`ame_ref32`; 8,192 held-out sequences, 16.8M
predictions). The four new cells are in bold, beside stage 1's three.

| Rank | Cell | σ | Gate | `ame_ref32` | − `doi36p8k` | `ame_prod` |
|---|---|---|---|---:|---:|---:|
| 1 | **`none_std001`** | 0.01 | none | **3.035771** | −0.0895 | 3.035911 |
| 2 | stage-1 B3 (`5w6p64bv`) | 0.01 | B3 | 3.035837 | −0.0894 | 3.035979 |
| 3 | **`b1open_std001`** | 0.01 | B1, trust from 0.95 | 3.107609 | −0.0176 | 3.107753 |
| 4 | **`b3_std1`** | 1 | B3 | 3.117312 | −0.0079 | 3.117461 |
| 5 | `doi36p8k` (production) | 1 | none | 3.125227 | | 3.125373 |
| 6 | stage-1 B1 (`621sjn71`) | 0.01 | B1, trust from 0.05 | 3.141958 | +0.0167 | 3.142109 |
| 7 | **`b1_std1`** | 1 | B1, trust from 0.05 | 3.211324 | +0.0861 | 3.211473 |

Stage 1's consistency checks hold:
- each new bucket pass reproduced its run's `ame_ref32` to six digits;
- `ame_prod` sits 0.00014-0.00015 above `ame_ref32` in every new run;
- the in-training `final_ce` (bf16) ranks all seven cells the same way.

**The plan's contrasts** (a − b). A tie is a gap inside ±0.019. The paired train-loss gap
(steps 3,434-3,814; same seed and batches) agrees with every verdict.

| Contrast | Formula | Gap | ÷ 0.019 | Verdict | Train loss, paired | Answers |
|---|---|---:|---:|---|---:|---|
| Init alone | `none_std001` − `doi36p8k` | −0.0895 | 4.7 | resolved, lower | −0.105 | Q1 |
| Init under B3 | stage-1 B3 − `b3_std1` | −0.0815 | 4.3 | resolved, lower | −0.096 | Q1 |
| Init under B1 | stage-1 B1 − `b1_std1` | −0.0694 | 3.7 | resolved, lower | −0.070 | Q1 |
| B3 beyond the init | stage-1 B3 − `none_std001` | +0.0001 | 0.0 | tie | +0.000 | Q2 |
| B3 at σ = 1 | `b3_std1` − `doi36p8k` | −0.0079 | 0.4 | tie | −0.009 | Q2 |
| B1 started open | `b1open_std001` − `none_std001` | +0.0718 | 3.8 | resolved, higher | +0.077 | Q3 |
| B1 at σ = 1 | `b1_std1` − `doi36p8k` | +0.0861 | 4.5 | resolved, higher | +0.083 | Q3 |

**The init's share of stage-1 B3's gain:** (3.035771 − 3.125227) ÷ (3.035837 − 3.125227) =
**1.00**. The init is all of it.

**Q1. Does σ = 0.01 alone lower final CE against production? Yes, by 0.0895, resolved.**
- That is 4.7 half-widths. The init lowers CE under every gate too: by 0.0815 under B3
  and 0.0694 under B1.
- The paired train-loss gap was widest early (−0.83 over steps 100-199) and then
  narrowed. Over the last three windows it held at −0.115, −0.107 and −0.105 (steps
  2,000-2,999, 3,000-3,433 and 3,434-3,814).

**Q2. Does B3 add anything beyond the init? No, at either init.**
- Against `none_std001` at σ = 0.01 it is +0.0001, and against production at σ = 1 it is
  −0.0079. Both are ties.
- B3 led `none_std001` early, by 0.05-0.09 in paired train loss over steps 0-199. From
  step 440 on, the window means stayed within 0.0032 of each other.
- The plan expected `b3_std1` to tie production, since rows 8 long would leave the base
  nothing to do. It ties, but the base was not idle: its norms reached 4.8 (q) and 2.2
  (k) (Finish). The buckets below show what that bought.

**Q3. Can B1 help if its trust doesn't start at 0.05? No.**
- Started open, B1 ends 0.0718 behind no gate. At σ = 1 it ends 0.0861 behind production.
  Both are resolved.
- **The trust stays where it is put,** in every run and bucket: 0.949-0.950 in
  `b1open_std001`, 0.050-0.054 in `b1_std1`. Starting it open does not let B1 learn a
  per-token confidence. It only runs the scores at 0.90 of their size instead of 0.0026.
- **`b1open_std001`'s deficit is not its spike's.** In paired train loss it was +0.19 over
  steps 0-99, +0.10 just before the spike (steps 300-380) and +0.077 at the end. Its
  attention readouts match `none_std001`'s (below), so they do not locate the cost.
- **`b1_std1` was checked first, as the plan asks.** Its banks trained: their gradient was
  level with `b3_std1`'s from step 500 (Finish). Its trust never opened, though, so its
  scores barely depend on token identity (below). Its CE is B1's at σ = 1, not a run's
  whose banks never trained.

**Recommendation: σ = 0.01 with no gate, i.e. `none_std001`'s config.**
- **By the rule:** `none_std001` has the lowest CE. Stage-1 B3 ties it (+0.0001), and
  every other cell is 0.07 or more behind. Of the tied pair, both at σ = 0.01, no gate
  ranks before B3.
- **Against production it is one option,** `common.attention.ame.bank_init_std: 0.01`.
  With no gate, nothing has to be folded into the banks at export.
- **The choice between it and B3 rests on a tie,** so replicates are proposed below. Only
  a B3 lead of more than about 0.010 over three runs each would overturn it. Today's gap
  is +0.0001 the other way, and +0.0002 in paired train loss.
- **Measured at h12 (4B tokens) only.** Every stage-1 h16 arm also started at σ = 0.01, so
  the init's own gain at h16 is unmeasured. If B3 again adds nothing there, its −0.050 at
  h16 is that gain's size. One `none_std001` run at h16 (about 15.3 GPU-hours) would
  measure it. Flagged, not submitted.

**Replicates for the tie (proposed; not submitted).**
- **Which runs:** two more of `none_std001` and two more of stage-1 B3 (`b3.yaml`), at the
  plan's same-seed recipe.
- **What they resolve:** with three runs per arm, the 95% half-width falls to about
  ±0.010, as in stage 1's proposal.
- **What else they measure:** LEM's own same-seed spread at h12.
- **Cost:** about 4.3 h per run on one LEM H100, all four in parallel: about 17 GPU-hours.
- **Status:** waiting for Maciej's go.

**Beyond the seven cells** (context, not part of the rule):
- **Stage 1's other h12 gates add nothing resolved either.** B2 (3.035505) is 0.0003
  below `none_std001` and ST (3.019112) 0.0167 below it: both ties.
- **σ works the same lever as `bank_lr_ratio`.**
  [ame_improve_runs_report.md](ame_improve_runs_report.md) found that 4× the banks'
  learning rate, at σ = 1, lowers h12's in-training `final_ce` by 0.031.
  - Under the hard QK-norm only a row's direction counts.
  - Adam moves each entry by about the learning rate, whatever the row's size.
  - So a smaller σ turns the rows' directions faster, as a larger bank learning rate does
    (buckets, point 2). Whether the two stack is untested.

**Rare-row buckets.** Every cell was bucketed with the same counts, so the composition is
stage 1's h12 one ([table](ame_bank_gates_report.md#results)).
- u = 1-10 is in effect one token, ' COVID'.
- u = 0 and u = 11-100 are a handful of documents.
- So the reading rests on u = 101-1000 (stage 1's decision 8).

**Loss** at positions whose input token is in the bucket:

| u | `none_std001` | stage-1 B3 | `b1open_std001` | `b3_std1` | `doi36p8k` | stage-1 B1 | `b1_std1` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 (5 positions) | 5.44 | 7.09 | 5.62 | 5.75 | 6.26 | 5.60 | 6.39 |
| 1-10 (' COVID') | 6.04 | 4.73 | 5.61 | 4.94 | 3.99 | 4.97 | 4.59 |
| 11-100 | 4.00 | 4.02 | 4.07 | 4.47 | 4.44 | 4.34 | 4.52 |
| **101-1000** | **2.962** | **2.952** | **3.020** | **3.339** | **3.305** | **3.416** | **3.563** |
| 1001-3814 | 2.359 | 2.354 | 2.417 | 2.474 | 2.471 | 2.525 | 2.663 |
| 3815 | 3.121 | 3.122 | 3.195 | 3.198 | 3.208 | 3.220 | 3.280 |

**The key question: does `none_std001` reproduce stage 1's rare-token gains? Yes.**
- At u = 101-1000 it reaches 2.962. That is within stage 1's band (ST 2.942, B3 2.952, B2
  2.963) and 0.343 below production's 3.305.
- Its gain over production matches the gated arms' in every bucket with data:

| u | `none_std001` − `doi36p8k` | stage-1 B3 | stage-1 ST | stage-1 B2 | Share of `none_std001`'s overall gain |
|---|---:|---:|---:|---:|---:|
| 11-100 | −0.448 | −0.425 | −0.344 | −0.362 | 0.0% |
| 101-1000 | −0.343 | −0.353 | −0.363 | −0.342 | 0.4% |
| 1001-3814 | −0.112 | −0.117 | −0.128 | −0.111 | 14% |
| 3815 | −0.087 | −0.086 | −0.103 | −0.087 | 86% |

A bucket's share is its gap times its fraction of the held-out positions, divided by the
overall gap (−0.0895). ' COVID' takes back 0.2%.

**Influence as queries and as keys**, at u = 101-1000 / u = 3815:

| | `none_std001` | stage-1 B3 | `b1open_std001` | `b3_std1` | `doi36p8k` | stage-1 B1 | `b1_std1` |
|---|---|---|---|---|---|---|---|
| Identity spread as queries (std of q·k over the visible keys) | 1.21 / 1.16 | 1.26 / 1.19 | 1.21 / 1.15 | 1.40 / 1.48 | 0.90 / 1.00 | 0.32 / 0.64 | 0.04 / 0.23 |
| Attention entropy as queries (÷ ln of the visible keys) | 0.532 / 0.519 | 0.486 / 0.510 | 0.539 / 0.517 | 0.416 / 0.421 | 0.561 / 0.519 | 0.577 / 0.543 | 0.689 / 0.656 |
| Mass received as keys ÷ uniform share | 0.933 / 0.791 | 0.882 / 0.725 | 0.900 / 0.776 | 0.684 / 0.639 | 0.835 / 0.808 | 0.782 / 0.748 | 0.932 / 0.901 |
| q row length per head | 0.72 / 1.01 | 0.69 / 0.99 | 0.74 / 1.07 | 7.97 / 7.96 | 7.97 / 8.00 | 0.32 / 0.74 | 7.97 / 7.97 |

**Mechanism readouts by bucket:** B1's trust per (token, head), and B3's cos(row, base).

| u | `b1open_std001` trust, q / k | `b1_std1` trust, q / k | stage-1 B1 trust, q / k | `b3_std1` cos(row, base), q / k | stage-1 B3 cos(row, base), q / k |
|---|---|---|---|---|---|
| 0 | 0.9500 / 0.9500 | 0.0500 / 0.0500 | 0.0500 / 0.0500 | 0.000 / 0.000 | 0.000 / −0.001 |
| 1-10 | 0.9500 / 0.9500 | 0.0500 / 0.0500 | 0.0501 / 0.0500 | 0.000 / 0.001 | 0.009 / −0.001 |
| 11-100 | 0.9500 / 0.9500 | 0.0500 / 0.0500 | 0.0506 / 0.0501 | 0.000 / 0.000 | 0.013 / −0.007 |
| 101-1000 | 0.9499 / 0.9501 | 0.0501 / 0.0500 | 0.0522 / 0.0509 | 0.002 / 0.000 | 0.005 / −0.015 |
| 1001-3814 | 0.9496 / 0.9500 | 0.0504 / 0.0502 | 0.0548 / 0.0551 | 0.006 / 0.001 | 0.022 / 0.009 |
| 3815 | 0.9491 / 0.9493 | 0.0530 / 0.0535 | 0.0612 / 0.0686 | 0.013 / 0.003 | 0.076 / 0.026 |

**What the buckets show:**
1. **The init alone reproduces stage 1's rare-token gain, and its whole profile.**
   - The gain per token grows as tokens get rarer: −0.087 at u = 3815, −0.343 at u =
     101-1000, −0.448 at 11-100.
   - Most of the total still comes from common tokens. Those seen in every step carry 86%
     of it.
2. **Under σ = 0.01 the rows' directions are learned; under σ = 1 they stay mostly
   random.** This is a reading of the row lengths, not a direct measurement.
   - σ = 0.01 rows grow from 0.08 per head to about 0.7 at u = 101-1000 and 1.0 at
     u = 3815 (less under stage-1 B1's shut trust). Almost all of their direction is
     learned.
   - σ = 1 rows end 7.96-8.00 long in every bucket, against 7.97 for rows that never
     trained (u = 0).
   - Adam moves each entry by about the learning rate per step, whatever the row's size,
     so a σ = 1 row gains a learned part of similar length, about 0.6-1 per head.
     Production's rows growing from 7.97 to 8.00 fits a part about 0.6 long at right
     angles to the start. Against a start 8 long, that turns a row by only 5-7° (cos ≈
     0.99).
   - So the init decides how much of each row's direction training can reach. Rare
     queries' identity logits come out sharper than production's (1.21 against 0.90),
     and rare keys draw more mass (0.93 against 0.84). Stage 1 found the same signature
     in ST, B2 and B3.
3. **B3's one effect beyond the init is on the token it was built for.**
   - At u = 1-10 (' COVID', seen in 2 training steps), stage-1 B3 is 1.32 below
     `none_std001` (4.73 against 6.04). Its backoff replaces the untrained row with the
     base.
   - That is 1,702 of 16.8M predictions, worth 0.0001 of CE. Production's untrained random
     row does better still there (3.99).
   - From u = 11 up, stage-1 B3 is within 0.023 of `none_std001` in every bucket.
4. **At σ = 1, B3's base sharpens attention but buys nothing.**
   - The base reaches 4.8 (q) and 2.2 (k) against rows 8 long. It stays nearly orthogonal
     to them (|cos| ≤ 0.013), so it moves their directions through its length, as in
     stage 1.
   - `b3_std1`'s queries are the sharpest of all (identity spread 1.40, entropy 0.42), and
     its rare keys draw the least mass (0.68).
   - Yet its loss is 0.034 above production's at u = 101-1000, and only 0.009 below it at
     u = 3815.
5. **B1's trust does not move from either start.**
   - At 0.05 (`b1_std1`), the scores run at 0.0026 of their size. The identity spread
     collapses to 0.04 at u = 101-1000 (production: 0.90), the attention is the most
     diffuse (entropy 0.69), and the loss is the worst in every bucket from u = 101 up.
   - At 0.95 (`b1open_std001`), the readouts at u = 101-1000 match `none_std001`'s:
     identity spread 1.21 in both, entropy 0.539 against 0.532, row lengths within 3%.
     Yet its loss is 0.06-0.08 higher in every bucket from u = 11 up. Whatever costs it
     does not show here.
   - Stage-1 B1, at σ = 0.01, sits between them. Its rows learned (0.32 long at
     u = 101-1000), but its trust at 0.05 holds the identity spread to 0.32.
