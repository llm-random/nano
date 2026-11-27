# AME init follow-up: pkmf-scale `Linear`s and head under the old embedding (plan and log)

**Archived 2026-11-27 (historical).** Done on 2026-11-10: pkmf-scale `Linear`s and head cost
about +0.10 in both models, so the current init stays. **Today:**
[standard_model_improvements.md](../standard_model_improvements.md) ("Other rejected
changes").

**Status:** DONE, 2026-11-10. Both runs ran on entropy (moved from LEM; *Decisions*
3). pkmf-scale `Linear`s and head cost about +0.10 in both models (*Analysis*).
Maciej asked for them on 2026-11-10.
Background:
- [ame_init_embed_ablation.md](ame_init_embed_ablation.md), its *Analysis, step 1*;
- [ame_bank_gates_stage2_report.md](ame_bank_gates_stage2_report.md).

**Question:** take the best old-init h12 runs and change only the `Linear`s' and the
head's init to pkmf's √(0.1/fan_in). Does that help or hurt?
- AME is compared with `kf4q4t4s`: σ = 0.01 banks, no gate.
- Dense is compared with `x0xlqroj`.
- Both run at n = 8.

## Why

- **AME.** In ame_init_embed_ablation.md, the new init's head, `Linear`s and banks
  together came out neutral for AME (−0.0001, from `oldinit_embnowd` to `oldemb`). Yet
  σ = 0.01 banks alone gain −0.105 in paired train loss (bank-gates stage 2).
  - The inference there: the new head and `Linear`s cost about +0.10 and cancel the
    banks' gain.
  - No run isolated them. This one does, against `kf4q4t4s`, which already has the
    σ = 0.01 banks.
- **Dense.** The new head and `Linear`s cost +0.2325 there, but that run had a Xavier
  head and an undecayed embedding. This run prices pkmf-scale `Linear`s and head with
  the embedding as `x0xlqroj` had it.

## The runs

Maciej's spec (2026-11-10). One change from each reference: the `Linear`s' and the
head's init.

| Weight | Both runs | The references |
|---|---|---|
| Token embedding | N(0, 1) cut at ±2 (std 0.88), decayed at 0.1 | the same |
| AME banks (`q_bank`, `k_bank`) | N(0, 0.01²), not cut, not decayed | the same (`kf4q4t4s`) |
| Attention `Linear`s: AME's V and O (its Q and K are the banks), dense's Q, K, V and O | **√(0.1/fan_in), cut at ±2σ** | 1/√fan_in, cut at ±2σ |
| FF `Linear`s: in, gate and out | **√(0.1/fan_in), cut at ±2σ** | 1/√fan_in, cut at ±2σ |
| Head | **√(0.1/768) = 0.0114, cut at ±2σ** | 1/√768 = 0.0361, cut at ±2σ |
| Everything else | as the reference | |

- **`pkmflin_ame_h12`.** S2-PS (AME with the KV shift and ALiBi), no gate, h12.
  - Recipe: 3,815 steps = 4.0B tokens, n = 8 (LR 2⁻⁸), warmup 38, seed 123; one H100
    (entropy's, moved from LEM: *Decisions* 3), micro-batch 32 × 16, Tier 2 banks, no
    FSDP2, autocast bf16,
    `compile_blocks_full`; 7 h limit.
  - Reference: `kf4q4t4s` (`bank_gates_stage2_none_std001`), in-training final_ce
    3.0352, 3.758 s/step, 70.40 GiB.
- **`pkmflin_dense_h12`.** S1-D (GQA with RoPE): the same recipe without banks, with a
  5 h limit.
  - Reference: `x0xlqroj` (`s1_d_h12_lr8`), final_ce 2.9824, 3.272 s/step.
- **The cut is relative (±2σ), as asked.** A weight's actual std is therefore 0.88σ,
  as our `Linear`s always had. pkmf's own draw cuts at a fixed ±0.2. At these σ
  (0.0057-0.0114) that is 17-35σ out, so it never bites
  ([pkmf_init_comparison.md](../pkmf_init_comparison.md)).

## How to analyze

- **Each run:**
  - final_ce and its Δ against the reference;
  - the paired same-seed train-loss gap per quarter and over the last 500 steps
    ([scripts/paired_gap.py](../../../scripts/paired_gap.py), on the branch);
  - health: spikes, the grad-norm max after step 100, s/step and peak memory;
  - block 0's embedding norm against its attention update norm.
- **The paired gap decides.** A gap inside ±0.022 is a tie: that is AME's floor, and
  dense borrows it.
- **AME, against `kf4q4t4s`:** the gap is the price of pkmf-scale `Linear`s and head,
  given σ = 0.01 banks.
  - Near +0.10: confirms the inference above. The `Linear`s and head cancel the banks'
    gain.
  - Inside ±0.022: they are neutral here. The cancellation in
    ame_init_embed_ablation.md then came from something this run keeps at the old
    values: the Xavier head (0.0063, against 0.0114 here) or the Xavier banks (0.0063,
    against 0.01).
  - Below −0.022: they add a gain on top of the banks'.
- **Dense, against `x0xlqroj`:** the gap is the price of pkmf-scale `Linear`s and head
  with the embedding decayed. Set it against ame_init_embed_ablation.md's +0.2325,
  which had a Xavier head and an undecayed embedding.

## Log

### Implementation

2026-11-10, on branch `ame_initialization` (worktree `~/code/nano-ame-init`).

- **Code: `model.embedding.weight_decay`**
  ([src/core/model.py](../../../src/core/model.py), `TransformerEmbedding`).
  - The branch exempts the embedding from decay by a class attribute, so no config
    could decay it again.
  - `weight_decay: true` sets an instance attribute that shadows the exemption.
  - The default is false, so no other config changes. Anything but a bool is refused.
- **Tests:** two new ones in
  [tests/test_init_defaults.py](../../../tests/test_init_defaults.py).
  - With the knob on, the embedding is back in the decayed group, and a dense model is
    back to the one group holding exactly `model.parameters()`, as stage 1-2's dense
    runs had.
  - A non-bool value is refused.
  - The four test files that touch `parameter_groups` pass on CPU: 177 passed, 13 GPU
    tests deselected.
- **Configs.** Each writes out every init, so the branch's defaults don't reach it:
  - [pkmflin_ame_h12.yaml](../../../configs/ame_init/pkmflin_ame_h12.yaml) is a copy of
    `configs/bank_gates/base.yaml` with `stage2_none_std001`'s two attention options;
  - [pkmflin_dense_h12.yaml](../../../configs/ame_init/pkmflin_dense_h12.yaml) is a copy
    of `oldinit_embnowd_dense_h12.yaml`.
- **Against the reference's W&B-logged config** (composed as `run_exp.py` does, with
  `check_point`, then resolved), the only training keys that differ are the init's:
  - `model.embedding.weight_decay`: absent → true (the references were decayed by
    default);
  - `attention_fn.init_scale`, the scales of the FF's `ff_pre_act_fn`, `gate_fn` and
    `ff_post_act_fn`, and the head's scale: 1 → 0.3162.

  Everything else that differs is names, tags and runtime keys (job id, paths). For
  dense it also includes Slurm's `mem_per_gpu` (220G → 228G, LEM's default since) and
  the time limit (7 h → 5 h).

- **Built on CPU with the real init** (seed 123), each against its reference's
  config: the new runs on the branch, the references on `ame_main`.
  - `ame_main` hasn't changed `src/` or any config these runs read since `03317133`,
    `kf4q4t4s`'s commit.
  - Dense reference: `configs/stage_1/dense_h12_lr_grid.yaml` set to n = 8 (the init
    doesn't read n).
  - AME reference: `configs/bank_gates/stage2_none_std001.yaml`.

  | | `pkmflin_dense_h12` | `pkmflin_ame_h12` |
  |---|---|---|
  | Parameters | 111 (183,454,464), as the reference | 171 (1,103,599,056), as the reference |
  | Bit-identical to the reference | 26: the embedding and the 25 norm weights | 110: the embedding, the banks, τ, α, β, the sinks and the norms |
  | Exactly √0.1 × the reference's draw | 85: Q, K, V, O, FF in, gate and out in 12 blocks, and the head | 61: V, O, FF in, gate and out in 12 blocks, and the head |
  | Embedding std, max \|w\| | 0.8798, 2.000 | 0.8798, 2.000 |
  | Head, attention, FF in and gate (fan_in 768): std, max \|w\| | 0.0100, 0.0228 (reference 0.0317, 0.0722) | the same |
  | FF out (fan_in 3072): std | 0.00502 (reference 0.0159) | the same |
  | Banks: std | — | 0.0100, as the reference |
  | Weight decay 0.1 / 0 (numel) | 183,454,464 / 0: one group, as the reference | 176,376,576 / 927,222,480, as the reference, with the embedding decayed |

  The worst element is 1.9e-7 off the common ratio √0.1, which is float32 rounding.
  The cut is relative, so the new draws use the same uniform samples as the
  reference's. Every parameter drawn after them, the banks included, therefore
  starts exactly as in the reference.

### Runs

Submitted from `~/code/nano-ame-init` at `3c55ed99` (code, configs and this doc),
with each config's `run_exp.py` command plus `+infrastructure.slurm.exclude=r16-11`,
one after the other. Each clone is under
`/lustre/pd03/plgrid/plgllmefficont3/nano/plgmstefaniak/llmrandom_cemetery/`.

| Run | Job | Clone | W&B | Node | Submitted | Finished | State |
|---|---|---|---|---|---|---|---|
| `pkmflin_ame_h12`, LEM | 6295253 | `ame_init_pkmflin_ame_h12_2026-11-10_10-03-09` | | | 11-10 10:03 | 11-10 12:35 | CANCELLED while pending: moved to entropy |
| `pkmflin_dense_h12`, LEM | 6295256 | `ame_init_pkmflin_dense_h12_2026-11-10_10-03-56` | | | 11-10 10:04 | 11-10 12:58 | CANCELLED while pending: moved to entropy |
| `pkmflin_ame_h12`, entropy | 37649 | `ame_init_pkmflin_ame_h12_2026-11-10_12-24-57` | `x7tg7dpf` | h100a | 11-10 12:25 | 11-10 16:16 | COMPLETED, exit 0, 3 h 51 min |
| `pkmflin_dense_h12`, entropy | 37650 | `ame_init_pkmflin_dense_h12_2026-11-10_12-25-28` | `6k0wn5s6` | h100a | 11-10 12:25 | 11-10 19:38 | COMPLETED, exit 0, 3 h 22 min |

Moved to entropy at `0325e7dc` (*Decisions* 3); the entropy clones are under
`/home/mstefaniak/llmrandom_cemetery/`.

### Checks

- **Submit, 11-10 10:15:** both PENDING (Priority) behind our four other LEM jobs: the
  ame_init step-2 pair and the doc-mask pair. The worktree was clean after each
  `run_exp.py`.
- **LEM's Slurm controller went unreachable, 11-10 from about 11:09.**
  - Both jobs were PENDING at 11:07. Both watchers then exited with "left the
    queue": a watcher bug that read a failed `squeue` as every job gone.
  - Fixed: the watcher accepts only rc=0 (details in ame_init_embed_ablation.md,
    *Checks*).
  - **Back at 11:24, after about 15 min.** Both jobs survived, still PENDING
    (Priority). The watchers were re-armed with the fix at 11:25.
- **`pkmflin_ame_h12` on entropy, start, 11-10 12:33:** running on h100a since 12:25.
  - Step 0 loss 10.875, about ln V (10.83), against `kf4q4t4s`'s 11.25. The old head's
    larger logits start above ln V; the pkmf-scale head starts at it.
  - Step-0 grad norm 0.446, against 2.223.
  - s/step 3.348 (entropy's expected 3.36-3.38). No NaN and no spike; the grad-norm
    max after step 100 is 0.36.
  - **Early paired gap against `kf4q4t4s`, not a verdict:** +0.44 over steps 1-9,
    +0.35 over 10-49, +0.72 over 50-99 and +0.80 over 100-108. The step-500 check
    shows whether it closes.
  - The LEM copy, 6295253, was cancelled at 12:35. The dense LEM copy, 6295256, stays
    queued until 37650 starts on entropy.
- **Next check: the end of training (Maciej, 11-10 12:40):** "check them at the end
  of their training, there is no need for you to wake up more frequently".
  - The step-500 check is dropped. One watcher waits for the end of all three entropy
    AME jobs, `pkmflin_ame_h12` included, and another for dense (37650). Only a NaN
    ends either earlier.
  - So as not to wake at dense's start, its LEM copy 6295256 is cancelled now instead
    of when 37650 starts. LEM's login node hung at 12:45-12:48, so a detached retry
    cancels it once LEM answers. It checks that the job is still exactly
    `ame_init_pkmflin_dense_h12` and PENDING.
- **The dense LEM copy, 6295256, was cancelled at 12:58,** on the retry's second
  attempt, with its name and PENDING state as expected.
- **`pkmflin_ame_h12`, end, 11-10 16:31-16:45:** COMPLETED with exit 0 at 16:16. The
  slurm log has final_ce and no error; the clone has the step-3814 checkpoint (13 GB).
  - No spike. The grad-norm max after step 100 is 0.74, at step 119 (`kf4q4t4s`:
    0.45, at step 142).
  - 3.352 s/step; peak memory 70.6 GiB (`kf4q4t4s`: 70.4).
- **`pkmflin_dense_h12`, start, 11-10 16:35:** running on h100a since 16:16, when
  `pkmflin_ame_h12`'s GPU came free; W&B `6k0wn5s6`.
  - Step 0 loss 10.875, against `x0xlqroj`'s 11.19; grad norm 0.449, against 2.378.
  - 2.954 s/step (entropy's expected 2.95-2.97). No NaN and no spike; the grad-norm
    max after step 100 is 0.62.
  - **Early paired gap against `x0xlqroj`, not a verdict:** +0.37 over steps 0-99 and
    +0.74 over steps 200-329. `oldemb_dense_h12` stood at +0.79 at step 500 and ended
    at +0.27.
  - Next check: the end of training, about 19:40. One watcher covers it and
    ame_init_embed_ablation.md's n = 6 (ending about 20:45) and wakes when both have
    ended, or at once on a NaN or a job that leaves the queue before step 3,814.
- **`pkmflin_dense_h12`, end, 11-10 20:51-21:10:** COMPLETED with exit 0 at 19:38.
  The slurm log has final_ce and no error; the clone has the step-3814 checkpoint
  (2.1 GB).
  - No spike. The grad-norm max after step 100 is 0.62, at step 265 (`x0xlqroj`:
    2.58, in its spike at steps 194-203).
  - 2.958 s/step; peak memory 47.0 GiB (47.0).
  - **No early event** (*Decisions* 3). Its largest loss rise over steps 20-400,
    against the minimum of the 30 steps before, is +0.19, at step 50 (grad norm 1.49).

    | Run | Largest early rise | Grad-norm max, steps 20-400 |
    |---|---|---|
    | `pkmflin_dense_h12` | +0.19, step 50 | 1.49, step 50 |
    | D-B2, the clean rerun (`0w2whzba`) | +0.25, step 113 | 1.21, step 29 |
    | D-B3, early bump (`1uld8i4r`) | +0.69, step 83 | 2.75, step 82 |
    | D-B1, spike (`r86x2cjb`) | +3.66, step 354 | 3.64, step 349 |
    | `x0xlqroj`, the reference | +1.13, step 198 | 2.58, step 198 |

    The three D-B runs are same-seed reruns of `x0xlqroj`'s config on entropy, in
    W&B project `nano-ame-parity`.

### Results

| Run | final_ce | Reference | Δ final_ce | Paired gap, last 500 | Verdict |
|---|---:|---|---:|---:|---|
| `pkmflin_ame_h12` (`x7tg7dpf`) | 3.1270 | `kf4q4t4s` 3.0352 | +0.0918 | +0.0988 | **Much worse.** Per quarter: +0.438, +0.177, +0.116, +0.100 |
| `pkmflin_dense_h12` (`6k0wn5s6`) | 3.0732 | `x0xlqroj` 2.9824 | +0.0908 | +0.1013 | **Much worse.** Per quarter: +0.605, +0.209, +0.116, +0.102 |

### Analysis

**AME: pkmf-scale `Linear`s and head cost +0.099, given σ = 0.01 banks.** That is
the plan's "near +0.10" case. It confirms ame_init_embed_ablation.md's inference: the
new head and `Linear`s cost about +0.10, cancelling the banks' gain (−0.105 in paired
train loss).
- **The gap closes, then levels off:** +0.54 over the first 100 steps, +0.116 in the
  third quarter, +0.100 in the fourth and +0.099 over the last 500 steps.
- **The cost is not the embedding's.** Both runs decay the old embedding, which
  shrinks alike, from 23.6 to 10.9 (`kf4q4t4s`: 10.6).
- **Block 0's attention output against the embedding** is about twice
  `kf4q4t4s`'s all run:

  | Run | Step 100 | Step 500 | Step 1,000 | Step 2,000 | Step 3,800 |
  |---|---:|---:|---:|---:|---:|
  | `pkmflin_ame_h12` | 1.0× | 1.2× | 1.3× | 1.4× | 1.3× |
  | `kf4q4t4s` | 0.65× | 0.85× | 0.83× | 0.78× | 0.69× |

- **Caveat: one learning rate.** The pair runs at n = 8, as asked. In
  ame_init_embed_ablation.md's step 2, a model with the same pkmf-scale `Linear`s
  (Xavier head and banks, old embedding undecayed) did best at n = 7, by −0.040
  against its own n = 8. Part of this +0.099 may therefore be the learning rate, not
  the init.

**Dense: pkmf-scale `Linear`s and head cost +0.101, with the embedding decayed.**
That is less than half of ame_init_embed_ablation.md's +0.2325, which came with a
Xavier head (std 0.0063, against 0.0100 here) and an undecayed embedding.
- **The gap counts.** Same-seed reruns of `x0xlqroj`'s config on entropy pair with it,
  over the last 500 steps, at −0.015 for the clean one (D-B2). The two with early
  events are at +0.066 (D-B1) and +0.283 (D-B3).
  - This run has no early event (*Checks*).
  - The reference's own spike, at steps 194-203, left no lasting deficit: the clean
    rerun ties it.
- **Block 0's attention output against the embedding** starts high and falls back
  to the reference's by step 2,000. The embedding decays alike (10.97 at the end,
  against 10.86).

  | Run | Step 100 | Step 500 | Step 1,000 | Step 2,000 | Step 3,800 |
  |---|---:|---:|---:|---:|---:|
  | `pkmflin_dense_h12` | 3.5× | 0.90× | 0.34× | 0.21× | 0.17× |
  | `x0xlqroj` | 0.25× | 0.29× | 0.24× | 0.19× | 0.15× |

- The same one-learning-rate caveat applies: n = 8 only.

**Both models pay the same, about +0.10:** +0.101 dense, +0.099 AME. Their gaps also
close alike over the second half: +0.116 then +0.102 for dense, against +0.116 then
+0.100 for AME. At the references' n = 8, pkmf's √(0.1/fan_in) `Linear`s and head
(cut at ±2σ) are worse than the old 1/√fan_in in both models. Whether a retuned
learning rate wins it back is untested here.

### Decisions and issues

1. **Names: `pkmflin`**, for pkmf-scale `Linear`s, with the head counted among them.
   W&B names `ame_init_pkmflin_<model>_h12`; tag `ame_init_pkmflin`.
2. **The banks' σ = 0.01 is `kf4q4t4s`'s code path:** `bank_init_std: 0.01` draws
   `normal_(0, 0.01)` in both trees. The branch changed only the default (null, which
   means Xavier).
3. **Moved to entropy, 2026-11-10, Maciej's call:** "Move all pending jobs to entropy,
   there they should start sooner because entropy is less occupied". LEM had 237
   higher-priority jobs pending ahead of ours.
   - Entropy's FineWeb copy is LEM's
     ([ame_efficiency_parity_report.md](ame_efficiency_parity_report.md), item 6), and
     its node is also an H100. Only the cluster changes in the configs
     (`_cluster: entropy`, `gres: gpu:h100:1`). Composed, only cluster keys differ.
   - **AME's cross-cluster caveat is inside the tie floor.** Three same-seed runs of
     the production AME config on entropy ended at 3.1396, 3.1250 and 3.1328, against
     `doi36p8k`'s 3.1250 on LEM: a spread of 0.015.
   - **Dense comparisons are noisy on any cluster.** Three same-seed runs of the
     production dense config on entropy ended at 2.9688, 3.0439 and 3.2412, against
     `x0xlqroj`'s 2.9824. The two high ones had early events: a spike at step 354, and
     a bump at step 83 followed by slower training. A dense gap counts only if neither
     run has an early event.
   - Expected speed: 3.36-3.38 s/step for AME, 2.95-2.97 for dense (LEM: 3.758 and
     3.272).
   - The LEM jobs are cancelled once their entropy copies have started.
