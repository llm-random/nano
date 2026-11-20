# What to fold into the standard AME and dense models

**Status:** decided on 2026-11-21, except the document mask. This digest covers every
improvement and configuration test run on AME and dense from 2026-11-04 to 2026-11-20, the
settle runs included ([settle_runs_report.md](settle_runs_report.md)). For each change it
gives what the tests found, the verdict, and which branch and commit hold it. Nothing below
is on `ame_main` unless its *Where* says so. Every number comes from the report cited next
to it, and the branch facts are as of 2026-11-21.

- **AME:** the S2-PS recipe, plus the bank init σ = 1e-4 and the 4× bank learning rate. No
  bank gate.
- **Dense:** RoPE, plus the KV shift, and QK-norm for stability.
- **Open:** the document mask, about −0.02 in both models at +12% s/step.

The configs are in [Proposed standard configs](#proposed-standard-configs), and the reasoning
in [Conclusions](#conclusions-what-to-apply-to-ame-and-dense).

The work ran in six Claude Code sessions, kept in the "Overwatch" sessions folder:

| Session | What it did | Records |
|---|---|---|
| AME Experiments | Stages 1–2c: learning-rate grids for dense and AME, then ALiBi × KV shift for AME, at h12, h16 and h20 | [stage_runs.md](stage_runs.md) and the six `stage_*_report.md` |
| AME Experiments IMPLEMENT RUN | Four features on `ame_improve`, then two runs of the 4× bank learning rate | [ame_improve_runs_report.md](ame_improve_runs_report.md) |
| AME Implement Mask | Document masking for packed rows, run on both models | [doc_mask_report.md](doc_mask_report.md) |
| AME Experiments GATE | Four bank gates (stage 1, h12 and h16), then the bank-init ablation (stage 2) | [ame_bank_gates_report.md](ame_bank_gates_report.md), [ame_bank_gates_stage2_report.md](ame_bank_gates_stage2_report.md) |
| AME Experiment INIT | A new init bundle and its ablations, then the bank-init σ grid | [ame_init_exp.md](ame_init_exp.md), [ame_init_embed_ablation.md](ame_init_embed_ablation.md), [ame_init_pkmf_linears.md](ame_init_pkmf_linears.md), [ame_bank_sigma_grid.md](ame_bank_sigma_grid.md) |
| AME and Dense model improvements | This digest, then the settle runs and the decisions, 2026-11-18 to 11-21 | [settle_runs_report.md](settle_runs_report.md) |

## Summary

| Change | Model | Verdict | Where |
|---|---|---|---|
| ALiBi + KV shift (S2-PS) | AME | **Standard.** Already the layer's default | `ame_main` |
| RoPE | dense | **Standard.** NoPE costs +0.28 | `ame_main` |
| Learning rate: AME n = 8 at every size; dense n = 8 at h12, n = 9 at h16 and h20 | both | **Standard.** Dense h16's n = 9 is bracketed: n = 8 is unstable | `ame_main` stage configs |
| Bank init σ = 1e-4 (`bank_init_std`) | AME | **Adopted.** The largest gain found: −0.09 against σ = 1 at h12, −0.08 at h16 | Setting on `ame_main`, default still 1.0 |
| Banks at 4× the learning rate (`bank_lr_ratio`) | AME | **Adopted** (Maciej, 2026-11-18). −0.03 at σ = 1, but only −0.008 (a tie) on top of σ = 1e-4. No cost | `ame_improve` (pushed) |
| KV shift for dense (`shift`) | dense | **Adopted.** −0.022 alone and −0.040 on top of QK-norm, at +2.7% s/step | `ame_improve` (pushed) |
| QK-norm for dense (`qk_norm`) | dense | **Adopted for stability** (Maciej, 2026-11-20). A loss tie either way, at +1.8–2.4% s/step and 4 GiB | `ame_improve` (pushed) |
| Document mask (`document_mask`) | both | **Open, Maciej's call.** About −0.02 in both models, replicated, at +12% s/step. Recommendation: off in both until a block-sparse mask makes it free | `doc_mask` (local) |
| ST (STEM) bank gate | AME | **Rejected** (Maciej, 2026-11-21): it breaks AME's precomputed score table. A tie at h12; −0.048 beyond the small init at h16 | `ame_main`, off by default |
| Bank gates B1, B2, B3 | AME | **Rejected.** Ties or losses beyond the small init | `ame_main`, off by default |
| New init bundle: Xavier embedding and head, pkmf-scale Linears, undecayed embedding | both | **Rejected.** +0.014 (AME) to +1.271 (dense) at h12 | `ame_initialization` (local) |

## Proposed standard configs

As decided on 2026-11-21 ([Conclusions](#conclusions-what-to-apply-to-ame-and-dense)):
- AME adds the bank init σ = 1e-4 and the 4× bank learning rate to the S2-PS recipe, and
  no bank gate.
- Dense adds the KV shift, which carries the loss gain, and QK-norm, kept for stability.
- The document mask is the one open call, so it is commented out in both. If it goes on,
  it goes on in both models ([Open: the document mask](#open-the-document-mask-both-models)).

Three of the keys, `bank_lr_ratio`, `qk_norm` and `shift`, need `ame_improve`, which is not
on `ame_main` yet ([Before merging](#before-merging)).

AME:

```yaml
common:
  attention:
    type: ame
    # document_mask: true    # open call: -0.02 at +12% s/step; branch doc_mask
    ame:
      shift: true            # ame_main default; the ALiBi ladder and the sink are defaults too
      bank_init_std: 1.0e-4  # ame_main default is still 1.0
      bank_gate: none        # default; every gate is rejected
trainer:
  learning_rate: 8           # LR 2^-8 at h12, h16 and h20
  bank_lr_ratio: 4           # branch ame_improve (its default there); a tie on sigma 1e-4, no cost
```

Dense:

```yaml
common:
  attention:
    type: gqa                # RoPE on by default
    qk_norm: true            # branch ame_improve; for stability; a loss tie at +1.8-2.4% s/step
    shift: true              # branch ame_improve; -0.022 alone, -0.040 with qk_norm; +2.7% s/step
    # document_mask: true    # open call, as for AME; with the shift it needs caveat 2's gating
trainer:
  learning_rate: 8           # h12; 9 at h16 and h20
```

## How to read the numbers

- **Recipe.** Unless a row says otherwise, numbers are for h12 at 4B tokens. h16 runs
  8B tokens and h20 runs 16B. Every run uses a batch of 512 × 2,048 tokens, seed 123 and
  learning rate 2⁻ⁿ (written n). h12 and h16 run on one LEM H100, h20 on 4 GPUs. Lower
  loss is better.
- **`final_ce`** is the held-out loss logged at the end of training. It is rounded to bf16,
  so it is accurate to ±0.008 per run.
- **Paired gap** is the mean of (new − reference) training loss over the last 500 steps.
  Both runs use the same seed and batches, so it is finer than `final_ce`.
- **Tie.** For AME, a difference under 0.022 is a tie: 3 × the same-seed sd of 0.0072
  ([ame_comparison_h12.md](ame_comparison_h12.md),
  [ame_efficiency_parity_report.md](ame_efficiency_parity_report.md)). The bank-gates
  reports score with a common evaluator (`ame_ref32`) and treat ±0.019 as a tie. Dense
  varies 6–24× more than AME between same-seed runs, so one dense difference is weaker
  evidence.

## The standard recipe (Stages 1–2c)

Best run per arm and scale, `final_ce`, W&B project `ideas_cv/nano-ame`
([stage_runs.md](stage_runs.md)):

| Arm | h12, 4B tokens | h16, 8B tokens | h20, 16B tokens |
|---|---|---|---|
| Dense with RoPE (S1-D, S1b-D, S1c-D) | n = 8: **2.9824** `x0xlqroj` | n = 9: **2.7441** `eh2sjsh7`; n = 8 is unstable (3.4346) | n = 9: **2.5059** `1x82rzby` |
| AME with ALiBi and the KV shift (S2-PS, S2b-PS, S2c-PS) | n = 8: **3.1250** `doi36p8k` | n = 8: **2.8594** `hjw7znbz`, at the edge | n = 8: **2.5596** `tg4h4g66` |
| AME − dense, best against best | +0.143 | +0.115 | **+0.054** |

- **AME's 2 × 2 at h12:**
  - ALiBi alone: 3.2480.
  - The shift alone: 3.9346.
  - Neither: 4.2656, at n = 6, its own best.
  - ALiBi is worth about a nat with or without the shift. The shift adds 0.44–0.49 without
    positional encoding, but only 0.11–0.12 on top of ALiBi
    ([stage_2_report.md](stage_2_report.md)). h16 orders the cells the same way.
- **S2-PS is already the AME layer's default:** `shift=True`, `sink=True` and
  `slopes=None` (the ALiBi ladder), at
  [src/core/ame.py:333-339](../../src/core/ame.py#L333-L339).
- **Dense without RoPE** (S2-D0, `b93tal8j`, the gqa option `rope: false`) scored 3.3027
  at n = 9, against 3.0244 with RoPE: +0.28. Keep RoPE.
- **Learning rate:**
  - "At the edge" means the best n was the highest learning rate tried.
  - Dense h16 at n = 8 ran in the settle runs and was unstable: 102 spike steps, +0.69.
    So its n = 9 is bracketed ([settle_runs_report.md](settle_runs_report.md), 6).
  - AME h16 at n = 7 was never run.
  - On 2026-11-06 Maciej set n\* = 9 at h12 for stability
    ([stage_1_report.md](stage_1_report.md), decision 9). n = 8 measured best in both
    models, though, and every later h12 test ran at n = 8.
- **Configs, all on `ame_main`:**
  - AME: [stage_2/ame_h12_pe_shift.yaml](../../configs/stage_2/ame_h12_pe_shift.yaml),
    [stage_2b/ame_h16_pe_shift.yaml](../../configs/stage_2b/ame_h16_pe_shift.yaml),
    [stage_2c/ame_h20_pe_shift.yaml](../../configs/stage_2c/ame_h20_pe_shift.yaml).
  - Dense: [stage_1/dense_h12_lr_grid.yaml](../../configs/stage_1/dense_h12_lr_grid.yaml),
    [stage_1b/dense_h16_lr_grid.yaml](../../configs/stage_1b/dense_h16_lr_grid.yaml),
    [stage_1c/dense_h20_lr_grid.yaml](../../configs/stage_1c/dense_h20_lr_grid.yaml).
  - A run's logged W&B config is the truth about what it trained.

## Adopted

### Bank init σ (AME)

- **What it is:** `common.attention.ame.bank_init_std`, the std of the N(0, σ²) draw for
  the Q/K banks. The draw is not truncated
  ([src/core/ame.py:393](../../src/core/ame.py#L393)), and the default is 1.0
  ([src/core/ame.py:330](../../src/core/ame.py#L330)).
- **Evidence:**
  - **Bank-gates stage 2.** σ = 0.01 with no gate (`kf4q4t4s`) scores 3.035771 on the
    common evaluator, against 3.125227 for production σ = 1 (`doi36p8k`). That is −0.0895,
    resolved, and it is all of stage 1's gated gain at h12
    ([ame_bank_gates_stage2_report.md](ame_bank_gates_stage2_report.md)).
  - **σ grid.** 14 runs at n = 8, each compared by paired gap with σ = 0.01
    (`bank_sigma_m200`, `plabns2v`) ([ame_bank_sigma_grid.md](ame_bank_sigma_grid.md)):
    - Every σ from 1e-5 to 0.0178 ties, at −0.0111 to +0.0089. The measured best is
      3.16e-5 (−0.0111), and 1e-4 gives −0.0050.
    - The upper edge is a cliff between 0.0178 and 0.0316: 0.0316 gives +0.070, 0.1 gives
      +0.080 and σ = 1 gives +0.097. Both runs above the cliff had large loss spikes.
    - The lower edge lies between 1e-7 and 1e-5: 1e-7 gives +0.064, 1e-8 +0.183 and
      1e-10 +0.442.
      The cause: when a token is first seen, its row's gradient scales as 1/‖row‖. At tiny
      σ that one gradient dominates the global norm, so the clip (1.0) shrinks every
      weight's step, for about 600 steps.
  - **Rare tokens.** They gain the most: 2.962 against 3.305 at u = 101–1000. Still, 86%
    of the total gain comes from tokens seen in every step.
  - **h16 (settle run 2, 2026-11-20).** σ = 0.01 with no gate (`eeaz5uy1`) against σ = 1
    (`hjw7znbz`): paired −0.080, `final_ce` 2.7959 against 2.8594.
- **Verdict:** adopted, σ = 1e-4: mid-plateau and two decades below the cliff.
  - σ = 0.01 ties at h12 and n = 8, and more runs used it.
  - However, 0.01 sits only a quarter-decade below the cliff, and the cliff may move with
    the learning rate (untested).
- **Where:**
  - The setting has been on `ame_main` since the first AME commit, so adopting it only
    changes a default.
  - The grid's configs are on the local branch `bank_sigma_grid`
    (`configs/bank_sigma/round1.yaml` to `round5.yaml`).
  - Its chart and data are in [ame_bank_sigma_figures/](ame_bank_sigma_figures/).
- **Not measured:**
  - small σ at h20;
  - any learning rate but n = 8.

### Banks at 4× the learning rate (AME)

- **What it is:** `trainer.bank_lr_ratio` sets the Q/K banks' learning rate as a multiple
  of the base rate.
  - It holds through warmup and the cosine decay, and under memory parallelism.
  - Each optimizer group gets its own cosine floor in `src/core/schedulers.py`. At ratio 1
    the schedule is bit-identical to the old one.
- **Evidence** ([ame_improve_runs_report.md](ame_improve_runs_report.md)): S2-PS at n = 8
  and σ = 1. No cost in s/step, MFU or memory, and no NaN or spike. Only ratio 4 was tried.

  | | Ratio 4 | Ratio 1 | `final_ce` Δ | Paired gap |
  |---|---|---|---:|---:|
  | h12 | 3.0938 `y5ogvt3k` | 3.1250 `doi36p8k` | −0.031 | −0.034 |
  | h16 | 2.8311 `hn1qal9z` | 2.8594 `hjw7znbz` | −0.028 | −0.033 |
  | h12 on σ = 1e-4 (settle run 1) | 3.0312 `zdmv3qci` | 3.0400 `7te9jv9u` | −0.009 | −0.008 |

- **Verdict:** adopted, by Maciej's call (2026-11-18). On top of σ = 1e-4 it is a tie:
  −0.008, against −0.034 at σ = 1 ([settle_runs_report.md](settle_runs_report.md), 1).
  - **The two levers mostly overlap.** Under the hard QK-norm only a row's direction
    counts. A small σ lets Adam turn the rows faster, and so does a larger bank learning
    rate.
  - **It costs nothing:** the same s/step and memory, no spike. Keeping it is harmless.
- **Where:**
  - The feature: `ame_improve`, commit `e64a466c`, pushed (`origin/ame_improve`).
  - The run configs: `configs/ame_improve/` on the local branch `ame_improve_runs`
    (`cb7704cd`).
- **Caveat:** the branch makes 4.0 the default, in the trainer schema and in
  `configs/_trainer/llama.yaml`.
  - Merged as-is, it moves every AME config to 4×.
  - Old AME checkpoints have two optimizer groups, so they need `bank_lr_ratio: 1` to
    resume.

### QK-norm and the KV shift (dense)

- **What they are:** two switches on `ame_improve` that give dense attention what AME
  already has. Both passed the CPU suite and small end-to-end CPU training runs, and first
  ran on a GPU in the settle runs.
  - `common.attention.qk_norm`, or `gqa: {qk_norm: true}` (`7762da98`): ℓ2 per head, with a
    learned per-head τ initialized at √d_head and exempt from weight decay. Off for dense
    by default, on for AME.
  - `common.attention.shift`, or `gqa: {shift: true}` (`7345eebc`): the KV shift of Xu et
    al. (arXiv 2411.19574), after QK-norm and before RoPE. One (c1, c2) pair per KV head
    for the keys and one for the values.
- **Evidence:** the full 2 × 2 at h12 ([settle_runs_report.md](settle_runs_report.md), 5).
  Each corner is a paired gap against `x0xlqroj`, with its `final_ce`:

  | | No shift | Shift |
  |---|---:|---:|
  | **No QK-norm** | `x0xlqroj`: 0 (2.9824) | `aak55dd1`: −0.022 (2.9629) |
  | **QK-norm** | `8lqp6pk3`: +0.013 (2.9941) | `chne4w91`: −0.027 (2.9561) |

  - **The shift carries the gain:** −0.022 alone, on the tie line, and −0.040 on top of
    QK-norm. Neither run with the shift had a loss spike, while both runs without it had an
    early episode.
  - **QK-norm adds nothing measurable to the loss:** +0.013 alone and −0.006 on top of the
    shift, both ties. In both it is about 0.06 behind in the first quarter, then level.
  - **Cost:** the shift +2.7% s/step and 2.3 GiB; QK-norm +1.8% alone, +2.4% on top of the
    shift, and 4.0 GiB; both together +5.3% and 6.3 GiB.
- **Verdict:** adopt both. Maciej put them into the standard dense config on 2026-11-18,
  before these runs.
  - The runs back the shift.
  - Maciej keeps QK-norm for stability (2026-11-20). Its case lies at larger models and
    higher learning rates, which h12 at n = 8 does not test. Dense h16 at n = 8, unstable
    without it, would test it in one run.
- **Not measured:** either switch at h16 or h20. The dense τ, α and β are not instrumented.

## Open: the document mask (both models)

- **What it is:** `common.attention.document_mask: true`.
  - Attention stays inside each document of a packed row, in every layer, and AME's KV
    shift stops at document starts.
  - Documents are found from `common.eos_token_id` (0, NeoX; set in the branch's
    `configs/_model/h_balanced/base.yaml`).
  - On CUDA it is a flex score_mod; on CPU, a materialized mask.
  - Until now nothing masked across the documents in a row. That was a deliberate choice,
    equal for both methods.
- **Evidence:** h12, n = 8, same seed. Sources: [doc_mask_report.md](doc_mask_report.md)
  for the first runs, [settle_runs_report.md](settle_runs_report.md) for the replicates.
  - The first masked runs ran on entropy, against LEM baselines.
  - The settle runs repeated both on LEM on 2026-11-19.
  - Step 0 matched in all of them.

  | | Masked | Unmasked | `final_ce` Δ | Paired gap |
  |---|---|---|---:|---:|
  | AME (S2-PS), entropy | 3.1084 `j6wxx6h2` | 3.1250 `doi36p8k` | −0.017 | −0.016 |
  | AME, LEM replicate | 3.1084 `h53srjtk` | 3.1250 `doi36p8k` | −0.017 | −0.018 |
  | Dense (S1-D), entropy | 2.9639 `k3xx6w71` | 2.9824 `x0xlqroj` | −0.019 | −0.020 |
  | Dense, LEM replicate | 2.9629 `igfkre78` | 2.9824 `x0xlqroj` | −0.020 | −0.022 |

  - **Each replicate reproduces its original within 0.002.**
  - **For AME the gain is resolved.** The parity study holds three more unmasked runs of
    this config, on entropy. Both masked runs lie below all four unmasked runs, in 8 of 8
    pairs, from −0.016 to −0.033, mean −0.022.
  - **Dense has a single unmasked run,** `x0xlqroj`. It had an early spike episode that
    the masked runs lack.
  - **The AME − dense gap stays put:** +0.143 without the mask, +0.145 with it.
- **Cost: about +12% s/step in both models,** measured within one cluster:
  - AME 4.200 against 3.746 on LEM;
  - dense 3.681 against 3.272 on LEM;
  - AME on entropy, 3.746 against 3.36–3.38.

  The first report's "unchanged" and "+1.5%" compared entropy with LEM, and entropy runs
  this workload about 10% faster.
- **Verdict: open, Maciej's call.** The gain is real in both models, about −0.02, and it
  costs about 12% more time per step.
  - **Recommendation: off in both models for now, on in both once it is free.** It leaves
    the AME − dense gap where it was, so the comparison does not need it.
  - The mask is a flex `score_mod`, so it skips no work. A `BlockMask` built from the
    document boundaries could skip the cross-document blocks, and might make the mask
    free. Untested.
  - Whenever it goes on, it goes on in both models.
- **Where:** the local branch `doc_mask` (not pushed), cut from `ame_main` `3ef6e363`.

  | Commit | What |
  |---|---|
  | `18175ed7` | The feature: new `src/core/document_mask.py`, plus edits to `src/core/ame.py`, `src/core/attention.py`, `src/core/model.py`, `src/definitions.py` and `configs/_model/h_balanced/base.yaml` |
  | `3974addd` | The fix that makes the compiled eval run: document ids become the custom op `nano::document_ids`. Without it, the no_grad graph fails to lower in torch 2.7.1 |
  | `56866314`, `da59f08d` | The run configs |
  | `87acc781` | `documents/scripts/doc_mask_results.py`: a run against its baseline from W&B |
- **Lesson:** the compiled eval runs a no_grad graph that grad-mode GPU tests never build.
  Test it at production shapes.

## Tested and rejected

### The ST (STEM) bank gate

**Rejected by Maciej on 2026-11-21:** it breaks the precomputed score table, AME's
decode-time payoff, so its gain at h16 does not count. The record below stays.

- **What it is:** `common.attention.ame.bank_gate: stem`, on `ame_main` since `157c3122`
  (gates listed at [src/core/ame.py:77](../../src/core/ame.py#L77)).
  - ST stands for the STEM gate, after STEM ([Sadhukhan et al., ICLR 2026](https://arxiv.org/abs/2601.10639)),
    which puts a token-indexed table under a SwiGLU gate.
  - Here it gates the query only: `q_i = silu(W_g x_i) ⊙ norm(E_Q[t_i])`, one weight per
    channel, read off the query's hidden state. `W_g` is a bias-free dmodel → H·d_head
    Linear, initialized like the SwiGLU gate and weight-decayed. Keys stay pure bank rows.
- **At h12 it ties.** ST scores 3.019112 (`l54pyyqp`) against 3.035771 for no gate at
  σ = 0.01 (`kf4q4t4s`): −0.0167.
- **At h16 it leads.** ST scores 2.753039 (`ug8g2i2x`), against:
  - B3, 2.811172 (`3yc7haqu`);
  - B2, 2.818295 (`jwirkswo`);
  - the reference, 2.860982 (`hjw7znbz`: σ = 1, no gate).

  The lead over B3 is 0.058, three times the h12 floor used as a proxy. No floor has been
  measured at h16 ([ame_bank_gates_report.md](ame_bank_gates_report.md)).
- **Cost:** +4.75% FLOPs per token, and +3.1–3.8% s/step.
- **It also gives up the precomputable score table,** AME's decode-time payoff. With the
  gate, a score depends on the hidden state, and a per-channel weight cannot be pulled
  out of the dot product. B1–B3 keep the table
  ([ame_bank_gates_plan.md](ame_bank_gates_plan.md), mechanism table).
- **Beyond the init, at h16** (settle run 2, 2026-11-20,
  [settle_runs_report.md](settle_runs_report.md)). The h16 run with σ = 0.01 and no gate,
  `eeaz5uy1`, scored `final_ce` 2.7959.
  - The init's own gain is −0.080 (paired, against `hjw7znbz`).
  - ST adds −0.048 on top (paired; `final_ce` −0.044), twice the h12 tie band.
  - B3 adds +0.017, nothing.
  - So at h16, ST is the one gate that helps beyond the init. At h12 its gain beyond the
    init was a tie.

### The other gates, the init bundle and the ablations

Keep the current init: the embedding N(0, 1) cut at ±2 with weight decay, and the Linears
and head at 1/√fan_in cut at ±2σ. The one exception is the banks' σ.

| Change | Result | Code |
|---|---|---|
| B1, the confidence gate | 0.07–0.11 behind no gate at the same σ, from either start. Its trust stays where it starts | `ame_main`, `bank_gate: confidence` |
| B2 (context gate) and B3 (backoff gate) | Nothing beyond the σ = 0.01 init at h12: B2 −0.0003, B3 +0.0001. B3 at σ = 1: −0.0079. All ties | `ame_main`, `bank_gate: context` and `backoff` |
| The new init bundle: Xavier embedding, head and banks; pkmf's √(0.1/fan_in) Linears; no weight decay on the embedding | Paired gaps: AME h12 +0.014, dense h12 +1.271, AME h16 +0.032, dense h16 +0.213 ([ame_init_exp.md](ame_init_exp.md)) | `ame_initialization` `79108114` |
| Part of that bundle: the Xavier embedding (σ ≈ 0.006) | Dense +0.999, AME +0.003 ([ame_init_embed_ablation.md](ame_init_embed_ablation.md)) | `ame_initialization` |
| Part of that bundle: no weight decay on the embedding | AME +0.012, dense +0.039. It never helps | `ame_initialization` |
| pkmf's √(0.1/fan_in) for the Linears and the head | AME +0.099 against `kf4q4t4s`, dense +0.101 against `x0xlqroj` ([ame_init_pkmf_linears.md](ame_init_pkmf_linears.md)) | `ame_initialization` `4457ba17`, which also adds `model.embedding.weight_decay` |
| The new init with the old embedding, AME at its best n | n = 7 beats `doi36p8k` by 0.029, but scores 3.1084 against `kf4q4t4s`'s 3.0352 | `ame_initialization` |
| Dense without RoPE | +0.28 | `ame_main`, `gqa: {rope: false}` (`c4cb1ef6`) |
| AME without ALiBi or without the shift | +0.12 (ALiBi only) to +1.14 (neither) | `ame_main` |

The new init's only real gain was the small bank σ. Xavier banks start at σ ≈ 0.006, close
to 0.01.

## Known defect to fix in standard AME

**Under FSDP2, AME's readouts are never logged.**
- The α/β/τ readouts, the horizons and the attention entropy are all lost, at any GPU
  count.
- `AMEAttention.instrumentation(token_ids)` runs `attention_diagnostics()` outside FSDP2's
  hooks and fails on the sharded parameters. The trainer then turns instrumentation off for
  the whole run.
- In W&B, 0 of 38 finished FSDP2 AME runs logged `alpha_2` or `tau`, and the h20 stage
  ran on FSDP2 ([stage_2c_report.md](stage_2c_report.md), "Readouts").
- The fix is small: `instrumentation(token_ids=None)` works under FSDP2, so split the two
  halves. It is not applied on `ame_main` as of 2026-11-21: no commit since `683c316b`
  touches the readout.
- Until it is fixed, runs that need readouts use `distributed: null` or DDP.

## Before merging

### Where everything is (2026-11-21)

| Branch | On origin | Commits ahead / behind `ame_main` | Holds |
|---|---|---|---|
| `ame_main` | No: 152 commits ahead of `origin/ame_main` (`e6116135`, 2026-11-07) | — | The stage configs and reports, the bank gates and `bank_init_std`, and every report cited here |
| `ame_improve` | Yes (`7345eebc`) | 4 / 204 | `qk_norm` (`7762da98`), the dense `shift` (`7345eebc`), `bank_lr_ratio` (`e64a466c`), and `trainer.n_tokens_b` (`62a59d2f`), the training length in billions of tokens |
| `ame_improve_runs` | No | 17 / 204 | `ame_improve`, plus the ratio-4 run configs `cb7704cd` and the `run_exp.py` fix `83ebaa85` (the missing-key check runs on the resolved copy) |
| `doc_mask` | No | 17 / 68 | The document mask, its fix and configs, and `doc_mask_results.py` |
| `ame_initialization` | No | 13 / 111 | The rejected init bundle and its ablation configs, and `scripts/paired_gap.py` |
| `bank_sigma_grid` | No | 24 / 43 | The σ-grid configs and scripts: `grid_points.py`, `watch_array.py`, `paired_gap.py`, `bank_sigma_next.py`, `bank_sigma_chart.py` |
| `settle_runs_improve`, `settle_runs_main`, `settle_runs_doc_mask` | No | 19 / 204, 1 / 20, 18 / 68 | The settle runs' configs, each on top of the branch it ran from: `ame_improve_runs`, `ame_main` and `doc_mask` |
| `bank_gates_lem` | No | 0 / 78 | Nothing new: it is merged into `ame_main` |
| `stage12_lem`, `stage1b2b_lem`, `stage2_lem`, `stage2b_lem`, `s2d0_lem`, `stage1c2c_helios`, `stage1c2c_lem` | No | 0 ahead. `stage1c2c_lem`'s one commit has a twin on `ame_main`, `acb68ff5` | Snapshots the jobs were submitted from |

The counts include docs-only commits.

### Integration caveats

1. **Expect merge conflicts.** `ame_improve` and `doc_mask` both edit
   `src/core/attention.py`, `src/core/model.py` and `src/definitions.py`, and
   `ame_improve` is 204 commits behind `ame_main`.
2. **The dense KV shift would cross document boundaries, if the mask goes on.**
   - `doc_mask` zeroes the predecessor at document starts only inside AME's forward
     (`continues_document`, in `src/core/ame.py`).
   - `ame_improve`'s dense shift (`RoPEAttention` in `src/core/model.py`) calls `_shift1`
     directly.
   - The standard dense config has the shift on and the mask off, so this matters only if
     the mask goes on. The merge must then add the same gating, or each document's first
     key and value mix with the previous document's EOS.
3. **`bank_lr_ratio` defaults to 4** ([Banks at 4× the learning rate](#banks-at-4-the-learning-rate-ame)).
   Merged as-is, it moves every AME config to 4×. The standard config sets 4 explicitly,
   on small σ, where it ran once: a tie, −0.008. Old AME checkpoints need
   `bank_lr_ratio: 1` to resume.

## The settle runs

They ran on LEM on 2026-11-19, the two single-switch dense runs on 2026-11-20. The record is
[settle_runs_report.md](settle_runs_report.md).

| Run | Question | Answer |
|---|---|---|
| AME h12, ratio 4 on σ = 1e-4 | Do the two bank levers stack? | Barely: −0.008 against ratio 1, a tie |
| AME h16, σ = 0.01, no gate | The init's own gain at h16; how much of ST's lead it explains | The init gains −0.080. ST adds −0.048 beyond it, B3 nothing |
| σ = 1e-8, round 5 of the σ grid | Where the lower edge lies | +0.183 ([ame_bank_sigma_grid.md](ame_bank_sigma_grid.md)) |
| A same-seed replicate of each document-mask run | Is the ≈0.02 gain real? | Yes: each reproduced within 0.002, and the AME gain is resolved. But the mask costs +12% s/step |
| Dense h12 with QK-norm and the shift, then each alone | Do the dense switches help, and what do they cost? | Together −0.027 at +5% s/step. The shift carries it (−0.022 alone, +2.7%); QK-norm adds a tie |
| Dense h16 at n = 8 | Is dense h16's best n at the grid edge? | No: n = 8 is unstable (+0.69), so n = 9 is bracketed |

## Conclusions: what to apply to AME and dense

Drawn on 2026-11-20, once the settle runs were in, and updated with Maciej's decisions on
2026-11-21. Every change was measured alone or in pairs, at h12 (some also at h16), with one
seed. No run has trained a full standard config yet.

### AME

| Change | Effect at h12 | Cost | Conclusion |
|---|---|---|---|
| Bank init σ = 1e-4 | −0.09 against σ = 1; −0.08 at h16 | None | **Apply.** The default is still 1.0, so the standard configs must set it |
| Banks at 4× the learning rate | −0.008 on σ = 1e-4, a tie (−0.034 at σ = 1) | None | **Apply** (Maciej, 2026-11-18). It is the default on `ame_improve`, and old checkpoints need `bank_lr_ratio: 1` to resume |
| ALiBi, KV shift and sink (S2-PS), n = 8 | The recipe | — | **Keep.** Already the defaults |
| Document mask | −0.016 to −0.018, resolved | +12% s/step | **Maciej's call.** Recommendation: off for now (below) |
| Bank gates B1–B3; the new init bundle | Ties or losses | — | **Don't apply** |
| ST (STEM) bank gate | A tie; −0.048 beyond the init at h16, one run | +3.1–3.8% s/step | **Don't apply** (Maciej, 2026-11-21): it breaks the precomputed score table |

### Dense

| Change | Effect at h12 | Cost | Conclusion |
|---|---|---|---|
| KV shift | −0.022 alone, −0.040 on top of QK-norm; neither run spiked | +2.7% s/step, 2.3 GiB | **Apply.** It must stop at document starts before it meets the document mask (caveat 2) |
| QK-norm | +0.013 alone, −0.006 on the shift: both ties | +1.8–2.4% s/step, 4 GiB | **Apply,** for stability (Maciej, 2026-11-20). At h12 the loss neither gains nor loses |
| RoPE; n = 8 at h12, n = 9 at h16 and h20 | The recipe. NoPE costs +0.28; n = 8 at h16 is unstable | — | **Keep** |
| Document mask | −0.020 to −0.022, one unmasked baseline | +12% s/step | **Maciej's call**, as for AME |
| The new init bundle and its parts | +0.04 to +1.27 | — | **Don't apply** |

### The open call: the document mask

**Recommendation: off in both models for now, on in both once it is free.**
- The gain is real, about −0.02 in each model. But it leaves the AME − dense gap where it
  was (+0.143 without, +0.145 with), so the comparison does not need it, and it costs 12%
  of every run.
- A `BlockMask` that skips the cross-document blocks may remove that cost. Whenever the
  mask goes on, it goes on in both models.

### Decided by Maciej

- **Dense keeps QK-norm, for stability** (2026-11-20).
  - At h12 it changes the loss by a tie either way, at +1.8–2.4% s/step and 4 GiB.
  - Its stability case lies at larger models and higher learning rates, which h12 at
    n = 8 does not test. Dense h16 at n = 8 was unstable without it (102 spike steps).
    Rerun with QK-norm, it would test that case in one run, about 13 GPU-h.
- **AME gets no ST gate** (2026-11-21). It breaks the precomputed score table, so its h16
  gain does not count, and the two h16 replicates are off.
  - The reason holds for any change to AME: a score must stay a function of token ids and
    distance, so the table can be precomputed.
  - Every bank gate is now rejected, and AME keeps `bank_gate: none`.

### What the changes do to the AME − dense gap

`final_ce`, single runs, all without the document mask:

| Runs | AME | Dense | AME − dense |
|---|---|---|---:|
| h12, before | 3.1250 `doi36p8k` | 2.9824 `x0xlqroj` | +0.143 |
| h12: AME with σ = 1e-4 and the 4× rate; dense with QK-norm and the shift | 3.0312 `zdmv3qci` | 2.9561 `chne4w91` | **+0.075** |
| h16, before | 2.8594 `hjw7znbz` | 2.7441 `eh2sjsh7` | +0.115 |
| h16: AME with σ = 0.01 only | 2.7959 `eeaz5uy1` | 2.7441 `eh2sjsh7` | **+0.052** |

- At h12 the changes halve AME's deficit. Nearly all of AME's part is the bank init: −0.085
  of its −0.094 (`7te9jv9u`, σ = 1e-4 at ratio 1, scored 3.0400).
- At h16 the AME init alone more than halves it. Dense h16 with QK-norm and the shift has
  not run.
- h20 (+0.054 before) has run none of the changes.

### Before they become the standard

1. **One confirming run per model at h12:** the full standard config, paired against its
   current baseline. No run has stacked every change, and two of them already overlap
   (σ and the 4× rate). About 4 GPU-h each.
2. **The merge** ([Before merging](#before-merging)):
   - gate the dense shift at document starts, if the document mask goes on (caveat 2);
   - settle `bank_lr_ratio`'s default (caveat 3);
   - expect conflicts between `ame_improve` and `doc_mask`, whenever the mask merges
     (caveat 1).
3. **Fix the FSDP2 readout defect** ([Known defect](#known-defect-to-fix-in-standard-ame)),
   so the h20 runs, which use FSDP2, log AME's readouts.
4. **Still not measured:** any change at h20; σ at any learning rate but n = 8; AME h16 at
   n = 7 (its best n is at the edge of the grid); dense h16 with QK-norm and the shift.

### Lower priority

- **A block-sparse document mask:** a `BlockMask` that skips cross-document blocks could
  remove the mask's 12% cost.
- **The σ cliff at n = 7 and n = 9:** does it move with the learning rate?
- **The pkmf Linears at n = 7:** could a retuned learning rate win back their +0.10? Even
  if it did, AME would still trail `kf4q4t4s` by about 0.06.
