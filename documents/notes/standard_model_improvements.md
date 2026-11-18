# What to fold into the standard AME and dense models

**Status:** proposed, 2026-11-18. This is a digest of every improvement and configuration
test run on AME and dense from 2026-11-04 to 2026-11-17. For each change it gives what the
tests found, whether to adopt it, and which branch and commit hold it. Nothing below is on
`ame_main` unless its *Where* says so. No code was run to write this note. Every number
comes from the report cited next to it, and the branch facts are as of 2026-11-18.

The work ran in five Claude Code sessions, kept in the "Overwatch" sessions folder:

| Session | What it did | Records |
|---|---|---|
| AME Experiments | Stages 1–2c: learning-rate grids for dense and AME, then ALiBi × KV shift for AME, at h12, h16 and h20 | [stage_runs.md](stage_runs.md) and the six `stage_*_report.md` |
| AME Experiments IMPLEMENT RUN | Four features on `ame_improve`, then two runs of the 4× bank learning rate | [ame_improve_runs_report.md](ame_improve_runs_report.md) |
| AME Implement Mask | Document masking for packed rows, run on both models | [doc_mask_report.md](doc_mask_report.md) |
| AME Experiments GATE | Four bank gates (stage 1, h12 and h16), then the bank-init ablation (stage 2) | [ame_bank_gates_report.md](ame_bank_gates_report.md), [ame_bank_gates_stage2_report.md](ame_bank_gates_stage2_report.md) |
| AME Experiment INIT | A new init bundle and its ablations, then the bank-init σ grid | [ame_init_exp.md](ame_init_exp.md), [ame_init_embed_ablation.md](ame_init_embed_ablation.md), [ame_init_pkmf_linears.md](ame_init_pkmf_linears.md), [ame_bank_sigma_grid.md](ame_bank_sigma_grid.md) |

## Summary

| Change | Model | Verdict | Where |
|---|---|---|---|
| ALiBi + KV shift (S2-PS) | AME | **Standard.** Already the layer's default | `ame_main` |
| RoPE | dense | **Standard.** NoPE costs +0.28 | `ame_main` |
| Learning rate: AME n = 8 at every size; dense n = 8 at h12, n = 9 at h16 and h20 | both | **Standard** | `ame_main` stage configs |
| Bank init σ = 1e-4 (`bank_init_std`) | AME | **Adopt.** The largest gain found: −0.09 against σ = 1 | Setting on `ame_main`, default still 1.0 |
| Document mask (`document_mask`) | both | **Adopt.** About −0.02 in both models at little cost; one replicate advised | `doc_mask` (local) |
| Banks at 4× the learning rate (`bank_lr_ratio`) | AME | **Adopt only if it stacks with small σ.** −0.03 at σ = 1; never run with small σ | `ame_improve` (pushed) |
| ST (stem) bank gate | AME | **Open.** Ties at h12, best at h16; the h16 baseline with small σ and no gate is missing | `ame_main`, off by default |
| QK-norm and KV shift for dense | dense | **Untested.** Built, never trained | `ame_improve` (pushed) |
| Bank gates B1, B2, B3 | AME | Reject | `ame_main`, off by default |
| New init bundle: Xavier embedding and head, pkmf-scale Linears, undecayed embedding | both | Reject | `ame_initialization` (local) |

## Proposed standard configs

What the tested pieces add up to. Two of the keys need branches that are not on `ame_main`
yet ([Before merging](#before-merging)).

AME:

```yaml
common:
  attention:
    type: ame
    document_mask: true      # branch doc_mask; reads common.eos_token_id (0, NeoX)
    ame:
      shift: true            # ame_main default; the ALiBi ladder and the sink are defaults too
      bank_init_std: 1.0e-4  # ame_main default is 1.0
      bank_gate: none        # default
trainer:
  learning_rate: 8           # LR 2^-8 at h12, h16 and h20
  bank_lr_ratio: 1           # branch ame_improve (default 4 there); 4 only if the stacking run wins
```

Dense:

```yaml
common:
  attention:
    type: gqa                # RoPE on by default
    document_mask: true      # branch doc_mask
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
| Dense with RoPE (S1-D, S1b-D, S1c-D) | n = 8: **2.9824** `x0xlqroj` | n = 9: **2.7441** `eh2sjsh7`, at the edge | n = 9: **2.5059** `1x82rzby` |
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
  - "At the edge" means the best n was the highest learning rate tried. AME h16 at n = 7
    and dense h16 at n = 8 were never run.
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

## Adopt

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
    - The lower edge lies between 1e-7 and 1e-5: 1e-7 gives +0.064, 1e-10 gives +0.442.
      The cause: when a token is first seen, its row's gradient scales as 1/‖row‖. At tiny
      σ that one gradient dominates the global norm, so the clip (1.0) shrinks every
      weight's step, for about 600 steps.
  - **Rare tokens.** They gain the most: 2.962 against 3.305 at u = 101–1000. Still, 86%
    of the total gain comes from tokens seen in every step.
- **Verdict:** adopt σ = 1e-4, mid-plateau and two decades below the cliff.
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
  - small σ without a gate at h16 or h20;
  - any learning rate but n = 8;
  - σ = 1e-8 (round 5, queued).

### Document mask (both models)

- **What it is:** `common.attention.document_mask: true`.
  - Attention stays inside each document of a packed row, in every layer, and AME's KV
    shift stops at document starts.
  - Documents are found from `common.eos_token_id` (0, NeoX; set in the branch's
    `configs/_model/h_balanced/base.yaml`).
  - On CUDA it is a flex score_mod; on CPU, a materialized mask.
  - Until now nothing masked across the documents in a row. That was a deliberate choice,
    equal for both methods.
- **Evidence** ([doc_mask_report.md](doc_mask_report.md)): h12, n = 8, same seed. The
  masked runs ran on entropy and the baselines on LEM; step 0 matched exactly.

  | | Masked | Unmasked | `final_ce` Δ | Paired gap | s/step |
  |---|---|---|---:|---:|---|
  | AME (S2-PS) | 3.1084 `j6wxx6h2` | 3.1250 `doi36p8k` | −0.017 | −0.016 | 3.746, unchanged |
  | Dense (S1-D) | 2.9639 `k3xx6w71` | 2.9824 `x0xlqroj` | −0.019 | −0.020 | 3.320 against 3.272: +1.5% |

  - Each gain is under its tie threshold. Both point the same way, and the paired gap
    stayed negative through training.
  - The AME − dense gap stays put: +0.143 without the mask, +0.145 with it.
- **Verdict:** adopt for both models. One same-seed replicate pair would confirm the
  ≈0.02.
- **Where:** the local branch `doc_mask` (not pushed), cut from `ame_main` `3ef6e363`.

  | Commit | What |
  |---|---|
  | `18175ed7` | The feature: new `src/core/document_mask.py`, plus edits to `src/core/ame.py`, `src/core/attention.py`, `src/core/model.py`, `src/definitions.py` and `configs/_model/h_balanced/base.yaml` |
  | `3974addd` | The fix that makes the compiled eval run: document ids become the custom op `nano::document_ids`. Without it, the no_grad graph fails to lower in torch 2.7.1 |
  | `56866314`, `da59f08d` | The run configs |
  | `87acc781` | `documents/scripts/doc_mask_results.py`: a run against its baseline from W&B |
- **Lesson:** the compiled eval runs a no_grad graph that grad-mode GPU tests never build.
  Test it at production shapes.

### Banks at 4× the learning rate (AME, conditional)

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

- **Verdict:** adopt only if it still helps on top of small σ.
  - Both changes pull the same lever. Under the hard QK-norm only a row's direction counts.
    A small σ lets Adam turn the rows faster, and so does a larger bank learning rate.
  - Small σ alone gains three times as much (−0.0895).
- **Where:**
  - The feature: `ame_improve`, commit `e64a466c`, pushed (`origin/ame_improve`).
  - The run configs: `configs/ame_improve/` on the local branch `ame_improve_runs`
    (`cb7704cd`).
- **Caveat:** the branch makes 4.0 the default, in the trainer schema and in
  `configs/_trainer/llama.yaml`.
  - Merged as-is, it moves every AME config to 4×.
  - Old AME checkpoints have two optimizer groups, so they need `bank_lr_ratio: 1` to
    resume.

## Open: the ST (stem) bank gate

- **What it is:** `common.attention.ame.bank_gate: stem`, on `ame_main` since `157c3122`
  (gates listed at [src/core/ame.py:77](../../src/core/ame.py#L77)).
- **At h12 it ties.** ST scores 3.019112 (`l54pyyqp`) against 3.035771 for no gate at
  σ = 0.01 (`kf4q4t4s`): −0.0167.
- **At h16 it leads.** ST scores 2.753039 (`ug8g2i2x`), against:
  - B3, 2.811172 (`3yc7haqu`);
  - B2, 2.818295 (`jwirkswo`);
  - the reference, 2.860982 (`hjw7znbz`: σ = 1, no gate).

  The lead over B3 is 0.058, three times the h12 floor used as a proxy. No floor has been
  measured at h16 ([ame_bank_gates_report.md](ame_bank_gates_report.md)).
- **Cost:** +4.75% FLOPs per token, +3.1% s/step.
- **What is missing:**
  - Every h16 arm started at σ = 0.01, and no run with σ = 0.01 and no gate exists at h16.
    So how much of ST's h16 lead goes beyond the init is unmeasured.
  - At h12, B3 added nothing beyond the init. If that holds at h16, B3's −0.050 is about
    the init's gain there, and ST adds about 0.058 on top.

## Built but never trained

All three are on `ame_improve`, pushed. Each passed the CPU suite and small end-to-end CPU
training runs. The two attention switches have never run on a GPU.

| Feature | Key | Commit | State |
|---|---|---|---|
| QK-norm for every attention type: ℓ2 per head, a learned per-head τ initialized at √d_head, τ exempt from weight decay | `common.attention.qk_norm`, or `gqa: {qk_norm: true}`. Off for dense, on for AME | `7762da98` | Never trained |
| KV shift for dense (Xu et al., arXiv 2411.19574), applied before RoPE and after QK-norm | `common.attention.shift`, or `gqa: {shift: true}` | `7345eebc` | Never trained |
| Training length in billions of tokens, rounded down to whole batches. Setting steps as well is an error before anything is submitted | `trainer.n_tokens_b`, with `n_steps: null` | `62a59d2f` | Used by the ratio-4 runs. Its `run_exp.py` fix `83ebaa85` (the missing-key check runs on the resolved copy) is only on the local `ame_improve_runs` |

The dense model matched to AME, `attention: {type: gqa, qk_norm: true, shift: true}`, has
never trained. So nobody knows whether QK-norm or the shift helps dense, or what they cost
in speed. The dense τ, α and β are also not instrumented.

## Tested and rejected

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
  halves. It is not applied on `ame_main`: no commit since `683c316b` touches the readout.
- Until it is fixed, runs that need readouts use `distributed: null` or DDP.

## Before merging

### Where everything is (2026-11-18)

| Branch | On origin | Commits ahead / behind `ame_main` | Holds |
|---|---|---|---|
| `ame_main` | No: 127 commits ahead of `origin/ame_main` (`e6116135`, 2026-11-07) | — | The stage configs and reports, the bank gates and `bank_init_std`, and every report cited here |
| `ame_improve` | Yes (`7345eebc`) | 4 / 179 | `n_tokens_b`, `qk_norm`, `bank_lr_ratio` and the dense `shift` |
| `ame_improve_runs` | No | 17 / 179 | `ame_improve`, plus the ratio-4 run configs `cb7704cd` and the `run_exp.py` fix `83ebaa85` |
| `doc_mask` | No | 17 / 43 | The document mask, its fix and configs, and `doc_mask_results.py` |
| `ame_initialization` | No | 13 / 86 | The rejected init bundle and its ablation configs, and `scripts/paired_gap.py` |
| `bank_sigma_grid` | No | 20 / 18 | The σ-grid configs and scripts: `grid_points.py`, `watch_array.py`, `paired_gap.py`, `bank_sigma_next.py`, `bank_sigma_chart.py` |
| `bank_gates_lem` | No | 0 / 53 | Nothing new: it is merged into `ame_main` |
| `stage12_lem`, `stage1b2b_lem`, `stage2_lem`, `stage2b_lem`, `s2d0_lem`, `stage1c2c_helios`, `stage1c2c_lem` | No | 0 ahead. `stage1c2c_lem`'s one commit has a twin on `ame_main`, `acb68ff5` | Snapshots the jobs were submitted from |

The counts include docs-only commits.

### Integration caveats

1. **Expect merge conflicts.** `ame_improve` and `doc_mask` both edit
   `src/core/attention.py`, `src/core/model.py` and `src/definitions.py`, and
   `ame_improve` is 179 commits behind `ame_main`.
2. **The dense KV shift would cross document boundaries.**
   - `doc_mask` zeroes the predecessor at document starts only inside AME's forward
     (`continues_document`, in `src/core/ame.py`).
   - `ame_improve`'s dense shift (`RoPEAttention` in `src/core/model.py`) calls `_shift1`
     directly.
   - So a merged dense model with both `shift` and `document_mask` on would still mix each
     document's first key and value with the previous document's EOS. It needs the same
     gating.
3. **`bank_lr_ratio` defaults to 4** ([Banks at 4× the learning rate](#banks-at-4-the-learning-rate-ame-conditional)).
   Merging it unchanged stacks 4× on top of small σ, and that combination has never run.

## Runs that would settle what is left

None of these has been submitted. Costs are on one LEM H100 per run.

| Run | Answers | Cost |
|---|---|---|
| AME h12, ratio 4 and σ = 1e-4 (or 0.01), against `kf4q4t4s` | Do the two bank levers stack? | One run, about 4.2 GPU-h |
| AME h16, σ = 0.01 and no gate | The init's own gain at h16, and how much of ST's h16 lead it explains | About 15.3 GPU-h |
| σ = 1e-8 (round 5 of the σ grid) | Where the lower edge lies | LEM job 6348965, queued 2026-11-17 20:28; no result in [ame_bank_sigma_grid.md](ame_bank_sigma_grid.md) yet |
| A same-seed replicate of each document-mask run | Is the ≈0.02 gain real? | Two h12 runs |
| Dense h16 at n = 8 | Dense h16's best learning rate, at the grid edge | About 13.4 GPU-h |

Lower priority:
- **ST at h16, two more runs** (about 31 GPU-h): they would measure h16's own run-to-run
  spread.
- **The σ cliff at n = 7 and n = 9:** does it move with the learning rate?
- **The pkmf Linears at n = 7:** could a retuned learning rate win back their +0.10? Even
  if it did, AME would still trail `kf4q4t4s` by about 0.06.
