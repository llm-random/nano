# Review upgrades: AME analyses and ablations for later

**Status (2026-11-25): proposals only. Nothing is scheduled, and every run here needs Maciej's
OK before it costs GPU time.** The fixes to apply before the reviewed commits settle on
`ame_main` are in [review_issues.md](review_issues.md).

- **Sources:**
  - the review of the nine older unpushed `ame_main` commits: [rev1.md](rev1.md) and
    [rev2.md](rev2.md), each with Maciej's response, plus rev2's reply;
  - the checks run during that review;
  - the standard-models task ([standard_models_report.md](standard_models_report.md)).
- **"Proposed by"** names who raised each item. "This session" means the agent that checked
  the reviews against the code.

| # | Topic | The open question | Proposed by | What it needs |
|---|---|---|---|---|
| [U1](#u1-bank-rows-under-dense-adamw) | Bank rows under dense AdamW | How far does a rare token's bank row really move, and would a lazy or lower-β₂ bank optimizer serve rare tokens better? | Finding: rev1 #3; numbers: rev2's reply; ablations: this session | A readout (U4c), then h12 runs |
| [U2](#u2-b1s-analysis-under-u1) | B1's analysis | Does U1 change how B1's results read? | rev2's reply | Analysis only, if the gates are revisited |
| [U3](#u3-bank-init-σ) | Bank init σ | Does the σ plateau move with the learning rate? | standard_model_improvements.md, "Later" | Runs (the default became 1e-4 under the QK-norm on 2026-11-25) |
| [U4](#u4-readouts-for-future-analysis) | Readouts | Five readouts that future analysis needs | rev1 #4, rev2 #9 and #15, the standard-models report, this session | Code |
| [U5](#u5-the-common-evaluator) | The common evaluator | Score FSDP2 runs (h20), configs with a null learning rate, and `n_tokens_b` configs | rev2 #8 | Code |
| [U6](#u6-stability-of-the-ame-h12-recipe) | Recipe stability | How often does the AME h12 recipe spike, and what drives it? | This session (standard-models task) | Seeds |
| [U7](#u7-reproducing-a-pre-merge-run) | Reproducing pre-merge runs | Rules to follow, not experiments | rev2's reply; the response | Nothing |
| [U8](#u8-the-st-gate-if-it-is-ever-revived) | The ST gate, if revived | Distinct per-layer init under a GPU default device | rev1 #2, rev2 #10 and reply | Code, only if needed |
| [U9](#u9-runs-already-planned) | Runs already planned | Four runs from the standard-models plan | standard_model_improvements.md | Maciej's OK |

## U1. Bank rows under dense AdamW

**The finding (checked against the code, 2026-11-25).**
- The banks are stepped by a dense `torch.optim.AdamW`:
  - with the memory-parallel controller, its own optimizer
    ([memory_parallel.py:490](../../src/core/memory_parallel.py#L490), built at
    [main.py:343](../../main.py#L343));
  - without it, the backbone's, where the banks get a group of their own at
    `bank_lr_ratio` × lr ([model.py:753](../../src/core/model.py#L753)).
- Nothing in the repo sets `betas`, so both use the defaults (0.9, 0.999). The banks have no
  weight decay.
- A bank row gets a gradient only in the steps where its token occurs. Adam's moment estimates
  keep moving it in the steps after.
- One occurrence of a token whose row has no history (rev2's numbers, re-derived here):

  | Occurrence at step | Moves at once | Moves in total | 90% of it within |
  |---|---|---|---|
  | 1,000 | 2.5 lr | 25 lr | 23 steps |
  | 3,000 | 3.1 lr | 31 lr | — |

  - lr is the banks' own learning rate: `bank_lr_ratio` × the base learning rate.
  - The numbers assume the gradient is large against Adam's ε = 1e-8. A tinier gradient moves
    the row less.
- **So:** a token present in every step moves its row about 1 lr per step, while a rare token
  moves it many lr per occurrence. `steps_seen` (`trainer.count_token_updates`) counts the
  steps in which a row had a gradient, not its updates. The wording is fixed under
  review_issues.md, I7.

**What to check later.** Proposed by this session; nothing has been run.
- **a. Measure the real movement.** Per bank row, the cumulative |Δ|, or the number of steps
  in which its update exceeds a threshold, bucketed by how often the token occurs. Needs
  readout U4c.
- **b. Lazy bank updates.** Moments and updates only for the rows present in a step
  (SparseAdam-style), against dense AdamW, at the standard h12 recipe.
  - The question: does the momentum drift of absent rows help or hurt rare tokens?
  - Compare the rare-token buckets with `documents/scripts/bank_gates_eval.py`. It needs
    `trainer.count_token_updates: true` in the runs.
- **c. The banks' β₂.** At β₂ = 0.95, the same occurrence at step 1,000 moves an element
  0.45 lr at once and about 6 lr in total, against 2.5 and 25 (this session's arithmetic).
  - `build_optimizer` already passes keyword arguments on to AdamW.
  - The bank group in `parameter_groups` would need the same.
- These interact with `bank_lr_ratio` = 4, which scales every move, and with σ = 1e-4, the
  starting point.

## U2. B1's analysis under U1

- **The plan's estimate:** B1's trust "moves only in steps where t occurs, by about one
  learning rate per update", so reaching s = 0.5 "takes at least 754 of the token's own
  updates" ([ame_bank_gates_plan.md:121-124](ame_bank_gates_plan.md#L121-L124)).
- **Under U1** that holds for a token present in every step. A rare token's trust moves many
  lr per occurrence (rev2's reply).
- **What was measured:** the trust "stays where it is put", 0.949-0.950 in `b1open_std001` and
  0.050-0.054 in `b1_std1`
  ([ame_bank_gates_stage2_report.md:373-374](ame_bank_gates_stage2_report.md#L373-L374)). So
  the practical difference may be small.
- **What to do:** B1 is rejected and off by default. Re-read its results with U1 only if the
  gates are revisited. The plan gets a correction note now (review_issues.md, I7).

## U3. Bank init σ

- **The evidence** ([ame_bank_sigma_grid.md](ame_bank_sigma_grid.md), "The answer"):
  - at h12, all ten σ from 1e-5 to 0.0178 tie, inside one tie band; the measured best is
    3.16e-5;
  - below the plateau the init fails: 1e-7 is +0.064, 1e-8 +0.183, 1e-10 +0.442. The edge lies
    between 1e-7 and 1e-5, where nothing was run;
  - σ = 1 is +0.097.
- **The code follows it since 2026-11-25** (review_issues.md, I17 and I8): the default is
  1e-4 under the QK-norm (1.0 without it, N1), and the module docstring states the grid's evidence instead of calling σ ≈ 1
  load-bearing. The standard configs set 1e-4 too.
- **To check later** (standard_model_improvements.md, "Later"): σ = 1e-4 at other learning
  rates. Does the cliff move?

## U4. Readouts for future analysis

| # | Readout | The gap today | Proposed by | What it takes |
|---|---|---|---|---|
| a | The banks' grad norm, `train/grad_norm/banks`, under FSDP2 | Was skipped under FSDP2, even on one GPU, so the h20 standard runs (FSDP2, `mem_parallel_size: 4`) would never have logged it, though the σ grid relied on the banks' share of the clip | rev1 #4, rev2 #9 | **Done 2026-11-25** (review_issues.md, I19) |
| b | Attention entropy and top-k mass on a sharded layer | Still skipped after `8b78b805` ([standard_models_report.md:39](standard_models_report.md#L39)) | The standard-models report | The diagnostics re-run the gather and `v_proj` outside FSDP2's hooks. They would need gathered weights or a hook-aware path (this session's reading) |
| c | The real movement of a bank row | `steps_seen` counts the steps with a gradient (U1) | This session | Cumulative \|Δ\| per row, or the steps above a threshold, written beside `token_counts.pt` |
| d | Dense τ, α and β from a checkpoint | `ame_ckpt_readouts.py` reads AME layers only. A dense layer with QK-norm and the KV shift has `tau`, `alpha` and `beta` but no `slopes` (rev2 #15) | rev2 #15 | Accept a layer without `slopes` and skip its horizons |
| e | Token counts in the runs meant for rare-token analysis | `trainer.count_token_updates` is off by default ([trainer.py:232](../../src/core/trainer.py#L232)) | This session | Set it to `true` in those runs; it writes `token_counts.pt` beside every checkpoint |

## U5. The common evaluator

`documents/scripts/parity_numerics.py` scores h12 and h16 runs. Scoring the h20 standard runs,
and other configs, with the same evaluator needs the following (rev2 #8):

| Case | Fails because | Needs |
|---|---|---|
| FSDP2 runs: Stage 2c, `standard/ame_h20` | `ame_ref32` does not null `trainer.distributed`, so `build` reaches `setup_distributed_training` with no process group. `load_checkpoint` ([parity_numerics.py:177](../scripts/parity_numerics.py#L177)) reads only `__model_checkpoint_filename.pt` and `memory_parallel_rank0.pt`, while an FSDP2 run writes a distributed checkpoint and one sidecar per rank | `trainer.distributed=null` in `ame_ref32`, and a loader for sharded checkpoints and per-rank sidecars |
| A null learning rate (`stage_2c/ame_h20_pe_shift.yaml`) | `float(cfg.trainer.learning_rate)` at [parity_numerics.py:146](../scripts/parity_numerics.py#L146) raises a `TypeError` | Today, `--override trainer.learning_rate=<n>` gets past it |
| Configs that give their length as `trainer.n_tokens_b` | `build` never calls `resolve_n_steps`, so the warmup's `${trainer.n_steps}` resolves on null. No config on `ame_main` does this yet | Call `resolve_n_steps` in `build` |

The docstring that overclaims is fixed under review_issues.md, I11. The inexact `--attn-seqs`
of `bank_gates_eval.py` is I23 there.

## U6. Stability of the AME h12 recipe

**What happened** in the standard-models task ([standard_models_report.md](standard_models_report.md)):
- The recipe (σ 1e-4, `bank_lr_ratio` 4, 2⁻⁸) spiked in 1 of its 3 runs.
  - `zs1bceoy`: the grad norm reached 3.0 at step 356 and the loss peaked at 7.47 (reference
    4.03), recovering only slowly.
  - It ended +0.0771 over the last 500 steps against `zdmv3qci`.
  - The rerun, `6yx46ng1`, tied (+0.0017).
- Runs are not bit-reproducible: the losses of same-config runs first differ at step 20 (AME)
  and step 12 (dense).

**What to check later** (this session):
- The spike rate, with more seeds, and at h16 and h20.
- What drives it: the clipping, the banks' β₂ (U1c), the warmup.
- For tie tests: budget a second run up front, or use a pass rule that tells a spike episode
  from a level shift.

## U7. Reproducing a pre-merge run

- **Decided (Maciej, 2026-11-25):** the old AME configs stay at the default `bank_lr_ratio`,
  4.0, and take the default `bank_init_std`, 1e-4 under the QK-norm since 2026-11-25. Rerunning one gives
  today's recipe, not the recorded run.
- **To reproduce or resume a pre-merge run, set `trainer.bank_lr_ratio=1`** (rev2's reply),
  and `++common.attention.ame.bank_init_std=1.0` for its init.
  - That rebuilds the pre-merge optimizer exactly:
    - there is no bank group in `parameter_groups`
      ([model.py:753](../../src/core/model.py#L753));
    - the controller steps at the backbone's learning rate
      ([main.py:343](../../main.py#L343)).

    So an old checkpoint's optimizer state fits again.
  - Without it, a pre-merge checkpoint fails to load on the ordinary path. Under memory
    parallelism it resumes at 4× with no error
    ([llama.yaml:16-20](../../configs/_trainer/llama.yaml#L16-L20)).
- **Across clusters:** the entropy rerun tied LEM's reference at +0.0017, and earlier
  cross-cluster pairs differed by about ±0.002. Paper numbers stay within one cluster.
- **Run names:** a config that loads `/base_schema` types `learning_rate` as a float. So even
  `trainer.learning_rate=9` names the run `…lr9.0`, which `run_exp.py`'s tmux step could
  not handle until `913f4111` (review_issues.md, I13 and I18).

## U8. The ST gate, if it is ever revived

- **Status:** ST is rejected, because it breaks AME's precomputed score table, and is off by
  default. Its code stays (Maciej, 2026-11-25).
- **The limitation:** built under a non-CPU default device, every layer's ST gate gets the
  same init.
  - `_private_generator` ([ame.py:150-159](../../src/core/ame.py#L150-L159)) seeds from the
    CPU RNG's state, which does not advance there.
  - Only the GPU tests build that way. `main.py` builds on the CPU
    ([main.py:396](../../main.py#L396)), so every trained ST run is unaffected.
- **If distinct gates are ever needed:** add a salt only under a non-CPU default device. That
  keeps the CPU path's seeds, and so the trained ST runs, as they are (rev2's reply). Any
  unconditional change to the seed would change every ST init.

## U9. Runs already planned

From [standard_model_improvements.md](standard_model_improvements.md), "Later, each only with
Maciej's OK":
- **Dense h16 with QK-norm and the shift,** about 13 GPU-h per run:
  - at 2⁻⁹, to see whether the gain holds;
  - at 2⁻⁸, to see whether QK-norm makes it stable.
- **AME h16 at 2⁻⁷:** its best learning rate is at the edge of the grid.
- **The standard configs at h20,** about 14 h on 4 GPUs each.
- **σ = 1e-4 at other learning rates,** to see whether the cliff moves (U3).
