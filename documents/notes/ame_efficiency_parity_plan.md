# Does AME's efficiency work cost it quality? A parity audit against dense at h12: plan

**Status (2026-11-06, 21:30 CET): plan only, nothing run.** The `/goal` that executes it
is [the last section](#9-the-goal-text).

- **The run's record** will be
  [ame_efficiency_parity_report.md](ame_efficiency_parity_report.md), created by the run.
- **This file** changes only to correct the plan, and every change is logged in the report.
- **The comparison this audit serves** is [ame_comparison_h12.md](ame_comparison_h12.md).
- **The optimizations it audits** are in
  [ame_optimization_report.md](ame_optimization_report.md); `§` numbers below are that
  report's sections.

## 1. The question

The best AME and the best dense at h12 end 0.143 nats apart: S2-PS `doi36p8k` reaches
3.1250 and S1-D `x0xlqroj` 2.9824. Both ran at n = 8 for 4B tokens on the same batches.

The two arms do not share an implementation standard:

- **Dense runs a standard stack**: torch.compile, bf16 autocast, SDPA flash attention,
  fused AdamW.
- **AME's layer and bank path went through the report's inventory of about 40 MFU items**
  (§1.0). Several change precision, the attention kernel, the gradient path or the
  optimizer.
- **Each was gated for throughput, and some for computing the same function.** Those
  function gates ran at toy shapes, mostly in float64.
- **None was gated for training quality at the production shape and precision.**

**The question.** How much of the 0.143 nats comes from how AME is implemented, rather
than from what AME is?

**The null hypothesis, H0.** Every AME-only efficiency change active in S2-PS:

- (a) computes the same training step as a plain implementation of the same AME design
  under dense's mixed-precision policy, up to rounding no larger than dense's own
  production rounding; and
- (b) switched off, all together, moves S2-PS's `final_ce` by no more than run-to-run
  noise.

Dense's mixed-precision policy: fp32 master weights, matmuls in bf16 under autocast,
everything else in its input's dtype.

**A verdict for every inventory item:**

| Verdict | Meaning |
|---|---|
| **bug** | computes a different function or gradient than its reference, beyond rounding |
| **numerical** | differs from its reference by more than dense's rounding budget, with no detectable training effect |
| **harmful** | a one-directional training effect beyond the noise floor, sized in nats at 4B tokens |
| **cleared** | none of the above |

**The answer** is a decomposition: 0.143 = the implementation's share, measured or
bounded, plus the rest.

## 2. Scope

**In scope:** every difference between the two runs' training stacks that comes from an
efficiency decision, whether AME-only or shared.

**Out of scope, and why:**

- **The AME design**: id-indexed Q/K, QK-norm, τ, ALiBi, the KV shift, the sink and 12 V
  heads. That is the method under test.
- **Design-level hyperparameters**, C1–C4 in [section 4](#4-inventory-every-way-the-two-training-stacks-differ).
  They can explain part of the gap, but they are not implementation. They are listed so
  that no effect gets attributed to the wrong cause.
- **The §7 optimizations: O38(b), O39–O47, O45, compile autotune, and O9's re-measure.
  They are not in these runs' code.**
  - S2-PS ran `f9680709`. Its `src/` and `main.py` are identical to those at
    `40155323`, the merge-base with `ame_optimize`.
  - Every §7 code commit, from `922838c1` on, exists only on `ame_optimize`. The S2-PS
    config also enables none of their flags.
  - If they reach `ame_main` later, this plan's method applies to them. O45 already has
    a 300-step loss gate (§1.23).
- **Multi-GPU machinery**: Tier 3 (O6–O8), D3, O33, O40 and FSDP2. Both runs use one GPU
  with `distributed: null`.
- **Stages 1–2 and 1b–2b and their jobs.** Another session runs them.

## 3. The two runs

| | AME | dense |
|---|---|---|
| Arm | S2-PS: ALiBi and the KV shift | S1-D: GQA with RoPE |
| W&B run (ideas_cv/nano-ame) | `doi36p8k` | `x0xlqroj` |
| Slurm job | 6256577 | 6252398_0 |
| Code | `f9680709` (cemetery `3c8f550b`) | `8a240816` (cemetery `489342e2`); `src/` and `main.py` identical to `f9680709` |
| Config | [configs/stage_2/ame_h12_pe_shift.yaml](../../configs/stage_2/ame_h12_pe_shift.yaml) with `trainer.learning_rate=8` | [configs/stage_1/dense_h12_lr_grid.yaml](../../configs/stage_1/dense_h12_lr_grid.yaml) with the grid list `[8]` |
| `final_ce` | 3.1250 | 2.9824 |
| s/step on LEM, peak memory | 3.746 s, 70.4 GiB | 3.272 s, 47.0 GiB |
| Final checkpoint, under `/lustre/pd03/plgrid/plgllmefficont3/mstefaniak/llmrandom_cemetery/` | `s2_ps_h12_lr8.0_2026-11-06_13-52-40/checkpoint/6256577/0/step_3814`, plus the bank sidecar | `s1_d_h12_lrNone_2026-11-06_03-35-57/checkpoint/6252398/0/step_3814` |

**Shared by both runs**, from their logged configs:

- h12, 3,815 steps × 1,048,576 tokens, data seed 123;
- AdamW with betas (0.9, 0.999), eps 1e-8 and weight decay 0.1;
- a cosine schedule with 38 warmup steps, down to 10% of peak;
- clipping at 1.0;
- micro-batch 32 × 16;
- autocast bf16, no FSDP2, `compile_blocks_full`, `sequence_length` 2049;
- one LEM H100.

## 4. Inventory: every way the two training stacks differ

Read from the code at `f9680709` and from both logged configs. Phase 0 re-derives this
inventory and adds anything missing.

### AME-only efficiency changes active in S2-PS: the suspects

| # | Change (§) | What the run does | The reference it replaced | Switch it off with | Already verified | Risk to quality |
|---|---|---|---|---|---|---|
| **A1** | Tier 2 bank path, O1–O5 (§1.2): `trainer.mem_parallel_size: 1` | A custom autograd gather. Per-token bank gradients are `index_add_`-ed into a persistent fp32 `main_grad`. The banks leave the backbone optimizer for their own AdamW (not fused; defaults; wd 0) over fp32 masters. Joint clip; three step hooks | The banks as ordinary `nn.Embedding` parameters in the backbone's fused AdamW, in its wd-0 group; standard autograd and clip | `trainer.mem_parallel_size: null` | On a float64 CPU toy model: parameter-for-parameter equal to the ordinary path without clipping, and within one fp32 ulp with it (`tests/test_memory_parallel.py`) | None in exact arithmetic. Never compared at production precision and shape |
| **A2** | O3, the bf16 bank compute copy (§1.2). `build_memory_parallel_controller` in `main.py` picks bf16 whenever autocast is on | The gathered Q/K rows are bf16, so the whole prep runs in bf16, forward and backward: QK-normalize, the KV-shift mix (`alpha.to(k.dtype)`) and the τ scale (`tau.to(q.dtype)`) | fp32 banks: the same prep runs in fp32, and the result is cast to bf16 only for the kernel | Together with A1: `mem_parallel_size: null`. Alone: needs an opt-in compute-dtype flag | Nothing. `main.py`'s docstring says an fp32 bank "gets cast anyway", which holds only after the prep | Extra bf16 rounding in the normalize, the mix and τ, and in their gradients, bank gradients included. In the forward, τ ≈ 8.4 sits on a bf16 grid of 0.0625 |
| **A3** | D2, the fused Q/K pair (§1.4), traced by torch.compile inside the block (§1.18) | One gather serves both banks. Inductor fuses the normalize backward with the scatter into `main_grad` | Two separate gathers, eager | Together with A1. Alone: needs an opt-in flag | On CUDA at a toy shape, mp 1: `main_grad` within 1e-4 relative of eager (job 33504, `ame_optimize` only). **Inductor on CPU miscompiles this scatter** (k banks; rows id and id+1; 7.3e-3) | A codegen bug of the CPU kind at the production tiling would silently give wrong bank gradients |
| **A4** | `flex_attention`, compiled, with ALiBi in `score_mod` and a cached causal `BlockMask` | The production kernel | `backend: reference`: fp32 scores, an explicit bias, the matrix materialized | `common.attention.ame.backend: reference` | Forward equals the reference on CPU in fp32, and on CUDA at a small shape (rtol 1e-4). **No test anywhere compares gradients**: the CUDA training tests only assert that gradients exist | A wrong or lossy backward would train a different model than the forward describes |
| **A5** | S2, the sink as a log-sum-exp merge (§1.7) | `out = out_seq + sigmoid(s − lse)·(v_sink − out_seq)`, which needs a gradient through flex's returned LSE | The sink as kv column 0 inside attention | `backend: reference` puts the sink back as a column | Forward only. "Loss trajectory unchanged" over a 30-step h20 benchmark | **If flex's LSE output carries no gradient, dq loses its ∂lse/∂q term and the sink and score gradients are wrong.** Nothing checks this |
| **A6** | S3b, the sink logit as an fp32 multiply-and-sum (§1.9) | — | An einsum | `backend: reference` | Forward | Low: it runs in fp32 |
| **A7** | τ folded into q before the bf16 cast | The score scale is applied to a bf16 q, and the kernel runs at scale 1 | Dense's kernel applies the fixed 1/√64 to its fp32 accumulator | `backend: reference` covers part; the rest needs an opt-in flag | — | One extra bf16 rounding on every score |
| **A8** | The compiled AME block | 8–10 graphs per block per micro-step (§1.21), with the bank gather traced inside | Eager, or one graph | `compile_blocks_full: false` | CUDA compiled-against-eager forward at a small shape | Inductor fusion and tiling at the production shape |
| **A9** | The bank sidecar checkpoint (O30) | Bank masters and moments are saved beside the checkpoint | In the state dict | — | Round trip at mp 1/2/4 in float64 | None for training. A precondition for every offline check |

### Shared efficiency choices: both arms

| # | Choice | Bearing on the comparison |
|---|---|---|
| X1 | `compile_blocks_full` | Both; AME runs 8–10 graphs per block and dense 4 (A8) |
| X2 | Autocast bf16 with no FSDP2 at one GPU (O37, §1.11) | Both |
| X3 | Cross-entropy outside autocast, on bf16 logits (`_loss_ctx`, §1.6) | Both. A known one-directional cost: O45's 300-step gate found an fp32 log-sum-exp CE training about 0.3% lower (§1.23). Symmetric unless it interacts with the arm |
| X4 | `sequence_length` 2049 (S3, §1.8) | Both; the same data |
| X5 | Fused AdamW on the backbone | Both |
| X6 | Micro-batch 32 × 16 | Both |
| X7 | The residual-metrics hook breaking the graph on every micro-step (before O41) | Both; scheduling only |
| X8 | `compile_ce`, `compile_loss` and `fused_linear_ce` off | Both, as both logged configs confirm |

### Dense-only: the standard stack

| # | What | Role here |
|---|---|---|
| D1 | SDPA (flash, then efficient, then math) with `is_causal`; the 1/√64 scale applied inside the kernel; RoPE cos and sin cast to bf16; GQA `repeat_kv` | The standard. It sets the rounding budget A1–A8 are judged against |

### Design differences: not implementation, kept in view

| # | Difference |
|---|---|
| C1 | Weight decay: the banks, α, β, τ and the sinks get 0. Dense decays every parameter, norm gains and embedding included |
| C2 | Bank init σ = 1 at the global learning rate. With QK-norm a bank row turns about lr/σ per step, roughly 30× slower than a dense projection ([ame_implementation.md](ame_implementation.md), "Weight initialization") |
| C3 | AME's learning rate is not bracketed. n = 8 is the edge of S2-PS's check over 8, 9 and 10, and S1-A0's optimum was n = 6 |
| C4 | MHA-shaped V with 12 heads, against GQA's 6 KV heads; no Q/K projections |

Every A/B here compares AME with AME at the same n, so none of C1–C4 confounds an A/B.
They matter only when the decomposition is read.

**C3 is being settled elsewhere.** The Stage 2 session submitted S2-PS at n = 7
(`zdlmmz7n`) and n = 6 (`egvqspym`) at 20:15 on 2026-11-06 (`stage_2_report.md`). If
either beats n = 8, keep this audit at n = 8 anyway: it measures the implementation, and
n = 8 is the pair that ame_comparison_h12.md analyses. Record the new best AME and its
gap to dense in the report's decomposition.

## 5. What is already known, and the gaps

| Known | Gap |
|---|---|
| Tier 2 equals the ordinary path in float64 on a CPU toy model (A1) | Never compared in bf16 at the production shape. The bf16 prep (A2) exists only on the Tier 2 path, so that gate cannot see it |
| The traced bank path is correct on CUDA at a toy shape, and inductor miscompiles it on CPU (A3, §1.18) | Production shape untested. `tests/test_bank_compile.py` exists only on `ame_optimize`, added in `922838c1` |
| flex equals the reference in the forward (A4, A5) | **No gradient comparison for the production attention path, on any device** |
| Two loss-trajectory gates exist in the report: `compile_ce` (rejected, one-directional) and O45 (passed with a downward shift; not used here) | **No AME-only change active here has a training-quality check** |
| — | Neither arm has a run-to-run noise measurement at 4B tokens |

[ame_comparison_h12.md](ame_comparison_h12.md) also leaves two observations for the
implementation to explain or rule out:

- **F3:** at step 10 both runs spike on the same batch, but AME's gradient norm reaches
  46.5 against dense's 15.9.
- **F4 and H5:** AME's gradient norm runs at about 0.8× dense's after warmup, and the
  residual-stream explanation was rejected.

## 6. Method

**Order:**

1. Phase 0;
2. submit wave 1 (Phase 2);
3. Phase 1 while wave 1 trains;
4. read wave 1;
5. Phase 3 only if needed;
6. Phase 4 only for harmful items;
7. Phase 5.

### Phase 0: set up, with no training

- **0.1 Work branch.** Create `parity_lem` at `f9680709` in a clean worktree,
  `~/code/nano-parity`. Configs and any opt-in code go there. Docs go on `ame_main`.
- **0.2 Re-derive the inventory** from the code at `f9680709` and the two logged configs
  in W&B. Add anything missing; confirm each "active" by the code path the run took.
- **0.3 Configs** in `configs/parity/`, one per run of Phase 2. Each copies the S2-PS or
  S1-D recipe with:
  - `learning_rate: 8`;
  - a literal run name without a dot (see section 7);
  - the tag `parity`;
  - the final checkpoint on.

  Schema-check each locally.
- **0.4 Smoke runs**, 60 steps each on LEM, for P1 and P-REF. They set:
  - s/step, as the median over steps ≥ 15, and peak memory;
  - P-REF's micro-batch: the largest power of two that fits, with
    `gradient_accumulation_steps` adjusted so the batch stays 512;
  - every job's time limit, at 1.5× its projection.

### Phase 1: offline numerics, on one LEM GPU per job, with no training

**The harness** (`documents/scripts/parity_numerics.py`):

- builds either model from its config, with switches for `mem_parallel_size`, `backend`,
  compile, autocast and dtype;
- loads either checkpoint format into either path. AME's production format puts the
  banks in the sidecar; P1 and P-REF put them in the state dict;
- runs forward and backward on fixed batches: the held-out set's first micro-batches and
  the first training micro-batches of seed 123;
- works from two states: init (seed 123) and the two final checkpoints.

**The checks, in priority order:**

- **1.0 The existing CUDA gates, on LEM, at `f9680709`.** Run
  `pytest -m gpu tests/test_ame.py tests/test_memory_parallel.py`. Port the mp-1 case of
  `tests/test_bank_compile.py` from `ame_optimize` if it runs against `f9680709`'s bank
  path; otherwise write its equivalent.
- **1.1 Reproduce `final_ce` from each checkpoint**: 3.1250 and 2.9824, within 0.001.
  This validates the harness and the sidecar load. Then evaluate both again with the fp32
  reference path. This pair of evaluators is the **common evaluator** Phase 2 uses.
- **1.2 The attention backward (A4–A7)** for one AME layer at the production shape: 32 ×
  2,048, 12 heads of 64, with the trained layer's weights.
  - Compare flex (bf16, compiled, as run) with the reference (fp32).
  - Compare the output; dq, dk and dv; and the gradients of α, β, τ, `k_sink` and
    `v_sink`.
  - **The named risk is the gradient through flex's LSE.** Test layers with small and
    with large sink weight.
- **1.3 The bank gradient (A1–A3).** Compare `main_grad` from the compiled fused path
  with an eager fp32 `nn.Embedding` gradient, for every layer, on a real batch. The
  structural checks:
  - rows of ids absent from the batch are exactly zero;
  - no error correlates with a neighbouring row, the CPU bug's signature;
  - the ids reaching each layer equal `input_ids` position for position: not the
    targets, and not shifted.
- **1.4 The prep's precision (A2, A7).** Compare the bf16 prep with an fp32 prep: the
  mixed q and k, and the gradients of α, β, τ and the banks. Also report the forward's τ
  against the fp32 τ.
- **1.5 One optimizer step (A1).** From the same state (init, and the final checkpoint
  with its moments), on the same batch, compare Tier 2 with the ordinary path:
  - the clip coefficient;
  - every updated parameter, bank masters included;
  - the Adam moments;
  - the compute copy, which must equal bf16(master) after the step.
- **1.6 The whole-model error budget.** For each model, compare the production path with
  its fp32 eager reference, at init and at the final checkpoint, on 8 micro-batches.
  - The references: AME with mp null, `backend: reference`, fp32 and no compile; dense
    with fp32, SDPA math and no compile.
  - Measured: the loss difference, and for each parameter group the gradient cosine, the
    relative error and the signed projection ⟨Δ, g⟩/‖g‖².
  - Add a third AME path, **standard mixed-precision AME**: autocast bf16, mp null, flex,
    compiled. It separates what the optimizations add from what bf16 does to AME anyway.
- **1.7 The incidental machinery.**
  - The cached `BlockMask` equals a freshly built one for every shape the run used:
    train, eval and `final_ce`.
  - AME's instrumentation consumes no RNG and touches no parameter, buffer or cache.
  - Neither run's `output.log` shows a recompile-limit fallback.

**Flag rules.** These are defaults; departing from one needs a recorded reason.

- **Exact items** (1.1, the structure and alignment checks of 1.3, 1.5 and 1.7) must be
  equal, or within fp32 rounding (≤ 1e-6 relative).
- **Tolerance items** (1.2, the values in 1.3, 1.4 and 1.6) are flagged if any of these
  holds:
  - a parameter group's relative error exceeds twice the largest of dense's groups in
    1.6;
  - its signed projection exceeds 1e-3 in magnitude;
  - its loss difference exceeds twice dense's.
- **A structural flag in 1.2 or 1.3 is a bug.** Characterize it fully before anything
  else: which shapes, which layers, what size. Record it for Maciej at once.

### Phase 2: full-length training A/Bs (wave 1, submitted together)

Every run: 3,815 steps, n = 8, seed 123, the code of `f9680709`, final checkpoint saved.

| Run | What | Why |
|---|---|---|
| **P0** | S2-PS, identical | The run-to-run floor at 4B tokens. It also shows whether the production path is deterministic |
| **P0b** | S2-PS at micro-batch 16 × 32 | A reassociation floor: the same function with different rounding. The floor to use if P0 turns out bit-identical |
| **P1** | S2-PS with `mem_parallel_size: null` | A1, A2 and A3 off |
| **P-REF** | S2-PS with `mem_parallel_size: null` and `backend: reference`, at the micro-batch from 0.4 | Every AME-only efficiency change off, A1–A7. The closest thing to AME implemented the way dense is |
| **D0** | S1-D at n = 8, identical | Dense's floor |

**Fallback.** If P-REF projects past 22 h, run screens instead: 1,000 steps on a
1,000-step schedule, beside a production run on the same schedule and its twin. The
attention items then rest on Phase 1.

**Reading the results.**

- **Paired train-loss gap**: each run against `doi36p8k` on the same batches, as
  50-step means.
- **Both evaluations of every final checkpoint**: `final_ce` as logged, and the common
  evaluator (1.1) in both modes. A difference in evaluation numerics must not read as a
  training effect.
- **The floor F**: the largest |Δ `final_ce`| of P0 and P0b against `doi36p8k`, under
  either evaluator. D0 gives dense's.
- **Harmful** means |Δ `final_ce`| ≥ max(3F, 0.005) *and* the 50-step paired gap keeps
  one sign in at least 90% of the last 1,000 steps. Below that the result is "no
  effect detectable", with the bound |effect| < max(3F, 0.005).
- **The gradient-norm traces of P0, P1 and P-REF.** Check whether the reference paths
  change F3's step-10 spike or F4's 0.8× ratio. If they do, the implementation shapes
  AME's gradient.

### Phase 3: bisect, only for a harmful P1 or P-REF

- **If P1 is harmful:**
  - separate A2 from A1 with an opt-in `trainer.bank_compute_dtype: float32` on the Tier
    2 path;
  - separate A3 with an opt-in eager bank path;
  - then run the culprit at full length.
- **If P-REF is harmful and P1 is not:** test the attention items, guided by 1.2 and 1.4:
  - for A5, an opt-in flex path with the sink as a column, as before S2, taken from git
    history;
  - for A7, τ applied in fp32.
- **Screens may rank the candidates.** A screen is a 1,000-step schedule run beside a
  production screen and its twin. The final attribution is always full length.
- **New code is opt-in only**: off by default, with a test proving the default
  bit-identical, on `parity_lem`.

### Phase 4: fix and price, only for harmful items

- **The fix** is the cheapest one that keeps the throughput gain, for example the prep in
  fp32 inside the compiled graph.
- **Its throughput cost**: a 60-step run on LEM, median s/step over steps ≥ 15.
- **Its quality**: a full-length run.
- **Leave every default unchanged** and recommend in "For Maciej".

### Phase 5: report

**The report holds:**

- the inventory, with a verdict and its evidence for every item;
- every check and every job;
- the floors;
- the decomposition of 0.143, with its uncertainty.

**Also:**

- Add the result to [ame_comparison_h12.md](ame_comparison_h12.md): a finding, and the
  status of the hypotheses it bears on.
- Do not edit [ame_implementation.md](ame_implementation.md) or
  [ame_optimization_report.md](ame_optimization_report.md); other sessions own them. Put
  the pointers in "For Maciej" instead.

## 7. Budget, cluster and rules

**Budget.**

- **Projected cost, about 30 GPU-h for wave 1:**
  - P0: 4.2 h;
  - P0b: about 4.4 h;
  - P1: about 5 h;
  - P-REF: about 12 h;
  - D0: 3.7 h.

  The smoke runs set the P1 and P-REF figures.
- **Phase 1**: about 3 GPU-h.
- **Phases 3 and 4, if needed**: 10–30 GPU-h.
- **The cap is 80 GPU-h.** Past it, record the case in "For Maciej" and submit nothing
  more.

**Cluster.**

- **LEM**, one H100 per job. Partition, account and gres are as in the memory note
  `reference_experiment_submission.md`.
- **Exclude node r16-11** with `+infrastructure.slurm.exclude=r16-11`.
- **Startup** takes about 15 minutes of dataset indexing.
- **At most 5 concurrent jobs.** The Stage 1–2 and 1b–2b session runs on the same
  account: never cancel or touch a job you did not submit.

**Rules.**

- **Submit from the clean worktree.** `run_exp.py`'s `version_code` commits `git add -A`
  to a cemetery branch.
- **No dot in a run name.** `run_exp.py`'s tmux step fails on one. If it happens, run
  `cd <clone> && sbatch exp.job` by hand.
- **Clear `generated_configs/`** after every submit.
- **Keep SSH connections few.** Use one persistent ControlMaster with keepalives: many
  new connections got LEM's port 22 blocked for 15 minutes on 2026-11-06.
- **Disk**: four AME checkpoints at 13 GB each, plus dense's 2 GB.
- **Commit only your own files.** Other sessions' uncommitted edits stay out. At the
  time of writing those are `documents/notes/ame_implementation.md`,
  `documents/notes/pretraining_stage2_plan.md` and `scripts/free_gpus.sh`.
- **Never delete branches, never message other sessions, never persist secrets.**

## 8. What the outcome means

- **All cleared.** The efficiency work is quality-neutral at h12. The 0.143 is the method
  plus the design choices C1–C4, and the next step is ame_comparison_h12.md's angles
  A1–A6.
- **Something harmful.** Its size, its mechanism, its fix and that fix's price; the gap
  restated with the fix applied.
- **A bug.** It supersedes everything else. Every AME number since the change that
  introduced it carries it, and the report must name the runs affected.

## 9. The `/goal` text

```text
/goal Run the AME efficiency-parity audit in documents/notes/ame_efficiency_parity_plan.md on the LEM cluster: find out whether the MFU optimizations applied to AME, and not to the dense baseline, cost AME model quality, using the two best h12 runs, S2-PS (W&B doi36p8k) and S1-D (x0xlqroj), at n = 8. That document is the plan; read it in full before starting, and follow it. documents/notes/ame_efficiency_parity_report.md is the record: every check, job, result and decision with its reason, committed on ame_main after every wave and at least every 3 hours. Code and configs live on branch parity_lem, in a clean worktree at f9680709, so every run shares the code of the runs it tests; submission on LEM is as in memory reference_experiment_submission.md. This runs unattended; Maciej is not available. Never stop to ask or wait for confirmation: decide what the plan leaves open, record why, and keep going. Queue waits, failed jobs, OOM, time limits and disk space are yours to handle; none of them ends the run. When a check fails, dig until its mechanism is understood and its size measured; a first anomaly is a lead, not a result. Out of scope: changing the default behavior of AME, dense or the trainer (new code only as opt-in flags whose default a test proves bit-identical), the AME design and its hyperparameters, the §7 optimizations that are not in these runs' code, Stages 1-2 and 1b-2b and their jobs, and cancelling jobs you did not submit. Stay within the plan's budget: 80 GPU-hours, at most 5 concurrent jobs. Anything beyond it, any default you think should change, and anything else that needs Maciej's decision goes to a "For Maciej" section at the end of the report. Done when every item in the plan's inventory has a verdict with its evidence, the report states how much of the 0.143-nat gap the implementation explains or bounds it, ame_comparison_h12.md carries the result, both are committed, and none of your jobs is queued or running.
```
