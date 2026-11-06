# Does AME's efficiency work cost it quality? A parity audit against dense at h12: plan

**Status (2026-11-06, 22:00 CET): plan only, nothing run.** The `/goal` that executes it
is [the last section](#10-the-goal-text).

- **Revised the same evening, before anything ran.** No model training: short entropy
  jobs and local CPU tests only. The agent works alone and records its questions for
  Maciej as it goes.

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
- (b) switched off, singly or all together, does not move a short training run's loss
  one-directionally beyond run-to-run noise.

Dense's mixed-precision policy: fp32 master weights, matmuls in bf16 under autocast,
everything else in its input's dtype.

**A verdict for every inventory item:**

| Verdict | Meaning |
|---|---|
| **bug** | computes a different function or gradient than its reference, beyond rounding |
| **numerical** | differs from its reference by more than dense's rounding budget, with no detectable training effect |
| **harmful** | a one-directional training effect beyond the noise floor within a 300-step gate |
| **cleared** | none of the above, as far as checks and 300-step gates reach |

**What this audit can deliver** is everything short of a full training:

- the bugs, at the production shape and precision;
- the numerics, against dense's rounding budget;
- the training effects visible within 300 steps.

From those, the report states the implementation's share of the 0.143: measured, bounded,
or "no evidence of any". The full-length runs that would size an effect at 4B tokens are
prepared and proposed, not run
([Proposed full-length runs](#proposed-full-length-runs-prepared-not-run)).

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
- **Model training.** Nothing past 300 optimizer steps or 45 minutes per job (section 6).
- **Jobs on LEM.** LEM is used only to copy the two checkpoints and read the two runs'
  logs.

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

## 6. Where the work runs, and what it may not do

- **Local CPU**, in this machine's pixi env, for unit tests and the harness's own tests.
  Two things can never be concluded on CPU:
  - whether compiled code is correct: inductor miscompiles the bank scatter on CPU
    (§1.18);
  - anything about the flex backward: compiled flex with gradients cannot even lower on
    CPU.
- **Entropy, in short GPU jobs on one GPU or several.** These cover:
  - the GPU tests;
  - the numerics harness;
  - training gates of at most 300 steps;
  - throughput probes of about 60 steps.
- **No model training.** No job runs past 300 optimizer steps or 45 minutes of wall time.
  The full-length A/Bs that would size an effect at 4B tokens are prepared, not run: see
  [Proposed full-length runs](#proposed-full-length-runs-prepared-not-run).
- **No jobs on LEM.** LEM is read once, to copy the two final checkpoints.

## 7. Method

**Order:**

1. Phase 0;
2. Phase 1;
3. Phase 2;
4. Phase 3, only if a gate flags;
5. Phase 4, only for harmful items;
6. Phase 5.

While an entropy job runs, do CPU work locally. Phases 1 and 2 may interleave.

### Phase 0: set up

- **0.1 Work branch.** Create `parity_audit` at `f9680709` in a clean worktree,
  `~/code/nano-parity`. Code, configs and the harness go there; docs go on `ame_main`.
  - Every job runs a committed state of `parity_audit`, and its log records that commit.
- **0.2 Re-derive the inventory** from the code at `f9680709` and the two logged configs
  in W&B. Add anything missing; confirm each "active" by the code path the run took.
- **0.3 Baseline the local CPU suite** at `f9680709` with
  `pytest -m "not gpu and not slow"`, so that a later failure is known to be new.
- **0.4 Configs** in `configs/parity/`:
  - **The entropy gate configs.** Each copies the S2-PS or S1-D recipe with:
    - `_cluster: entropy` and gres `gpu:h100:1`;
    - n = 8 and the gate schedule (Phase 2);
    - a literal run name without a dot (section 8);
    - the tag `parity`.
  - **The LEM configs for the proposed full-length runs**, written but not submitted.

  Schema-check each locally.
- **0.5 Copy the two final checkpoints from LEM to entropy**, verifying sizes and
  checksums: AME's 13 GB with its bank sidecar, and dense's 2 GB.
  - `/storage_nvme_*` is mounted only on entropy's GPU nodes, and `/home` is NFS. Pick
    the destination with that in mind, and record it.
  - If the copy cannot be done, use the init weights and the final state of a saved
    300-step gate instead, and record the limitation.

### Phase 1: offline numerics (short entropy jobs, no training)

**The harness** (`documents/scripts/parity_numerics.py`):

- builds either model from its config, with switches for `mem_parallel_size`, `backend`,
  compile, autocast and dtype;
- loads either checkpoint format into either path. AME's production format puts the
  banks in the sidecar; with `mem_parallel_size: null` they sit in the state dict;
- runs forward and backward on fixed batches: the held-out set's first micro-batches and
  the first training micro-batches of seed 123;
- works from two states: init (seed 123) and the two final checkpoints.

**The checks, in priority order:**

- **1.0 The existing GPU tests, on entropy, at `f9680709`.** Run
  `pytest -m gpu tests/test_ame.py tests/test_memory_parallel.py`. Port the mp-1 case of
  `tests/test_bank_compile.py` from `ame_optimize` if it runs against `f9680709`'s bank
  path; otherwise write its equivalent.
- **1.1 Reproduce `final_ce` from each checkpoint**: 3.1250 and 2.9824, within 0.001.
  This validates the harness and the sidecar load.
  - **First confirm that entropy's FineWeb copy holds the same data as LEM's.** The
    paths differ, and LEM's says `fineweb-edu`: compare the row count and the first
    held-out batch's token fingerprint. If the data differ, `final_ce` cannot be
    reproduced on entropy. Every other check compares two paths on the same batches,
    so it does not depend on this.
  - Entropy and LEM are both H100s on torch 2.7.1+cu128. If entropy still differs on
    the same data, record its value and use it as the reference for entropy.
  - Then evaluate both checkpoints again with the fp32 reference path. This pair of
    evaluators is the **common evaluator** used everywhere below.
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
  - Neither LEM run's `output.log` shows a recompile-limit fallback (read over SSH).

Several checks may share one multi-GPU job.

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
  else: which shapes, which layers, what size. Add it to "Questions for Maciej" at once.

### Phase 2: training gates (entropy, at most 300 steps each)

**The gate recipe:**

- S2-PS or S1-D at n = 8, seed 123, micro-batch 32 × 16;
- 300 steps: the 38-step warmup, then a cosine down to 10% at step 300;
- `final_ce` at the end; no checkpoint unless 0.5 needs one;
- about 25–35 minutes per gate, startup included.

| Gate | What | Why |
|---|---|---|
| **G0a, G0b** | S2-PS production, twice | The run-to-run floor. It also shows whether the production path is deterministic |
| **G0c** | S2-PS at micro-batch 16 × 32 | A reassociation floor: the same function with different rounding |
| **G1** | `mem_parallel_size: null` | A1, A2 and A3 off |
| **G-REF** | `mem_parallel_size: null` and `backend: reference`, at the largest micro-batch a 20-step probe fits | Every AME-only efficiency change off, A1–A7 |
| **GD0a, GD0b** | S1-D, twice | Dense's floor, and whether AME is more sensitive to rounding than dense |

**Reading the gates:**

- **Paired per-step train-loss gap**, each gate against G0a, on the same batches.
- **The floor F**: the larger of |G0b − G0a| and |G0c − G0a|, as the mean over the last
  50 steps, and the same for `final_ce`.
- **Harmful at 300 steps** means all three:
  - the mean gap over the last 50 steps is at least max(3F, 0.005);
  - at least 45 of those 50 steps have its sign;
  - the gate's `final_ce` agrees.

  That is the rule that rejected `compile_ce` (§2.10).
- **The gradient-norm traces.** Check whether the reference paths change ame_comparison_h12.md's
  F3 (the step-10 spike, 46.5 against dense's 15.9) or F4 (about 0.8× dense after
  warmup).
- **A clean gate clears an item only for the first 300 steps.** Effects that need
  thousands of steps stay open, for the proposed full-length runs.
- **If G-REF cannot finish 300 steps within 45 minutes**, shorten it, and every gate it
  is compared with, to the same step count, but not below 150 steps. Record the choice.

### Phase 3: bisect, only if G1 or G-REF flags

- **If G1 flags:**
  - separate A2 from A1 with an opt-in `trainer.bank_compute_dtype: float32` on the Tier
    2 path;
  - separate A3 with an opt-in eager bank path.
- **If G-REF flags and G1 does not:** test the attention items, guided by 1.2 and 1.4:
  - for A5, an opt-in flex path with the sink as a column, as before S2, taken from git
    history;
  - for A7, τ applied in fp32.
- **Every candidate gets a gate**, run beside a fresh production gate.
- **New code is opt-in only**: off by default, with a test proving the default
  bit-identical, on `parity_audit`.

### Phase 4: fix and price, only for harmful items

- **The fix** is the cheapest one that keeps the throughput gain, for example the prep in
  fp32 inside the compiled graph.
- **Its throughput cost**: a 60-step job on entropy against production on the same node,
  median s/step over steps ≥ 15.
- **Its quality**: a gate.
- **Leave every default unchanged.** The recommendation goes to "Questions for Maciej".

### Phase 5: report

**The report holds:**

- the inventory, with a verdict and its evidence for every item;
- every check and every job;
- the floors;
- what the evidence says about the implementation's share of the 0.143: measured,
  bounded at 300 steps, or "no evidence of any";
- the proposed full-length runs that would settle what the gates cannot, with their
  configs ready.

**Also:**

- Add the result to [ame_comparison_h12.md](ame_comparison_h12.md): a finding, and the
  status of the hypotheses it bears on.
- Do not edit [ame_implementation.md](ame_implementation.md) or
  [ame_optimization_report.md](ame_optimization_report.md); other sessions own them. Put
  the pointers in "Questions for Maciej" instead.

### Proposed full-length runs: prepared, not run

These size an effect at 4B tokens. They need about 30 GPU-h on LEM, where the two runs
trained, and they run only on Maciej's answer. Their configs go on `parity_audit` and
their exact submit commands into the report.

| Run | What | Why |
|---|---|---|
| **P0** | S2-PS, identical | The run-to-run floor at 4B |
| **P0b** | S2-PS at micro-batch 16 × 32 | The reassociation floor |
| **P1** | `mem_parallel_size: null` | A1, A2 and A3 off |
| **P-REF** | `mem_parallel_size: null` and `backend: reference` | A1–A7 off |
| **D0** | S1-D at n = 8, identical | Dense's floor |

**Their rule, for when they run.** A run is harmful if |Δ `final_ce`| ≥ max(3F, 0.005),
where F is the largest |Δ `final_ce`| of P0 and P0b against `doi36p8k` under either
evaluator, *and* the 50-step paired train-loss gap keeps one sign in at least 90% of the
last 1,000 steps.

## 8. Budget, clusters and rules

**Entropy.**

- **Submit as `mstefaniak` only**, through `ssh entropy`: partition `h100`, gres
  `gpu:h100:N`, QOS `mstefaniak_h100`. The 8-GPU cap is Maciej's alone. See the memory
  notes `reference_cluster_entropy.md`, `feedback_entropy_account.md` and
  `reference_experiment_submission.md`.
- **Preempting other users is allowed and expected.** Entropy is mainly our group's
  cluster, and our jobs are meant to preempt other users' jobs when we need the GPUs
  (Maciej, 2026-11-06). Submit without a preemption probe and without asking. Never
  cancel another user's job by hand.
- **Jobs may run in parallel**, up to the 8-GPU cap.
- **Limits per job**: at most 45 minutes of wall time, a time limit of at most 1 h, and
  at most 300 training steps.
- **Total budget: 25 GPU-h.** Past it, add a question and submit nothing more.
- **SSH**: fail2ban locks out rapid failures, so retry at most once a minute and never in
  parallel.
- **Submitting gates**: through `run_exp.py`, from the clean worktree, because
  `version_code` commits `git add -A` to a cemetery branch.
  - No dot in a run name: `run_exp.py`'s tmux step fails on one. If it happens, run
    `cd <clone> && sbatch exp.job` by hand.
  - Clear `generated_configs/` after every submit.
  - The GitHub push from this laptop uses the 443 rewrite in `reference_run_exp_gotchas.md`.

**LEM**: no jobs. Only the checkpoint copy (0.5), and reading the two runs' logs (1.7).

**Everywhere.**

- **Commit only your own files.** Other sessions' uncommitted edits stay out. At the
  time of writing those are `documents/notes/ame_implementation.md`,
  `documents/notes/pretraining_stage2_plan.md` and `scripts/free_gpus.sh`.
- **Never delete branches, never message other sessions, never persist secrets.**

**Working alone, and questions.**

- **The audit runs without user interaction.** Never stop or wait for an answer.
- **A decision that needs Maciej becomes a question** in "Questions for Maciej", at the
  end of the report. Each question is numbered and dated, and gives:
  - the context;
  - the options;
  - the default you are taking.

  Then carry on with that default.
- **Maciej answers periodically**, when available: in the session, or by editing that
  section. Check for answers whenever you commit, apply each one, and record what it
  changed.
- **These always need his answer:**
  - submitting any full-length or long training;
  - changing a default;
  - going past the budget.

## 9. What the outcome means

- **All cleared.** No evidence that the efficiency work costs quality: bugs and
  numerics are ruled out at the production shape, and training effects up to 300 steps.
  The 0.143 is then the method plus the design choices C1–C4, unless the proposed
  full-length runs say otherwise. The next step is ame_comparison_h12.md's angles A1–A6.
- **Something harmful.** Its size at 300 steps, its mechanism, its fix and that fix's
  price; the full-length runs that would size it at 4B are proposed.
- **A bug.** It supersedes everything else. Every AME number since the change that
  introduced it carries it, and the report must name the runs affected.

## 10. The `/goal` text

```text
/goal Run the AME efficiency-parity audit in documents/notes/ame_efficiency_parity_plan.md: find out whether the MFU optimizations applied to AME, and not to the dense baseline, cost AME model quality, using the two best h12 runs, S2-PS (W&B doi36p8k) and S1-D (x0xlqroj), at n = 8. That document is the plan; read it in full before starting, and follow it. The reference for the improvements applied to AME and not to dense is documents/notes/ame_optimization_report.md: read its complete inventory (§1.0) and the section of every item the plan's inventory names, and use it to confirm that the plan's inventory misses nothing active in these runs' code. documents/notes/ame_efficiency_parity_report.md is the record: every check, job, result and decision with its reason, committed on ame_main after every phase and at least every 3 hours. Code, configs and the harness live on branch parity_audit, in a clean worktree at f9680709, so everything you test is the code those two runs used. You may run CPU tests locally, and submit short jobs, on one GPU or several, to entropy to test things: GPU tests, numerical checks on the two final checkpoints, training gates of at most 300 steps, throughput probes. Do not run model trainings, because they would take too long: no job past 300 optimizer steps or 45 minutes of wall time, and no jobs on LEM, which you use only to copy the two checkpoints. Prepare the full-length runs the plan proposes, but do not submit them. On entropy, submit as mstefaniak (memory notes reference_cluster_entropy.md and reference_experiment_submission.md). Preempting other users' jobs there is allowed and expected, so submit whenever you need GPUs, up to the 8-GPU cap, without a preemption probe and without asking. Work alone, without user interaction: never stop or wait for an answer. If you need Maciej's decision, add a question to the "Questions for Maciej" section at the end of the report, with the context, the options and the default you are taking, and carry on with that default. He answers periodically, when available, in that section or in this session, so check for answers whenever you commit, apply them, and record what changed. When a check fails, dig until its mechanism is understood and its size measured; a first anomaly is a lead, not a result. Out of scope: changing the default behavior of AME, dense or the trainer (new code only as opt-in flags whose default a test proves bit-identical), the AME design and its hyperparameters, the §7 optimizations that are not in these runs' code, Stages 1-2 and 1b-2b and their jobs, and cancelling jobs you did not submit. Stay within the plan's budget of 25 GPU-hours. Done when every item in the plan's inventory has a verdict with its evidence, the report states what the evidence says about the implementation's share of the 0.143-nat gap and lists the full-length runs that would settle what the short gates cannot, with their configs ready, ame_comparison_h12.md carries the result, everything is committed, and none of your jobs is queued or running.
```
