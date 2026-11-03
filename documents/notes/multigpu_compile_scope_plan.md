# Plan: close the compile-scope gap under FSDP2 (2026-11-03)

**Goal: take FSDP2 pretraining of the GPT-2-arch h20 from 33.7% MFU to ~44% on
4 H100s**, by compiling the work that regional `compile_blocks` currently leaves
eager. Measurements and conventions come from
[multigpu_efficiency.md](multigpu_efficiency.md); the single-GPU history is in
[training_efficiency.md](training_efficiency.md).

Branch `multigpu-mfu`, worktree `/home/maciej/code/nano-multigpu`. Cluster:
entropy `h100` (h100a, 8×H100 SXM, NV18 NVLink), one job at a time.

---

## 1. The gap, and why it is worth chasing

The baselines measured FSDP2 scaling at **98.3% / 97.8%** on 2 / 4 GPUs, with
compute per micro-step invariant across world sizes. Parallelism is not the
problem. The problem is that the only FSDP2-compatible compile mode today is
regional `compile_blocks`, which costs 11 MFU points **on a single GPU, before
any communication exists**:

| | step ms | MFU | note |
|---|---:|---:|---|
| 1 GPU, no FSDP2, `compile_model` + `compile_loss` | 9,578 | **45.6%** | cannot run multi-GPU — no gradient sync without FSDP2 |
| 1 GPU, FSDP2 + `compile_blocks` (B1) | 12,678 | 34.45% | the configuration multi-GPU inherits |
| 4 GPUs, FSDP2 + `compile_blocks` (B4) | 3,240 | 33.70% | 323,440 tok/s |

Kernel-level decomposition of B1's 3,620 ms busy-time gap against the reference:

| component | ms | nature |
|---|---:|---|
| eager elementwise + eager norms + eager CE, net of the triton the reference pays instead | **+2,969** | **compile scope — this plan's target** |
| FSDP2 plumbing: DTensor optimizer +341, copies +198, one-rank NCCL +286 | +825 | FSDP2's own cost; mostly a world-size-1 artifact |
| flash attention and GEMMs, which are *faster* under FSDP2 | −173 | already in our favour |

**Arithmetic of the prize.** Removing the compile-scope component takes the
1-GPU FSDP2 step to ~9,700 ms (**~45% MFU**, essentially matching the non-FSDP2
number) and the 4-GPU step to ~2,500 ms (**~43.7% MFU, ~420k tok/s**). In
practical terms 4 GPUs would be worth **3.84×** a best-configured single GPU
instead of today's 2.96×.

### Where the eager time actually is

`compile_blocks` compiles only `block.attention_layer.layer` and
`block.ff_layer.layer` — the *inner* modules ([main.py:151](../../main.py)). It
skips the `Residual` wrapper around each, which holds the RMSNorm, the `out + x`
add, and the dtype casts at the compiled-region boundary.

The launch counts confirm it: the dominant eager kernels fire **408 times per
step at B4 = 51 × 8 micro-steps**, and 51 = 25 blocks × 2 `Residual` wrappers +
the head norm. The fp32 `BinaryFunctor`/`AUnaryFunctor` kernels are the norm's
scale/rsqrt and the residual adds; the `bfloat16_copy` kernels are the boundary
casts. **The eager glue is the Residual layer, one level above what we compile.**

That exclusion was deliberate but is now obsolete. `Residual.forward` used to
pass its per-block `log_name` string to the metric logger, so dynamo guarded the
graph on it and recompiled per block until `recompile_limit` silently dropped
blocks back to eager — the original block-compile failure (training_efficiency.md
Part 3.4). The whole-model-compile work moved that logging into a
`@torch.compiler.disable` helper ([model.py:67](../../src/core/model.py)), so the
string never enters tracing. **Nobody widened `compile_blocks` afterwards.** The
blocker is gone and the win was never collected.

---

## 2. The ladder

Three rungs, cheapest and safest first. Each is independently shippable and
independently measurable; stop as soon as the remaining gap is not worth the
risk of the next rung.

### R1 — widen the compile unit to the whole `TransformerBlock`

**Change.** A new top-level knob `+compile_blocks_full=true` in
`get_model_optimizer_scheduler` ([main.py:151](../../main.py)) that calls
`block.compile()` per block instead of compiling the two inner modules.
A *new* key rather than changing `compile_blocks`, so every historical run in
training_efficiency.md stays reproducible by its recorded command.

**Why it is low risk.** The compiled unit becomes the same module FSDP2 shards
(`modules_to_shard` lists the block's `_target_`), which is the pairing FSDP2
documents and the one upstream recipes use. It is still regional compile — no
FSDP2 hook is asked to survive tracing.

**Targets.** The 3,674 ms eager elementwise and 549 ms eager norms at B1 (918 +
137 ms at B4), minus whatever triton grows by.

**Expected.** Most of the 11 points. If R1 alone lands ≥ 42% at B4, rungs 2–3
are optional.

**Watch for.** Dynamo recompile warnings — the failure mode here is *silent*
(training_efficiency.md Part 3.4). Grep the job log for `recompil` and require
**zero** hits; B1/B2/B4 all had zero, so any hit is new and attributable.

**Open sub-question.** `main.py` compiles **before** `fully_shard`. If R1 shows
graph breaks or recompiles, try the reverse order (shard, then compile) as R1b
before abandoning the rung — upstream recipes exist in both orders and the
interaction is exactly what is untested.

### R2 — compile the cross-entropy without tracing the model

**Change.** `trainer.compile_loss` cannot be used for this: it wraps
`_forward_loss`, which calls `self.model(input_ids)`
([trainer.py:477](../../src/core/trainer.py)), so under FSDP2 it is whole-model
compile and belongs to R3. R2 instead compiles **only** the loss computation —
the `F.cross_entropy` call and its reshapes — behind a new
`trainer.compile_ce` knob, **declared in `TrainerSchema` first** per the
typed-config policy.

**Targets.** The 230 ms eager softmax/CE at B1 (57 ms at B4) plus the fp32
log-softmax and bf16↔fp32 casts over the logits, which the single-GPU work
measured at ~500 ms/step combined when the loss sat outside the graph.

**Expected.** 1–2 points at B4. Small, but cheap and independent of R1.

### R3 — whole-model compile under FSDP2

**Change.** `+compile_model=true` or `trainer.compile_loss=true` with FSDP2
active, which requires FSDP2's pre-forward all-gather hooks to survive tracing:
`torch._dynamo.config.skip_fsdp_hooks = False` plus compiled autograd
(traceable FSDP2). This is the genuinely untested item flagged in
training_efficiency.md §6.3 and §8.2.

**Targets.** Whatever R1 and R2 leave: the embedding and head glue outside any
block, and cross-block fusion that per-block compile cannot see.

**Expected.** The residual few points — and possibly nothing, if per-block
compile already captures the fusible work.

**Risk.** Highest of the three. Known outcomes include graph breaks at the
hooks, silent recompiles, and compile times long enough to distort a 30-step
run. **Attempt only if R1+R2 leave more than ~2 points on the table**, and
time-box it: one working day, then record the outcome and stop.

---

## 3. Measurement protocol

Unchanged from [multigpu_efficiency.md](multigpu_efficiency.md) §6, which
exists so results stay comparable:

- Workload `configs/mfu_audit/h20.yaml` with the GPT-2-arch overrides
  (`GeluFeedForward`, `~...gate_fn`, `n_blocks=25`, `kv_heads=${common.q_heads}`),
  seq 2048, global batch 512, 30 steps, lr 2^-12, warmup 10.
- `gradient_accumulation_steps` = 32 / 16 / 8 for 1 / 2 / 4 GPUs, holding the
  micro-batch at 16. Do not change the micro-batch within this plan; a changed
  micro-batch is a separate best-fit row, not a like-for-like comparison.
- Submit with `~/submit_baseline.sh <ngpu> <ga>` on entropy, one job at a time.
- Reduce with `scripts/mfu_window_log.py --window 16 29` (the post-profiler
  steady state), kernels with `scripts/kprof_agg.py` / `kprof_table.py`,
  operators with `scripts/kprof_ops.py`, per-rank balance with
  `scripts/timers_by_rank.py`.
- **Noise bar: a delta under 3% needs a repeat run before it is believed.**
  Measured cv on the baselines was 0.07–0.38%, so a real rung should clear this
  easily; anything marginal gets repeated.

**Every rung reports a 1-GPU and a 4-GPU number.** The 1-GPU run is cheap and
isolates the compile win from any interaction with communication; the 4-GPU run
is the one that matters.

## 4. Numerics gates

Compile changes fuse kernels, so they are **not** bit-identical and must not be
gated as if they were. Follow the template that cleared the single-GPU compiled
loss (training_efficiency.md Part 5.2):

- First loss of a fresh model must land in the ~11.2 band (ln V + 0.4 at seq
  2048). Every baseline hit 11.125.
- A 300-step h16 overlay against the unchanged configuration, same seed,
  reporting mean and max |Δloss|. The single-GPU precedent was mean 0.0074,
  max 0.066, no divergence trend — treat that as the acceptance bar.
- Run the overlay **once**, after the last rung that is adopted, not per rung.

## 5. What this plan does not touch

Recorded so they are not silently folded in — each is separately measurable and
none is on the critical path to 45%:

| item | measured worth | why not now |
|---|---|---|
| batching gradient sync (sync only on the last micro-step) | **< 2%** | 405 of 432 reduce-scatters at B2 are avoidable, but ~70% of collective time already overlaps compute. The real argument is pushing NCCL off `RING_LL` onto a bandwidth protocol — worth testing *after* the compile work, when the step is shorter and communication is a larger share |
| `reshard_after_forward=False` | ≤ half the all-gathers | removes the backward re-gather at a memory cost; 55–57 GiB of 80 in use, so the memory exists |
| bf16 `reduce_dtype` | ~half the gradient traffic | the reduce-scatter is `_f32_` today; a numerics question, not a free win |
| micro-batch > 16 | unmeasured | changes ga, so it is a separate best-fit row |
| multi-node, world size > 4, non-NVLink | unmeasured | out of scope; nothing here licenses a claim about them |

## 6. Execution order and cost

| step | runs | cluster time | gate |
|---|---|---|---|
| R1 implement + local tests | — | — | pytest + zero recompile warnings in a short local run |
| R1 measure | 1 GPU, 4 GPUs | ~20 min | ≥ 42% at B4 → rungs 2–3 optional |
| R2 implement + measure | 1 GPU, 4 GPUs | ~20 min | any gain clearing the 3% bar |
| R3 (only if > ~2 points remain) | 1 GPU, 4 GPUs | ~1 day incl. debugging | time-boxed; record the outcome either way |
| numerics overlay on the adopted stack | 1 GPU, 300 steps | ~30 min | mean/max \|Δloss\| within the Part 5.2 precedent |
| 2-GPU confirmation of the final stack | 2 GPUs | ~10 min | fills the scaling curve |

Total if R1 and R2 suffice: **under two hours of cluster time.**

## 7. Pass criteria for the plan as a whole

1. **≥ 42% MFU at 4 GPUs** with FSDP2 (from 33.70%). Stretch: ≥ 43.7%, the
   arithmetic's prediction.
2. Scaling efficiency stays ≥ 95% from the new 1-GPU FSDP2 anchor — i.e. the
   compile win does not come at the cost of overlap.
3. Zero dynamo recompile warnings in every adopted configuration.
4. Numerics overlay within the established precedent.
5. `multigpu_efficiency.md` carries the new rows, and §6.3 of
   training_efficiency.md is corrected — it currently tells readers that FSDP2
   runs keep `compile_blocks` because whole-model compile is untested, which
   will be out of date the moment R1 lands.

## 8. What would falsify the premise

Worth stating so the plan can fail honestly rather than be rationalised:

- **R1 lands under 2 points.** Then the eager glue is not where the launch
  counts say it is, or compiling `Residual` reintroduces graph breaks that cost
  as much as they save. Re-profile before proceeding to R3.
- **R1 helps on 1 GPU but not on 4.** Then the glue was already overlapping
  with communication and the single-GPU decomposition does not transfer —
  interesting in itself, and a reason to stop.
- **Memory regresses past ~70 GiB/GPU.** Wider compiled regions hold more live
  activations; the single-GPU work saw whole-model compile *reduce* memory
  (66.9 → 60.5 GiB), but that is not guaranteed under sharding. A rung that
  forces a smaller micro-batch has changed the workload and its throughput
  number is not comparable.

## 9. Log

- **2026-11-03** — plan written off the B1/B2/B4 baselines.
- **2026-11-03 (same day)** — **executed; goal met at R1.**
  - **R1 adopted.** `+compile_blocks_full=true`: 4 GPUs 33.70% → **44.28%**
    (2,466 ms, 425k tok/s), 1 GPU 34.45% → **45.57%**, 2 GPUs **44.69%**. Peak
    memory −21%. Zero recompile warnings. Numerics clean (last-50 mean |Δloss|
    0.0063, exactly the baseline's final loss). The premise held: the eager glue
    was the `Residual` layer, and the `log_name` blocker really was obsolete.
  - **R2 rejected.** `trainer.compile_ce` bought ~1% step time and 3 GiB and
    cost ~1.6% loss quality, systematically (higher in 50/50 of the last 50
    steps). Knob kept, defaulted off, with the measurement beside it. Likely
    cause: eager cross-entropy upcasts to fp32 for the log-softmax; compiled
    standalone it stays in bf16, which over a 50,304-wide vocabulary is worse
    arithmetic.
  - **R3 not attempted**, per this plan's own criterion — 1.29 points remain
    between 1 and 4 GPUs and they are communication, not compile scope.
  - **Prediction check.** §1 predicted ~45% at 1 GPU and ~43.7% at 4; measured
    45.57% and 44.28%. The arithmetic was right, including that FSDP2's residual
    plumbing would mostly disappear — it predicted ~650 ms would survive at
    world size 1 and ~6 ms did, because the wider compiled region absorbs the
    glue around FSDP2's module boundaries too.
  - **Falsification check.** §8 listed "R1 lands under 2 points", "R1 helps on 1
    GPU but not on 4" and "memory regresses past ~70 GiB". None occurred; memory
    moved the other way, by −11.7 GiB/GPU.
