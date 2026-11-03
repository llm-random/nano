# Multi-GPU training efficiency: the running record (2026-11-03 → )

The multi-GPU counterpart of [training_efficiency.md](training_efficiency.md),
which took **one** H100 from 30.8% to 46.4% MFU at h20 and is the single-GPU
reference for everything here. Same conventions, same protocol, same noise
discipline; this file logs the multi-GPU work as it happens — plan, runs,
results, dead ends.

**Scope (set by Maciej, 2026-11-03):** benchmark and then improve MFU for
**FSDP2 pretraining of the GPT-2-architecture h20 model in our framework**, on
**2 and 4 H100s at entropy**, using the **same training hyperparameters as the
single-GPU benchmarks**. FSDP2 throughout — "fsdp" in this document always means
FSDP2 (`fully_shard`), never FSDP1.

Work happens on branch `multigpu-mfu` in the worktree
`/home/maciej/code/nano-multigpu`, off `pretraining` at `b9189d1c`.

---

## 1. The model under test

**GPT-2-arch h20** — the architecture nanoGPT runs, built inside our framework.
Chosen because it is the one shape where a *framework-vs-framework* comparison
is honest: our own h20 (GQA + SiLU-gated FF) and nanoGPT's h20-eq differ in
attention and FF, so any cross-framework number mixes architecture with
framework (see the Conventions glossary in training_efficiency.md).

| | value |
|---|---|
| blocks | 25 (`common.n_blocks=25`) |
| dmodel | 1280 |
| attention | **MHA** — 20 q heads, 20 kv heads (`common.kv_heads=${common.q_heads}`) |
| feed-forward | **2-matrix GELU MLP** (`src.core.model.GeluFeedForward`, no gate) |
| positions | RoPE (ours — deliberately kept; nanoGPT's learned positions are free) |
| params | 491M non-embedding |
| FLOPs/token | **4.122 GF** (measured from the live model) |

Single-GPU reference for exactly this model, from training_efficiency.md Part
4.4 (job 32064, **plain module + autocast bf16 + `compile_model` + `compile_loss`,
no FSDP2**):

| | step ms | tokens/s | MFU | peak mem | micro×ga |
|---|---:|---:|---:|---:|---:|
| GPT-2-arch h20, 1 GPU, no FSDP2 | 9,578 | 109,425 | 45.6% | 52.0 GiB | 16×32 |

That configuration cannot be used on 2 or 4 GPUs — without FSDP2 there is no
gradient synchronisation at all. The FSDP2-compatible compile mode is regional
`compile_blocks`, which is what the baselines below measure.

## 2. Workload — identical to the single-GPU benchmarks

Unchanged from `configs/mfu_audit/h20.yaml`: sequence length 2048, **global**
batch 512 sequences, 30 steps, cosine schedule with 10 warmup steps, lr 2^-12,
eval and checkpointing off, measurement window steps 10–29, bf16.
Tokens/step = 512 × 2047 = **1,048,064** (model-consumed).

`common.batch_size` is the *global* batch: `get_dataloader` hands each rank
`total_batch_size // world_size` sequences
([src/core/datasets.py:376](../../src/core/datasets.py)). To keep the
**per-forward micro-batch at 16** — the single-GPU audit's probed value — only
`gradient_accumulation_steps` changes:

| world size | sequences per rank | gradient_accumulation_steps | micro-batch | global batch |
|---:|---:|---:|---:|---:|
| 1 | 512 | 32 | 16 | 512 |
| 2 | 256 | 16 | 16 | 512 |
| 4 | 128 | 8 | 16 | 512 |

So the optimizer sees exactly the same 512-sequence step at every world size,
and the only deliberate difference is how many micro-steps each rank does. The
micro-batch ceiling must still be re-probed per world size (FSDP2 shards
parameters, gradients and optimizer state, freeing memory that may allow micro
32); per the single-GPU protocol a changed micro-batch is reported as a separate
best-fit row, never mixed into a like-for-like comparison.

## 3. Prerequisite: the MFU accounting fix

`processed_tokens` counts **global** tokens (`batch.numel() * WORLD_SIZE`) while
the MFU line divided by **one** GPU's peak — so a 2-GPU run would have reported
2× and a 4-GPU run 4× the true utilization. Flagged in training_efficiency.md
Part 8.1, fixed here before any multi-GPU run:
`_log_throughput` now builds the denominator as `peak_flops_bf16(device) ×
WORLD_SIZE`, i.e. the job's aggregate peak, making `train/mfu` the average
per-GPU utilization. `tokens_per_s` keeps its meaning (global tokens/s).
Guarded by `test_the_mfu_denominator_is_the_peak_of_every_gpu`
(world sizes 1/2/4) in [tests/test_mfu.py](../../tests/test_mfu.py).

At world size 1 the arithmetic is unchanged, so every number in
training_efficiency.md stands.

## 4. The hardware

entropy's `h100` partition is the single node **h100a**: 8×H100 80GB HBM3
(SXM — so the 989 TFLOP/s bf16 peak in `src/core/mfu.py` is the right basis),
256 CPUs, 1,007 GiB RAM. GPUs are connected **NV18** — 18 bonded NVLink lanes,
~900 GB/s — so every pair is full-bandwidth NVLink and FSDP2's all-gathers and
reduce-scatters run on the best interconnect this model will ever see. A
scaling result measured here is an upper bound for anything PCIe- or
network-bound; it says nothing about multi-node, which is out of scope.

Our QoS cap is 8 GPUs (`j321m_h100`), so 2- and 4-GPU jobs fit on one node.
The node is shared — other users' jobs run alongside, which is also true of
every single-GPU number in training_efficiency.md. Runs are submitted **one at
a time** so our own jobs never contend with each other.

## 5. Baselines

Three runs, same model, same workload, same commit; only the GPU count changes.
Configuration: FSDP2 (`trainer.distributed` from `configs/_trainer/llama.yaml`,
sharding embedding / block / head), regional compile (`+compile_blocks=true`),
no autocast knob (FSDP2's `MixedPrecisionPolicy` already computes in bf16 and
reduces in fp32), `trainer.profile_timers=true` and
`trainer.kernel_profile=true` on.

| # | world size | gres | ga | job | step ms (cv) | tokens/s | MFU | peak mem/GPU |
|---|---:|---|---:|---|---:|---:|---:|---:|
| B1 | 1 | `gpu:h100:1` | 32 | 32264 | **12,678** (0.15%) | **82,668** | **34.45%** | 62.0 GiB |
| B2 | 2 | `gpu:h100:2` | 16 | 32267 | **6,447** (0.07%) | **162,555** | **33.87%** | 57.4 GiB |
| B4 | 4 | `gpu:h100:4` | 8 | 32269 | **3,240** (0.17%) | **323,440** | **33.70%** | 55.1 GiB |

Ideal (linear) scaling from B1, for reading the rows above: 6,339 ms /
165,336 tok/s at 2 GPUs and 3,170 ms / 330,672 tok/s at 4 — MFU staying at
34.45%, since MFU is already per-GPU normalized. Any shortfall is exposed
communication.

**Two different "1 GPU" numbers — do not confuse them.** This framework reaches
**45.6% MFU** on one H100 for this model, but only in a configuration that
cannot be used on several (no FSDP2 at all; see §1). The multi-GPU-capable
configuration reaches **34.45%** on the same one GPU. Both are real; they
measure different things:

| configuration | 1 GPU | usable on 2+ GPUs? |
|---|---:|---|
| no FSDP2, whole-model compile + compiled loss | **45.6%** (9,578 ms, 52.0 GiB) | **no** — without FSDP2 there is no gradient synchronisation |
| FSDP2 + `compile_blocks` | **34.45%** (12,678 ms, 62.0 GiB) | yes — B1/B2/B4 |

So the deficit a multi-GPU run carries splits into two unrelated parts, and
only the second is about parallelism:

- **45.6% → 34.45% (−11.2 points, −24% relative): compile scope.** Paid in full
  at world size 1, before a second GPU exists. It is the ~3.9 s/step of eager
  glue that regional compile leaves outside the graph (§7.1), and it is the
  single largest number in this document.
- **34.45% → whatever the multi-GPU rows show: communication.** Measured at
  −1.7% relative for B2.

Comparisons in §7 are FSDP2-to-FSDP2 (B1 → B2 → B4) because that isolates the
cost of parallelism. Whenever a row is compared to 45.6% instead, it is said
explicitly.

B1 is not redundant with the 45.6% reference: that run had no FSDP2 and used
whole-model compile plus a compiled loss. B1 is the *same-commit, same-node*
FSDP2 + `compile_blocks` anchor, so the 1→2→4 scaling curve measures
communication rather than two weeks of code drift.

## 6. How a run is measured

No wandb key exists on entropy (`~/.wandb_key` is absent, which is why the
single-GPU GPT-2-arch run logged to stdout too), so runs use
`infrastructure.metric_logger.type=stdout` and are reduced from the job log by
[scripts/mfu_window_log.py](../../scripts/mfu_window_log.py) — the log-reading
twin of `scripts/mfu_window.py`. It reads **rank 0 only**; every rank prints the
same metric names and averaging them together would be meaningless.

**Window: steps 16–29, not 10–29.** With `kernel_profile=true` the profiler
fires at step 14 and the step after it is still elevated, so the usual 10–29
window mixes in two distorted steps. Measured on the single-GPU reference run
(job 32064): steady state is 9.57 s/step, step 14 is 24.5 s and step 15 is
11.2 s, so the 10–29 mean reads 10,408 ms at cv 32% while 16–29 reads
**9,573 ms at cv 0.13%**. The published figure for that run is 9,578 ms, so
16–29 is the window its headline numbers were taken over, and reproducing it to
0.06% is what validates this reducer:

| reducer on job 32064 | step ms | cv | tokens/s | MFU |
|---|---:|---:|---:|---:|
| window 10–29 (profiler steps included) | 10,408 | 32% | 105,265 | 43.87% |
| window 15–29 (Part 4.5's uniform reducer) | 9,683 | 4.4% | 108,408 | 45.18% |
| **window 16–29 (used here)** | **9,573** | **0.13%** | **109,486** | **45.63%** |
| published (training_efficiency.md Part 4.4) | 9,578 | — | 109,425 | 45.6% |

Everything else follows the single-GPU protocol (training_efficiency.md Part 7):
3% noise bar before a delta is believed, identical windows on both sides of any
comparison, a changed micro-batch reported as a separate best-fit row.

## 7. Results

### 7.1 What FSDP2 costs before any communication happens (B1, world size 1)

B1's kernel profile, next to the single-GPU optimum for the same model from
training_efficiency.md Part 4.4 (plain module, `compile_model` + `compile_loss`,
no FSDP2). Both at h20 GPT-2 arch, 16×32, same node; ms/step:

| kernel category | B1: FSDP2 + `compile_blocks` | reference: no FSDP2 + whole-model compile |
|---|---:|---:|
| GEMMs (nvjet) | 4,693 | 4,711 |
| flash attention | 2,263 | 2,418 |
| compiled (triton) | 843 | 1,982 |
| **eager elementwise** | **3,674** | 345 |
| **eager norm pieces** | **549** | 0 |
| **softmax/CE (eager)** | **230** | 0 |
| optimizer | 351 | 10 |
| cat / copies / memcpy | 198 | 0.2 |
| other (FSDP2 plumbing, see below) | 295 | 9 |
| **TOTAL busy** | **13,095** | **9,475** |

Two separable costs, and the compute itself is not one of them — GEMMs are
identical (4,693 vs 4,711) and flash attention is *faster* under FSDP2:

1. **Regional compile leaves the glue eager — about 3.9 s/step.** Eager
   elementwise, eager norms and eager cross-entropy (3,674 + 549 + 230 = 4,453)
   replace what whole-model compile fuses into triton (345 + 0 + 0 = 345, with
   triton correspondingly larger). This is the same gap Part 4.2 measured on one
   GPU, and it is the price of `compile_blocks` being the FSDP2-documented
   pairing. It is not a multi-GPU cost at all — it is the compile-scope question
   the multi-GPU effort inherits.
2. **FSDP2's own plumbing — about 255 ms/step even with nothing to communicate.**
   In the `other` row: an NCCL **one-rank** kernel (117 ms, 864 launches/step),
   `split_with_sizes_copy_out` (72 ms, 1,728) and `chunk_cat` (66 ms, 864, its
   template showing the bf16→fp32 cast for the fp32 reduce). At world size 1
   these move no data between devices; they are the all-gather/reduce-scatter
   machinery running against itself.

**The launch count names the gradient-sync problem exactly.** 864 = **27 × 32**:
27 sharded module groups (25 blocks + embedding + head) × 32 micro-steps. Every
micro-step runs a collective per sharded module, when only the last micro-step
of the accumulation needs to. That is finding F2 from the survey, now measured
rather than inferred — and at world size 1 it is nearly free, because the
"collective" is a local copy. At world size 2 and 4 the same 27-per-micro-step
pattern becomes real NVLink traffic: 432 and 216 collectives per optimizer step
where 27 would do.

Section timers at step 19 confirm where the 32% sits — entirely inside the model,
in equal proportion, with everything around it at parity (ms):

| section | reference (no FSDP2, whole compile) | B1 (FSDP2 + `compile_blocks`) | ratio |
|---|---:|---:|---:|
| forward (32 micros) | 2,948.8 | 3,895.2 | 1.32× |
| backward (32 micros) | 6,596.7 | 8,747.8 | 1.33× |
| grad clip | 2.8 | 2.8 | 1.00× |
| optimizer step | 8.8 | 11.5 | 1.31× |
| metric logging | 136.3 | 133.5 | 0.98× |
| data (fetch + H2D) | 0.7 | 0.7 | — |
| **step total** | **9,561.2** | **12,661.9** | **1.32×** |

(The ~134 ms/step in `log` is present identically on both sides, so it cancels
in every comparison here; it is ~1.4% of a step and nobody has looked at it.)

### 7.2 Prediction, written before B2 and B4 landed

Worth recording in advance so the measurement can contradict it. The model is
620.4M params; FSDP2 all-gathers parameters in bf16 (1.24 GB whole) and
reduce-scatters gradients in **fp32** (2.48 GB whole, per
`MixedPrecisionPolicy(reduce_dtype=fp32)`). Each rank moves (N−1)/N of that.
With `reshard_after_forward` at its default the parameters are gathered again
in backward, so per optimizer step:

| | all-gather | reduce-scatter | total | at ~400 GB/s |
|---|---:|---:|---:|---:|
| 2 GPUs, ga 16 | 19.8 GB | 19.8 GB | 39.7 GB | ~99 ms |
| 4 GPUs, ga 8 | 14.9 GB | 14.9 GB | 29.8 GB | ~75 ms |

Against ideal step times of 6,339 and 3,170 ms that is 1.6% and 2.4% — **small**,
because NVLink is fast and the model is not large. The interesting consequence:
**batching the gradient sync would save less bandwidth than expected**
(reduce-scatter would fall to 1.24 GB / 1.86 GB, worth only ~47 ms and ~33 ms),
so if the per-micro-step sync hurts, it should hurt through *launch and copy
overhead* rather than bytes moved — at world size 1 the plumbing cost 255 ms
across 864 launches, about 0.3 ms each, which would put 432 launches at ~128 ms
and 216 at ~64 ms.

So the prediction is: **scaling close to linear, with any shortfall dominated by
per-collective overhead, not bandwidth.** If B2/B4 fall well short of that, the
cause is something this model does not capture — exposed (non-overlapped)
communication, a straggler, or the data pipeline failing to feed N ranks.

### 7.3 B2 — two GPUs

**6,447 ms/step, 162,555 tok/s, 33.87% MFU, 57.4 GiB/GPU, cv 0.07%** (job
32267). Against B1's ideal of 6,339 ms that is a 108 ms shortfall —
**98.3% scaling efficiency**, with per-GPU MFU falling 34.45% → 33.87%, i.e.
parallelism costs 1.7%. Peak memory per GPU drops 62.0 → 57.4 GiB, which is
sharding doing its job.

Stated against the framework's best single-GPU result for this model (45.6%),
the same run is **26% down** — but §5 shows 24 of those points were already
lost at world size 1 to compile scope. The 1.7% is the multi-GPU part.

The two ranks are indistinguishable (mean over steps 16–29, ms):

| rank | forward | backward | clip | optim | log | step total |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1,975.0 | 4,460.2 | 1.5 | 7.1 | 139.4 | 6,447.2 |
| 1 | 1,967.7 | 4,467.5 | 1.5 | 7.2 | 139.7 | 6,447.2 |

Spread 0.0 ms — no straggler, as expected when every rank runs identical shapes.

**Every kernel category halves except communication.** Rank 0, ms/step, against
B1 (which did twice the micro-steps):

| category | B1 (1 GPU) | B2 (2 GPUs) | ratio |
|---|---:|---:|---:|
| GEMMs | 4,693 | 2,413 | 0.51 |
| eager elementwise | 3,674 | 1,789 | 0.49 |
| flash attention | 2,263 | 1,152 | 0.51 |
| compiled (triton) | 843 | 424 | 0.50 |
| eager norms | 549 | 283 | 0.52 |
| softmax/CE | 230 | 115 | 0.50 |
| optimizer | 351 | 111 | 0.32 |
| cat/copy/memcpy | 198 | 98 | 0.49 |
| **other (collectives + plumbing)** | **295** | **397** | **1.35** |
| TOTAL busy | 13,095 | 6,782 | 0.52 |

The `other` row is the only one that grows, and at 2 GPUs it resolves into real
NCCL kernels:

| kernel | ms/step | launches/step | what the count means |
|---|---:|---:|---|
| `ncclDevKernel_AllGather_RING_LL` | 188.9 | 864 | 27 modules × 16 micro-steps × 2 (forward, then again in backward because `reshard_after_forward` is on) |
| `ncclDevKernel_ReduceScatter_Sum_**f32**_RING_LL` | 123.5 | 432 | 27 modules × 16 micro-steps — **27 would suffice** |
| `chunk_cat` (bf16→fp32 for the reduce) | 33.3 | 432 | |
| `split_with_sizes_copy_out` | 33.1 | 864 | |

Three things worth keeping:

1. **Overlap is doing most of the work.** 312 ms of NCCL kernel time plus 66 ms
   of copy plumbing costs only 108 ms of wall clock, so roughly 70% of
   communication hides behind compute. Note `TOTAL busy` (6,782) exceeds the
   wall step (6,447): collectives run on their own streams.
2. **The prediction in §7.2 was right for partly the wrong reason.** It called
   ~1.6% shortfall and measured 1.7% — but it estimated ~99 ms of communication
   where the kernels actually take 312 ms. The error is the protocol: NCCL picked
   **RING_LL**, the low-latency protocol, which buys latency with bandwidth
   because the 27-per-micro-step pattern produces many small messages. Overlap
   covered the difference. Bandwidth modelling alone would have mispredicted.
3. **F2 is confirmed and quantified at this scale.** 432 reduce-scatters where
   27 are needed: **405 are avoidable**, ~116 ms of kernel time. But since most
   of it overlaps, batching the sync is worth *at most* the ~108 ms the step is
   actually losing — not the 116 ms the kernel total suggests. The stronger
   argument for batching is the protocol: fewer, larger messages should move NCCL
   off LL and raise achieved bandwidth.

### 7.4 B4 — four GPUs

**3,240 ms/step, 323,440 tok/s, 33.70% MFU, 55.1 GiB/GPU, cv 0.17%** (job
32269). Ideal from B1 is 3,170 ms, so the shortfall is 70 ms —
**97.8% scaling efficiency**, per-GPU MFU 34.45% → 33.70% (−2.2%).

Ranks remain indistinguishable; spread 0.2 ms across four (means, steps 16–29):

| rank | forward | backward | clip | optim | log | step total |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 989.8 | 2,241.7 | 0.8 | 5.1 | 140.7 | 3,240.2 |
| 1 | 985.8 | 2,245.4 | 0.8 | 5.3 | 141.2 | 3,240.1 |
| 2 | 987.9 | 2,243.1 | 0.8 | 5.4 | 140.8 | 3,240.0 |
| 3 | 985.6 | 2,245.5 | 0.8 | 5.3 | 141.5 | 3,240.2 |

Collectives: 432 all-gathers (165.8 ms) and 216 fp32 reduce-scatters (94.4 ms),
again 27 modules × 8 micro-steps (×2 for the backward re-gather), plus 33 ms of
chunk_cat/split plumbing. 260 ms of NCCL kernel time costs 70 ms of wall clock —
about 73% overlapped, the same story as B2.

### 7.5 The scaling curve

| world size | step ms | tokens/s | MFU | scaling eff. | peak mem/GPU | comm (NCCL) ms | comm as % of busy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12,678 | 82,668 | 34.45% | — | 62.0 GiB | 117¹ | 2.3% |
| 2 | 6,447 | 162,555 | 33.87% | **98.3%** | 57.4 GiB | 312 | 5.9% |
| 4 | 3,240 | 323,440 | 33.70% | **97.8%** | 55.1 GiB | 260 | 8.7% |

¹ world size 1 has no inter-device traffic; this is NCCL's one-rank local-copy
kernel, included so the row is comparable.

**FSDP2 scales almost perfectly on this node.** 1.97× at two GPUs and 3.91× at
four, with per-GPU MFU losing 0.6 and 0.8 points. Ranks stay balanced to within
0.2 ms, the data pipeline keeps up at four ranks (`data_fetch` is 0.1–0.2 ms —
the worry in training_efficiency.md §8.2 that 2 workers would not feed a
multi-GPU run does not bite at this scale), and memory per GPU falls as
sharding spreads parameters, gradients and optimizer state.

**The one trend that does not improve is communication.** Everything else
quarters from B1 to B4; collective time stays roughly flat in absolute terms
(295 → 397 → 304 ms of the `other` row) and therefore grows as a share of the
step: 2.3% → 5.9% → 8.7%. Extrapolating the same shape, 8 GPUs on one node
would be the first world size where it starts to bite, and anything off NVLink
would bite much sooner.

## 8. Where that leaves the effort

The baselines answer the question they were set, and they reorder the work.

**Multi-GPU is not where this model is losing MFU.** Parallelism costs 1.7% at
two GPUs and 2.2% at four. The deficit against the framework's own single-GPU
best (45.6% → ~33.8%) is **~24% relative, and all of it is paid at world size 1**
by using regional `compile_blocks` instead of whole-model compile plus a
compiled loss (§5, §7.1). On these measurements:

| candidate | measured worth | note |
|---|---|---|
| **whole-model compile + compiled loss under FSDP2** | **~11 MFU points** | the entire 45.6 → 34.45 gap; untested under FSDP2, which is the only reason we are not taking it |
| batching gradient sync (F2) | **< 110 ms/step, under 2%** | 405 of 432 reduce-scatters at B2 are avoidable, but ~70% of collective time already overlaps compute; the better argument is pushing NCCL off the RING_**LL** low-latency protocol onto a bandwidth protocol by sending fewer, larger messages |
| `reshard_after_forward` tuning | ≤ the all-gather half of comm | would remove the backward re-gather (half of 864/432 all-gathers) at a memory cost; memory is available — 55–57 GiB of 80 in use |
| micro-batch ≥ 32 | unmeasured | 55.1 GiB/GPU at B4 leaves headroom the single-GPU probe did not have; it changes ga, so it is a separate best-fit row, not a like-for-like one |

This inverts the ranking the survey started from, where per-micro-step gradient
sync was called the "top multi-GPU throughput suspect" (training_efficiency.md
§8.1). It is real, it is visible in the launch counts, and it is worth less than
2% — because NVLink is fast and FSDP2 overlaps well. **The compile-scope
question is the whole game.**

What has *not* been established, and should not be assumed from these runs:
multi-node (never run), any world size above 4, non-NVLink interconnects, a real
multi-GPU checkpoint round-trip, and resume across a changed GPU count (the
dataloader shards per rank, so a run is reproducible at fixed (seed, world_size)
only — see training_efficiency.md §8.2).

## 9. Log

- **2026-11-03** — baselines B1/B2/B4 complete (jobs 32264 / 32267 / 32269), all
  exit 0. FSDP2 scales at 98.3% / 97.8%; the multi-GPU deficit is dominated by
  compile scope, not parallelism (§8). Scope set; worktree `multigpu-mfu` created off `pretraining`.
  MFU accounting fixed (§3) with a world-size-parametrized regression test.
  `GeluFeedForward` cherry-picked from `mfu-abl/nanogpt-arch` (`9f3e7421`) — it
  had never been merged, so the GPT-2 architecture was not buildable on
  `pretraining`. Entropy checked: h100a is one node with 8×H100, our QoS cap is
  8 GPUs, so 2- and 4-GPU jobs are allocatable on a single node (multi-node is
  out of scope for now).
