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

## 0. Status report — where FSDP2 stands (2026-11-03)

**Multi-GPU training works, scales nearly linearly, and is no longer slower per
GPU than the best single-GPU configuration.** Before this round, using FSDP2
cost ~11 MFU points on every GPU; it now costs essentially nothing, and the
FSDP2 path is the fastest configuration this framework has, at any world size.

### How to run it

```
+compile_blocks_full=true          # top-level key, so it keeps the '+'
```
with FSDP2 as configured in `configs/_trainer/llama.yaml` (`trainer.distributed`
sharding embedding / block / head). Nothing else is needed; `autocast_bf16` is
not used, because FSDP2's `MixedPrecisionPolicy` already computes in bf16 and
reduces in fp32. Launch with `torchrun --nproc-per-node=<N>`; `common.batch_size`
is the **global** batch, so only `gradient_accumulation_steps` changes with the
GPU count.

### Measured, GPT-2-arch h20, seq 2048, global batch 512, entropy H100s

| world size | step ms (cv) | tokens/s | MFU | peak mem/GPU | scaling |
|---:|---:|---:|---:|---:|---:|
| 1 | 9,584 (0.21%) | 109,361 | **45.57%** | 50.3 GiB | — |
| 2 | 4,886 (0.13%) | 214,499 | **44.69%** | 45.7 GiB | 98.1% |
| 4 | 2,466 (0.25%) | 425,054 | **44.28%** | 43.4 GiB | 97.2% |

Four H100s return **3.89×** one H100's best. Because the adopted configuration
is also the best single-GPU one, that scaling figure is the real speedup, not a
comparison against a handicapped baseline.

### What is verified

- **Correctness of the metric.** MFU divided global throughput by *one* GPU's
  peak, so multi-GPU runs would have reported 2× and 4× too high. Fixed and
  regression-tested across world sizes 1/2/4 (§3).
- **Numerics.** 300-step overlay against the previous configuration: last-50
  mean |Δloss| 0.0063 — a fifth of a bf16 ulp — no systematic sign, and exactly
  the same final loss (§7.9).
- **Compile health.** Zero dynamo recompile warnings in every run. This matters
  because the failure mode is silent: it looks like "compile didn't help here".
- **Load balance.** Ranks agree to 0.2 ms at 4 GPUs; no straggler.
- **Data pipeline.** `data_fetch` is 0.1–0.2 ms at 4 ranks, so the long-standing
  worry that 2 dataloader workers could not feed a multi-GPU run does not bite
  at this scale (it may still bite at 8+).

### What is known-bad

- **`trainer.compile_ce` — do not enable.** It buys ~1% step time and 3 GiB and
  shifts the loss systematically upward (higher in 50 of the last 50 steps).
  Knob kept, defaulted off, measurement recorded beside it (§7.9).

### What is untested — do not assume

| area | state |
|---|---|
| multi-node | **never run.** The torchrun template supports `--nnodes`, nothing has exercised it |
| world size > 4 | never run. Communication is flat in absolute terms and so grows as a share of the step (2.3% → 5.9% → 8.7% at 1/2/4); 8 GPUs is where it should start to bite |
| non-NVLink interconnect | everything here is one node with NV18 full NVLink. These numbers are a best case |
| multi-GPU checkpointing | a real sharded dcp round-trip has never been run at world size > 1 |
| resume across a changed GPU count | **unsafe.** The dataloader shards per rank and the checkpoint does not record `world_size`, so a run is reproducible only at fixed (seed, world_size) |
| FSDP1 | untouched and unmeasured; it remains a legacy fallback |

### Open, ranked by measured worth

1. **micro-batch ≥ 32** — R1 freed 11.7 GiB/GPU and micro 32 has been OOM-bound
   since job 32024. Newly plausible, unmeasured.
2. **`reshard_after_forward=False`** — would remove the backward re-gather, half
   of all all-gathers, for memory we now have spare.
3. **batching gradient sync** — real but worth < 2%; the stronger argument is
   pushing NCCL off its `RING_LL` low-latency protocol by sending fewer, larger
   messages.
4. **bf16 `reduce_dtype`** — the reduce-scatter runs in fp32 today, doubling
   gradient traffic. A numerics question, not a free win.

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

### 7.6 Granular profile, all three runs side by side

Both instrumentation knobs were on for every baseline, so each run carries
per-section CUDA-event timings (every step) and a full per-kernel dump (steps
12–13). Raw logs: `~/mgpu_{32264,32267,32269}.log` on entropy.

**Section timers** — mean over steps 16–29, rank 0, ms:

| section | B1 (1 GPU) | B2 (2 GPUs) | B4 (4 GPUs) |
|---|---:|---:|---:|
| forward | 3,901.1 | 1,975.0 | 989.8 |
| backward | 8,758.4 | 4,460.2 | 2,241.7 |
| grad clip | 2.8 | 1.5 | 0.8 |
| optimizer | 11.1 | 7.1 | 5.1 |
| metric logging | 133.5 | 139.4 | 140.7 |
| data fetch + H2D | 0.7 | 0.4 | 0.3 |
| **step total** | **12,677.8** | **6,447.2** | **3,240.2** |

`log` is flat in absolute terms because it is a `.item()` synchronisation point,
not real logging work; it therefore grows as a share of a shrinking step
(1.1% → 2.2% → 4.3%) and is the one loop-level item worth revisiting if step
times fall much further.

**Per micro-step** — each rank runs the same 16-sequence micro-batch at every
world size (ga 32/16/8), so anything not flat here is a scaling cost:

| | B1 | B2 | B4 |
|---|---:|---:|---:|
| forward / micro-step | 121.9 | 123.4 | 123.7 |
| backward / micro-step | 273.7 | 278.8 | 280.2 |
| total busy / micro-step | 409.2 | 423.9 | 434.6 |
| **busy minus collectives / micro-step** | **400.0** | **399.0** | **396.6** |
| collectives / micro-step | 9.2 | 24.8 | 38.0 |

**Compute per micro-step is invariant to 0.9%**, and every bit of the scaling
loss is the collectives row. This is the cleanest form of the result: FSDP2
costs nothing in arithmetic, only in communication.

**Kernel categories** — ms/step, rank 0, against the ideal B4/B1 ratio of 0.25:

| category | B1 | B2 | B4 | B4/B1 |
|---|---:|---:|---:|---:|
| GEMMs (nvjet) | 4,693.4 | 2,412.7 | 1,216.9 | 0.259 |
| eager elementwise | 3,674.1 | 1,789.0 | 883.4 | 0.240 |
| flash attention | 2,262.6 | 1,151.8 | 574.7 | 0.254 |
| compiled (triton) | 842.8 | 424.3 | 212.5 | 0.252 |
| eager norms | 548.6 | 283.0 | 141.5 | 0.258 |
| softmax/CE (eager) | 229.5 | 114.8 | 57.4 | 0.250 |
| optimizer | 351.1 | 111.4 | 37.5 | 0.107 |
| cat/copy/memcpy | 198.2 | 97.7 | 48.7 | 0.246 |
| **collectives + plumbing** | **294.9** | **397.2** | **303.9** | **1.031** |
| **TOTAL busy** | **13,095.3** | **6,781.8** | **3,476.5** | 0.266 |

**Collectives opened up** — the `other` row per run:

| run | kernel | ms/step | launches | count is |
|---|---|---:|---:|---|
| B1 | NCCL one-rank (local copy, no traffic) | 117.1 | 864 | 27 × 32 |
| | `split_with_sizes_copy_out` | 72.4 | 1,728 | |
| | `chunk_cat` (bf16→fp32 for the reduce) | 65.8 | 864 | |
| B2 | `AllGather_RING_LL` | 188.9 | 864 | 27 × 16 × 2 |
| | `ReduceScatter_Sum_**f32**_RING_LL` | 123.5 | 432 | 27 × 16 |
| | `chunk_cat` + `split_with_sizes_copy_out` | 66.4 | 1,296 | |
| B4 | `AllGather_RING_LL` | 165.8 | 432 | 27 × 8 × 2 |
| | `ReduceScatter_Sum_**f32**_RING_LL` | 94.4 | 216 | 27 × 8 |
| | `chunk_cat` + `split_with_sizes_copy_out` | 33.0 | 648 | |

Three details in there matter for what to change next: the reduce-scatter is
**f32**, so `MixedPrecisionPolicy(reduce_dtype=fp32)` doubles gradient traffic
against a bf16 reduce; NCCL selects **RING_LL** at every world size, its
latency-optimised protocol that gives up bandwidth, because 27 collectives per
micro-step makes every message small; and the all-gather count carries a ×2 for
the backward re-gather, which is what `reshard_after_forward` buys memory with.

### 7.7 R1 — widening the compile unit (2026-11-03)

Executed from [multigpu_compile_scope_plan.md](multigpu_compile_scope_plan.md).
`+compile_blocks_full=true` compiles each whole `TransformerBlock` instead of
only its inner attention and ff modules, so the two `Residual` wrappers — the
RMSNorm, the `out + x` add and the boundary casts — are traced rather than left
eager between two compiled regions.

| | 1 GPU | 4 GPUs |
|---|---:|---:|
| `compile_blocks` (baseline) | 12,678 ms · 34.45% · 62.0 GiB | 3,240 ms · 33.70% · 55.1 GiB |
| **`compile_blocks_full` (R1)** | **9,584 ms · 45.57% · 50.3 GiB** | **2,466 ms · 44.28% · 43.4 GiB** |
| delta | **+11.1 points** | **+10.6 points** |

Jobs 32289 (1 GPU) and 32287 (4 GPUs); cv 0.21% and 0.25%; **zero dynamo
recompile warnings** in both, and all ranks logged "torch.compile applied to 25
whole blocks". Loss trajectories match the baselines step for step
(7.6875 → 7.5000 at 4 GPUs), which is the first numerics signal; the formal
overlay comes once the stack is final.

**The single-GPU FSDP2 penalty is gone.** R1 at world size 1 scores 9,584 ms /
45.57% against the non-FSDP2 optimum's 9,578 ms / 45.6% — a 0.06% difference,
inside noise — and uses *less* memory (50.3 vs 52.0 GiB). The decomposition in
§7.1 predicted ~650 ms of irreducible FSDP2 overhead would survive; the
measurement says ~6 ms. Widening the compiled region evidently absorbs the glue
around FSDP2's module boundaries as well as the Residual layer's own.

That collapses the two framings §5 had to keep apart: the best single-GPU
configuration **is now an FSDP2 configuration**, so scaling efficiency and
practical speedup are the same number. From the new 1-GPU anchor, 4 GPUs give
425,054 / 109,361 = **3.89×, i.e. 97.2%** — against 2.96× before R1.

**Memory fell rather than rose.** The plan listed a memory regression as a
plausible failure mode (wider compiled regions hold more live activations);
instead peak dropped 11.7 GiB/GPU at 4 GPUs and 11.7 GiB at 1 GPU, mirroring
what whole-model compile did on one GPU in training_efficiency.md §4.3. Micro
32 at h20, OOM-bound since job 32024, is plausibly back in reach.

**What remains.** 45.57% (1 GPU) − 44.28% (4 GPUs) = **1.29 points**, which is
communication and nothing else. The plan's criterion for attempting R3
(whole-model compile under FSDP2) was "more than ~2 points remain", so **R3 is
ruled out by its own test** — there is no longer a compile-scope gap to close.

**A measurement caveat worth recording.** R1's first 4-GPU run (job 32285) read
2,625 ms at **cv 13.3%**: steady state was a stable 2.465 s but steps 18, 19 and
21 spiked to 3.54/3.29/2.78 s while another user's jobs ran on the node. The
repeat (32287) gave 2,466 ms at cv 0.25%. Discarding slow steps would have been
cherry-picking, so the repeat is what is reported — and this is what the
protocol's repeat rule is for on a shared node.

### 7.8 R2 — compiling the cross-entropy (2026-11-03)

`trainer.compile_ce=true` on top of R1. `trainer.compile_loss` could not serve
here: it wraps `_forward_loss`, which calls `self.model(...)`, so under FSDP2 it
is whole-model compile and belongs to R3. R2 compiles the cross-entropy alone
and never touches the model, so it composes with regional block compile instead
of replacing it.

| run | world size | step ms (cv) | tokens/s | MFU | peak mem/GPU |
|---|---:|---:|---:|---:|---:|
| R1 (32287) | 4 | 2,466 (0.25%) | 425,054 | 44.28% | 43.4 GiB |
| R2 (32294) | 4 | 2,437 (0.25%) | 429,997 | 44.80% | 40.3 GiB |
| R2 repeat (32296) | 4 | 2,441 (0.24%) | 429,398 | 44.74% | 40.3 GiB |
| **R2 (32299)** | **1** | **9,474 (0.25%)** | **110,629** | **46.10%** | **47.3 GiB** |

Zero recompile warnings throughout; loss trajectories unchanged.

**Reported honestly: the gain is below the pre-registered bar but reproducible.**
R2 improves on R1 by 1.06% in step time, under the protocol's 3% threshold. The
two R2 runs agree to 0.16%, a spread six times smaller than the effect, so the
difference is real rather than node noise. The 3% bar was calibrated against
runs with 0.6–1.6% cv (training_efficiency.md §4.5) and is conservative at the
0.24–0.25% these runs achieve — but it did its job, because it is what forced
the repeat instead of a claim off one number.

The memory saving has a clear mechanism: compiling the cross-entropy keeps the
logits — batch × seq × vocab, the largest activation in the step — from being
materialised again in fp32 at a graph boundary. That same mechanism is the
warning sign in hindsight: **fp32 around the logits is not waste, it is the
precision the log-softmax needs**, which is exactly what §7.10 then measured.

> **These throughput numbers stand but the configuration was not adopted.**
> R2 was provisionally taken on throughput grounds and then **rejected by the
> numerics gate** — see §7.10. The adopted stack is R1 alone (§7.11). The
> section is kept as measured, because a rejected rung is a result.

### 7.9 The numerics gate rejects R2

The 300-step overlay was run at 4 GPUs, h20 GPT-2 arch, same seed, profiling
off, against the pre-R1 production configuration. The adopted stack failed it:
mean |Δloss| 0.0704, max 0.1562, last-50 mean 0.0912 (1.61% of final CE) —
roughly 10× the precedent that cleared the single-GPU compiled loss
(training_efficiency.md §5.2: mean 0.0074, max 0.066, last-50 0.0036) — and the
difference was **systematic**: the stack's loss was higher in 50 of 50 steps
across the last three 50-step buckets, settling at a stable +0.09. Logged
losses are bf16-quantized in 0.03125 steps, so that is 3 ulps always in the
same direction; a random walk would cross zero.

Because the stack changed two things at once, a 2×2 isolated them. All four
arms are 300 steps, 4 GPUs, same seed, compared against `old`:

| arm | config | mean \|Δ\| | max \|Δ\| | last-50 | higher in last 50 | final loss |
|---|---|---:|---:|---:|---:|---:|
| **r1only** (32307) | `compile_blocks_full` | **0.0093** | 0.0938 | **0.0063** | 0/50 | **5.5625** |
| r2only (32317) | `compile_blocks` + `compile_ce` | 0.0875 | 0.2188 | 0.1100 | **50/50** | 5.6562 |
| new (32302) | both | 0.0704 | 0.1562 | 0.0912 | **50/50** | 5.6562 |
| old (32303) | `compile_blocks` | — | — | — | — | 5.5625 |

**R1 is clean and R2 is the cause.** R1's last-50 mean of 0.0063 is a fifth of
one bf16 ulp, it shows no systematic sign, and it finishes on exactly the
baseline's final loss. The first hypothesis — that widening the compiled region
moved the residual stream and norms from fp32 to bf16, which would have
explained the speed, the 11.7 GiB memory drop and a worse loss together — is
**wrong**, and the isolation is what caught it.

`compile_ce` shifts the loss on its own, with or without R1. **Rejected**; the
knob stays in the code, defaulted off, with the measurement recorded next to it
so nobody re-tries it blind.

**What is established, and what is not.** `torch.compile` preserves semantics in
exact arithmetic, not in floating point: inductor may fuse, reassociate
reductions and choose its own intermediate dtypes, and floating-point addition
is not associative. That normally shows up as *unbiased* noise — which is
exactly what R1 shows. A one-directional shift across 50 consecutive steps is
not reassociation noise, and that much is solid.

The *mechanism* is a leading hypothesis, not a measured fact: eager
`F.cross_entropy` dispatches to a kernel that accumulates the log-sum-exp in
fp32 even for bf16 inputs, because summing 50,304 exponentials in bf16 is poor;
inductor generates its own kernel and is not obliged to keep that upcast.
Confirming it means reading the generated Triton kernel's dtypes, which nobody
has done.

**And one thing these runs cannot distinguish**: whether the *gradients* are
degraded (the model genuinely trains worse) or only the *reported* number is
biased (training fine, metric wrong). A growing offset does not settle it —
bf16 log-sum-exp error grows as logits spread during training, so a reporting
bias would grow too. The test that would settle it is cheap and unrun: evaluate
both final models with the same eager fp32 loss; equal means it was a reporting
artifact.

The rejection does not depend on which it is. Option one is worse training;
option two is a loss metric that silently lies, which would undermine every
number in this document. Neither is worth ~1% throughput.

Worth noting R2 had already been flagged as marginal on throughput grounds
(1.06%, below the 3% bar, adopted only because repeats agreed). The numerics
gate rejected it independently — two different safeguards pointing the same way.

### 7.10 Adopted stack and the final curve

**`+compile_blocks_full=true`** — R1 alone — with FSDP2 as configured in
`configs/_trainer/llama.yaml`. `compile_ce` off.

| world size | step ms (cv) | tokens/s | MFU | vs baseline MFU | peak mem/GPU |
|---:|---:|---:|---:|---:|---:|
| 1 (32289) | 9,584 (0.21%) | 109,361 | **45.57%** | +11.12 pt | 50.3 GiB |
| 2 (32318) | 4,886 (0.13%) | 214,499 | **44.69%** | — | 45.7 GiB |
| 4 (32287) | 2,466 (0.25%) | 425,054 | **44.28%** | +10.58 pt | 43.4 GiB |

Scaling from the 1-GPU anchor: **98.1%** at two GPUs (4,886 vs an ideal 4,792)
and **97.2%** at four (2,466 vs 2,396). Because the adopted configuration is
also the best single-GPU one, that scaling efficiency *is* the practical
speedup — four H100s return **3.89×** one H100's best, where before this work
the same comparison gave 2.96×.

Against the 33.70% baseline at 4 GPUs: **+31% throughput and −21% peak memory**
(55.1 → 43.4 GiB), which also puts micro-batch 32 back within reach after being
OOM-bound since job 32024. Zero dynamo recompile warnings anywhere.

**R3 is not attempted.** The plan's criterion was "more than ~2 points remain";
the gap between the 1-GPU anchor (45.57%) and 4 GPUs (44.28%) is **1.29
points**, and it is communication, not compile scope. Whole-model compile under
FSDP2 — the item training_efficiency.md parked as the open question — would be
a day of tracing risk spent on something the measurements say is no longer
there.

## 8. Where that leaves the effort

The baselines reordered the work, and §7.7–7.10 then closed the item they
promoted. Both halves are worth keeping, because the order they arrived in is
the argument.

**Multi-GPU was never where this model was losing MFU.** Parallelism costs 1.7%
at two GPUs and 2.2% at four. The ~24% relative deficit the baselines found was
**all paid at world size 1** by regional `compile_blocks` leaving the `Residual`
layer eager — and widening the compiled unit to the whole `TransformerBlock`
recovered it, lifting 4 GPUs from 33.70% to **44.28%** and one GPU from 34.45%
to **45.57%**, which matches the best configuration this framework had ever
reached by any route.

| candidate | status | note |
|---|---|---|
| **widen the compiled unit to `TransformerBlock`** | **done — +10.6 pt at 4 GPUs** | the whole gap; still regional compile, so the compiled unit is the sharded unit |
| compile the cross-entropy alone | **rejected** | +1% throughput, −1.6% loss quality; §7.10 |
| whole-model compile under FSDP2 | **not attempted, and no longer worth it** | the gap it targeted is gone; 1.29 points remain and they are communication |
| batching gradient sync (F2) | open, worth < 2% | 405 of 432 reduce-scatters at B2 are avoidable, but ~70% of collective time already overlaps compute; the better argument is pushing NCCL off the RING_**LL** low-latency protocol onto a bandwidth protocol by sending fewer, larger messages |
| `reshard_after_forward` tuning | open | would remove the backward re-gather (half of 864/432 all-gathers) at a memory cost; memory is now plentiful — 43–50 GiB of 80 in use |
| micro-batch ≥ 32 | open, newly plausible | R1 freed 11.7 GiB/GPU; micro 32 has been OOM-bound since job 32024. It changes ga, so it is a separate best-fit row, not a like-for-like one |

This inverted the ranking the survey started from, where per-micro-step gradient
sync was called the "top multi-GPU throughput suspect" (training_efficiency.md
§8.1). It is real, it is visible in the launch counts, and it is worth less than
2% — because NVLink is fast and FSDP2 overlaps well. The compile-scope question
was the whole game, and it is now answered.

**Two methodological results worth more than either number.** First, a
hypothesis that explains everything is not thereby true: "widening the compiled
region moved the residual stream to bf16" accounted for the speedup, the memory
drop *and* the worse loss in one stroke, and the isolation run showed it was
wrong. Second, both safeguards earned their keep independently — the repeat
rule caught a 13.3%-cv run contended by another user, and the numerics gate
caught a rung that had already passed on throughput.

What has *not* been established, and should not be assumed from these runs:
multi-node (never run), any world size above 4, non-NVLink interconnects, a real
multi-GPU checkpoint round-trip, and resume across a changed GPU count (the
dataloader shards per rank, so a run is reproducible at fixed (seed, world_size)
only — see training_efficiency.md §8.2).

## 9. Log

- **2026-11-03 (later)** — [multigpu_compile_scope_plan.md](multigpu_compile_scope_plan.md)
  executed. R1 (`compile_blocks_full`) adopted: 4 GPUs 33.70% → **44.28%**,
  1 GPU 34.45% → **45.57%**, 2 GPUs **44.69%**, peak memory −21%, zero recompile
  warnings, numerics clean. R2 (`compile_ce`) measured, then **rejected** by the
  numerics gate. R3 ruled out by the plan's own criterion. Jobs 32285–32318.
- **2026-11-03** — baselines B1/B2/B4 complete (jobs 32264 / 32267 / 32269), all
  exit 0. FSDP2 scales at 98.3% / 97.8%; the multi-GPU deficit is dominated by
  compile scope, not parallelism (§8). Scope set; worktree `multigpu-mfu` created off `pretraining`.
  MFU accounting fixed (§3) with a world-size-parametrized regression test.
  `GeluFeedForward` cherry-picked from `mfu-abl/nanogpt-arch` (`9f3e7421`) — it
  had never been merged, so the GPT-2 architecture was not buildable on
  `pretraining`. Entropy checked: h100a is one node with 8×H100, our QoS cap is
  8 GPUs, so 2- and 4-GPU jobs are allocatable on a single node (multi-node is
  out of scope for now).
