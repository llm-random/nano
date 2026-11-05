# AME MFU optimization — report

**Period:** 2026-11-04 20:05 → 2026-11-05 11:21 (cluster local time, UTC+1),
then a reproduction and micro-batch sweep 2026-11-05 17:11 → 18:27, and the
same sweep with dense attention as a control 18:49 → 19:32 (§1.16).
**Platform:** entropy `h100` partition, H100 80GB SXM, account `mim`, QOS
`j321m_h100` until 11:21; QOS `mstefaniak_h100` (Maciej's own account, same
node) for the evening block. Branch `ame_optimize`.
**Model:** AME h20 (H-balanced, 20 blocks, dmodel 1280, 2.576B bank + 587M
backbone, 3.768 GFLOPs/token) unless stated; h16 and h12 where noted.
**Workload:** global batch 512 × 2048, 30 steps, LR 2^-12, fineweb, seed fixed.
**Active plan:** §7 — AME h20 to 45% MFU, 4 GPUs first, then 1 GPU. The
unattended run started 2026-11-05 23:55; its experiments are §1.17 onward and
the night block of §3, its progress §7.7.

All MFU figures are **per GPU** (`tokens/s × GFLOPs/token ÷ (989 TFLOP/s ×
world_size)`) and are the **mean over steps ≥ 15**, which is what
`scripts/run_summary.py` computes (`STEADY_FROM = 15`) and what the §3 ledger
in `ame_optimization_plan.md` records. Figures agree with that ledger
throughout; the evening block postdates it and is computed the same way from
the job logs.

**One run is aggregation-sensitive and it matters.** For **O9** the mean over
steps ≥ 15 is 26.16% while the median is 27.32% — a 1.13-point gap that
*flips the sign of the result*, because that run contains outlier slow steps
(234k, 239k, 240k tokens/s against a 287k median). Every other run here agrees
between mean and median to within 0.19 points, most within 0.1. §1.1 says why
the mean is the right reading there. 

---

## Headline

| configuration | MFU/GPU | step | tokens/s | % of dense ceiling |
|---|---:|---:|---:|---:|
| **B1** — AME h20, 4 GPU, start of work | 26.84% | 3.720 s | 281,748 | 60.4% |
| **Best 4 GPU** — AME h20 + align + S3b + O33 | **35.02%** | 2.852 s | 367,623 | **78.8%** |
| **Best overall** — AME h16, 1 GPU, mb 16 | **37.96%** | 5.938 s | 176,573 | **85.4%** |
| **B2** — dense H-balanced h20, 4 GPU (ceiling) | 44.44% | 2.364 s | 443,432 | 100% |

**+30.5% on the reference baseline.** Two structural facts frame everything
below:

1. **Dense parity requires AME to be *faster* than dense in wall clock.** AME
   is 3.768 GFLOPs/token against dense h20's 3.97, so matching 44.44% MFU means 
   2.244 s/step against B2's 2.364. AME buys parameters with gathers, which
   cost time and contribute no FLOPs. Structural, not a tuning gap.
2. **AME's MFU falls as GPU count rises.** 1-GPU h16 (37.96%) beats 4-GPU h16
   (33.30%). The bank machinery — id all-gather, activation all-to-all,
   reassembly permute — is work that does not exist at world size 1 and grows
   while per-rank compute shrinks.

**Reproduced, and the micro-batch question closed (2026-11-05 evening, §1.15).**
On Maciej's own account, all five best configurations re-measured within 0.23
points with identical peak memory, and h12 at 4 GPUs was measured for the first
time (30.00%). A 24-run sweep then showed every best configuration already sits
at the largest micro-batch that fits: MFU rises at every doubling until OOM,
and at global batch 512 there is no micro-batch in between.

**Dense control, same 24 cells (2026-11-05 evening, §1.16).** AME runs at
76–86% of dense MFU, best against best. At equal micro-batch on 1 GPU it is at
85–87%; h20 drops to 78.8% best against best only because its banks' memory
holds it to micro-batch 4 where dense fits 16. Going from 1 to 4 GPUs dense
loses 1.1–2.7%, so AME's larger loss at h12 and h16 is its own.

---

## 1. Improvements implemented and measured

### 1.0 Complete inventory — every O-number, and every lettered item

The O-series is the pkmf-derived catalogue in
[archive/ame_optimization.md](archive/ame_optimization.md) (O1–O31) plus Tier 4's additions
(O32–O37). **Every one is listed here, built or not**, so this table is the
index and the subsections below are the detail. Lettered items (B, D, S) are
work that arose during this effort and was never catalogued.

"Effect" is a measured A/B where one exists; "—" means no isolated measurement.
Percentages are relative MFU change at the stated shape.

| # | item | status | effect on MFU | detail |
|---|---|---|---|---|
| **O1** | Banks out of the framework's gradient path entirely | **Built** | measured as Tier 2, not separable | §1.2 |
| **O2** | Manual all-reduce at the optimizer-step boundary only | **Built** | ditto; `bank_reduce` is 1.8 ms at 4 GPU | §1.2 |
| **O3** | fp32 master / bf16 forward-copy split for the banks | **Built** | ditto | §1.2 |
| **O4** | Gradient clipping over the masters, separately | **Built** | ditto; joint clip, one fp32 ulp vs reference | §1.2 |
| **O5** | Driven by a callback at exact hook points | **Built** | ditto; three hooks, `bank_step` 20.7 ms at 4 GPU | §1.2 |
| **O6** | Two named process groups, built once | **Built** | measured as Tier 3, not separable | §1.3 |
| **O7** | Vertical-parallel forward: all-gather ids, local lookup, all-to-all | **Built** | ditto; priced at 215.6 ms all-to-all + 134.6 ms permute | §1.3 |
| **O8** | The grad-divisor subtlety | **Built** | correctness, not speed; unit-tested at mp 1/2/4 with a negative control | §1.3 |
| **O9** | Suppress gradient sync during accumulation | **Built, REJECTED** | **−2.53%** (entropy); +3.33% on lem | §1.1 |
| **O10** | Activation checkpointing | **Cut** (§0, reaffirmed 2026-11-05) | not built | §2.3 |
| **O11** | Liger fused-linear-cross-entropy | **Cut → LIFTED 2026-11-05 → built** | **−7.0%**; memory −7.7 GiB | §1.13 |
| **O12** | Fused SiLU-mul on the memory product | **Not built, never scoped** | — | §2.9 |
| **O13** | `overlap_comm` is a trap with vertical sharding | **Noted, not an optimization** | — | below |
| **O14** | One fused gather + weighted-reduce scattering into `main_grad` | **Deferred** (§0) | — | §2.12 |
| **O15** | Deduplicate the gather by unique token id | **Deferred** (§0) | — | §2.12 |
| **O16** | `ChunkedAdamW` | **Cut** (§0, memory only) | not built | §2.12 |
| **O17** | `OffloadedChunkedAdamW` | **Cut outright** (§0, no CPU offload) | not built | §2.12 |
| **O18** | `SelectiveAdamW` | **Not built**; correctness, not speed | — | below |
| **O19** | Pattern-matched param groups with per-group `lr_mult`/`wd` | **Deferred** (§0, quality not throughput) | — | §2.12 |
| **O20** | Memory values train at 4× backbone LR, wd=0 | **Deferred** (§0) | — | §2.12 |
| **O21–O27** | Batch ramp, GPU-invariant sizing, variable ga, granular micro-batching, token-space WSD, Step-Law LR, branch-decay checkpoints | **Deferred** (§0, experiment throughput not MFU) | — | §2.12 |
| **O28** | Master switch + throttle for diagnostics | **Not built as specified**; the equivalent exists as `profile_timers` / `kernel_profile` flags | — | §1.14 |
| **O29** | Periodic `gc.collect()` + `empty_cache()` | **Not built** | — | §2.11 |
| **O30** | Rank-local optimizer-state checkpoints | **Built** — as Tier 2's bank sidecars (`memory_parallel_rank{N}.pt`). §0 lists it deferred; that is stale. | correctness, not speed | §1.2 |
| **O31** | Per-component and per-callback CUDA-event timing | **Built, design changed** — the `record_function` approach was abandoned (dynamo *drops* it inside a compiled block); replaced by the `kprof` trace path | — | §1.14 |
| **O32** | A 4-GPU kernel profile | **Done** | measurement, not a change | §1.14, §2 |
| **O33** | `reshard_after_forward=False` | **Built** | **+1.7%** (34.42 → 35.02) | §1.10 |
| **O34** | Revisit DDP at 4 GPUs post-`_loss_ctx` | **Not done** — superseded by D3's result at mb 4 | — | §1.5, §2 |
| **O35** | The h-sweep at 1 GPU | **Done** | h16 **+11.1%** over h20 | §1.12 |
| **O36** | Raise micro-batch at 4 GPUs | **Attempted, blocked** | h16 mb32 OOM → with O11 fits but **−2.6%** | §1.13 |
| **O37** | No FSDP2 at world size 1 | **Built** | **+4.2%** at h20, **−2.5%** at h16 | §1.11 |
| **O38** | Smaller bank optimizer state: bf16 first moment, or no bf16 bank copy, or 8-bit states | **(b) built 2026-11-06**, opt-in (`trainer.bank_compute_copy=false`, bit-identical); (a) not built; 8-bit is a §7.10 proposal | — (target: micro-batch 8 at h20, 1 GPU) | §7.6 B1 |
| **O39** | Bank exchange off the critical path: ids all-gathered once per micro-step, each layer's gather + all-to-all on a side stream two layers ahead, reassembly folded into the compiled block, backward exchange + scatter on the side stream | **Built 2026-11-06** (`trainer.bank_prefetch_depth`) | v1 **+2.1%** (35.06 → 35.79); v2 in progress | §1.20 |
| **O40** | FSDP2 parameters kept unsharded across micro-steps (`set_reshard_after_backward(False)` until the last) | **Built, opt-in, not kept** | **−0.9%** by the mean, +0.6% by the median | §1.19 |
| **A0** | §7 attribution: 4-GPU best and its dense twin, kernel profiles | **Done** | measurement; the target is the bank path + AME glue | §1.17 |
| **B1/B2** | Baselines: AME h20 and dense h20 at 4 GPU | **Done** | 26.84% / 44.44% | Headline |
| **B3** | Dense control: the §1.15 micro-batch sweep with dense attention, all 24 cells | **Done** (2026-11-05 evening) | measurement, not a change; AME at 76–86% of dense | §1.16 |
| **D1** | Diagnose why memory parallelism gave no speedup | **Done** | found the graph break, not mem-parallelism | §4 |
| **D2** | Fused Q/K bank pair | **Built** | **+3.2%** | §1.4 |
| **D3** | DDP backbone + memory-parallel banks | **Built, not adopted** | **−2.3%**; OOMs at mb 16 | §1.5 |
| **S1** | Attention decomposition benchmark | **Done** | measurement | §1.14 |
| **S2** | Sink as a log-sum-exp merge | **Built** | **+10.3%** | §1.7 |
| **S3** | 128-aligned sequence length | **Built** | **+15.8%** — largest single win | §1.8 |
| **S3b** | Sink logit as a reduction, not an einsum | **Built** | **+2.4%** | §1.9 |
| **—** | Cross-entropy outside autocast (`_loss_ctx`) | **Built** | removes a 6.14 GiB fp32 copy; throughput-neutral | §1.6 |
| **—** | Triton cache isolation per rank | **Built** | correctness/stability | §1.14 |
| **—** | Instrumentation made non-fatal under FSDP2 | **Built** | correctness | §1.14 |
| **—** | The traced bank path checked under compile (`tests/test_bank_compile.py`) | **Done** | correct on CUDA; inductor CPU miscompiles it | §1.18 |

**Two catalogue entries are not optimizations and have no section of their own.**
**O13** is a warning rather than a change: pkmf measured `overlap_comm` as a
trap under vertical sharding, and this work never enabled it. **O18**
(`SelectiveAdamW`, masking on `grad != 0` so an unvisited bank row is not
decayed) is a *correctness* property; the equivalent here is `parameter_groups`
excluding the banks from the backbone optimizer entirely, so AdamW never sees a
permanently-`None` gradient — §1.2.


Ordered by when they were built. "Effect" is always a measured A/B against the
stated comparand, never a projection.

### 1.1 O9 — suppress gradient sync during accumulation (Tier 1)

`set_requires_gradient_sync(False)` for all but the last micro-step, so FSDP2
reduce-scatters once per optimizer step instead of once per micro-step.

**Effect: 26.84% → 26.16% (−2.53%)**, job 32578 vs 32576. On lem the *same code
and config* gave +3.33%. **Same change, opposite sign on two clusters** — which
is why the plan forbids comparing rows across clusters.

Mechanism: O9 does not make communication free, it **trades NCCL traffic for
HBM traffic**. With sync suppressed FSDP2 accumulates into an unsharded fp32
buffer every micro-step — a read-modify-write of 12.65 GB per micro-step, ~101
GB of extra HBM traffic per optimizer step at ga 8 — in exchange for removing 7
of 8 reduce-scatters. On entropy's intra-node NVLink the reduce-scatter was
already cheap and well overlapped, so the trade loses. Peak memory confirms it:
60.6 → **69.5 GiB**.

**This is the one result in this document that depends on the aggregation.**
Mean over steps ≥ 15 gives 26.16% (−2.5%); the median gives 27.32% (+1.8%). The
gap is three outlier slow steps (234k, 239k, 240k tokens/s against a 287k
median). The mean is the right reading here and not merely the conventional
one: at 69.5 of 80 GiB the run is near the memory ceiling, and allocator stalls
under that pressure are *the effect of O9*, not noise to be medianed away. A
reader who prefers the median should know the verdict flips — it is a weaker
rejection than it reads.

**Verdict: rejected on this test bed.** The Tier 1 attribution separately showed
the bank gradient path at ~8% of step time, so the gate that was to authorise
Tiers 2–3 fired NEGATIVE. Maciej explicitly overrode it and directed both tiers
be built anyway.

### 1.2 Tier 2 — hand-managed bank gradient and optimizer path (O1–O5)

`src/core/memory_parallel.py`. Removes the 2.576B bank parameters from FSDP2's
dense gradient and optimizer path: persistent fp32 `main_grad`, fp32 master →
bf16 compute copy, one all-reduce per step, joint gradient clip, three step
hooks, rank-local checkpoint sidecars.

**Effect at micro-batch 16: 26.84% → 26.55% (−1.1%)**, job 32632 vs 32576.
**Effect at micro-batch 4: 18.28% → 22.44% (+22.8%)**, job 32659 vs 32656.
**Effect at 1 GPU, ga 128: 18.1% → 25.04% (+38.5%)**, job 32652.

The mechanism this exposed is the single most reusable finding of Tier 2: **its
value scales with `gradient_accumulation_steps` ÷ FSDP2's shard factor**,
because everything it deletes is per-micro-step work. 128/8/2 micro-steps →
+38.5% / +22.1% / −1.2%. At the plan's own test bed (ga 8) it is break-even.

### 1.3 Tier 3 — vertical bank sharding across 4 GPUs (O6–O8)

Column-sharded banks, two process groups (contiguous memory-parallel, strided
data-parallel), id all-gather → local gather → all-to-all, and the O8 divisor
(global world size, not dp size). Same module as Tier 2; `mem_parallel_size` is
the knob.

**Effect: it is what makes h20 fit at all.** `mem_parallel_size` 1 and 2 both
OOM at 4 GPUs (jobs 32649, 32638). Throughput at mb 16 is covered by 1.2 above.
The 4-GPU profile later priced it exactly: **215.6 ms of all-to-all plus 134.6
ms of reassembly permute = 350.2 ms/step**, against the 546 ms of FSDP2 bank
all-gather it replaced. A real but thinner margin than designed for.
`bank_reduce` is 1.8 ms and `bank / embedding` is 1.0% of the step — the path
Tier 1's gate called unimportant is, after Tiers 2–3, invisible.

### 1.4 D2 — fused Q/K bank pair

One `torch.cat` of the two bank weights so a single gather, single all-to-all
and single permute serve both. The kernels are launch-bound, not
bandwidth-bound, so halving the count is most of the cost.

**Effect: 26.55% → 27.41% (+3.2%)**, job 32666 vs 32632.

### 1.5 D3 — DDP backbone instead of FSDP2

Replicate the 587M backbone and let `memory_parallel` own the banks, in the
style of pkmf. Directed by Maciej.

**Effect: 23.52% → 22.99% (−2.3%)** at micro-batch 4, job 32664 vs 32674.
**Not adopted.** At micro-batch 16 it OOMs outright (jobs 32672, 32678); at
micro-batch 8 it reaches 25.23% (job 32680), still short of FSDP2's mb16.
The honest summary: DDP trades memory for fewer collectives, and at h20 there
is no memory to trade.

### 1.6 `_loss_ctx` — cross-entropy outside autocast

`F.cross_entropy` is on torch's autocast **fp32 list**, so inside an autocast
region it upcast the `[16, 2047, 50304]` logits to fp32 — 6.6 GiB materialised
on top of the bf16 logits that already existed. Disabling autocast around the
loss keeps them bf16.

**Effect: removes a 6.14 GiB fp32 copy.** The bf16 logits (3.07 GiB at
`[16, 2047, 50304]`) already existed and remain; what autocast added was a
second, fp32 tensor of twice that size on top of them. It is what the first DDP
micro-batch 16 run (32672) died on, and it unblocked DDP at micro-batch 8.
Throughput-neutral; a pure memory fix. Found because Maciej suspected a bug in
the loss path and was right.

### 1.7 S2 — the sink as a log-sum-exp merge instead of a KV column

The attention sink was a prepended key column. Two structural costs: `KV_LEN`
became 2049, one past flex_attention's 128-wide block grid, making every block
ragged; and it put a `torch.where` inside `score_mod`, which flex evaluates
once per score (n² per head) to special-case a single column. Replaced by
merging the sink after the fact — attention over `{sink} ∪ sequence` is exactly
the softmax-weighted merge of attention over each part, and flex returns the
log-sum-exp needed to do it:

    out = out_seq + α·(v_sink − out_seq),   α = sigmoid(s − lse)

Written as a broadcast FMA, not `torch.lerp` (which would expand `v_sink`) and
not the ratio form (which built five full-size fp32 temporaries, measured at
+156 ms forward and +403 ms backward).

**Effect: 27.41% → 30.24% (+10.3%)**, job 32687 vs 32666. Kernel-level:
1331.6 → 679.5 ms/step. Loss trajectory unchanged.

### 1.8 S3 — 128-aligned sequence length (the largest single win)

`_preprocess_input` takes `batch[:, :-1]` for next-token targets, so
`sequence_length: 2048` made the model see **n = 2047**, and 2047 = 15×128 +
127 — one token short of flex's block grid, leaving a ragged last row-block.
Present in every AME run in this work **including every baseline**. Fixed with
`sequence_length: 2049`, which also makes the model train on exactly the 2048
tokens per sequence the configs claim.

**Kernel effect (job 32688, 1 H100, scaled to one optimizer step):**

| | fwd | bwd | total |
|---|---:|---:|---:|
| n = 2047 | 482.1 | 674.8 | **1156.9 ms** |
| n = 2048 | 171.4 | 505.6 | **677.0 ms** |

479.9 ms for one token, 2.8× of it in the forward. The SDPA flash floor moves
only 22 ms across the same pair, so this is not a general alignment tax — it is
flex's handling of a partial row-block, the same mechanism as S2's ragged
`KV_LEN`, on the other axis of the same matrix.

**Effect at 1 GPU: 28.80% → 33.36% (+15.8%)**, jobs 32771 vs 32772, with
`sequence_length` the only overridden variable. 1.925 s/step saved against
1.920 predicted — a **100.3% transfer** from kernel benchmark to training.

**Caveat recorded as a deviation:** each example now spans 2049 dataset tokens,
so the token stream is not B1's and loss traces are comparable in shape but not
step-for-step. A data change, not a numerics change. The cleaner fix — the
dataloader fetching `sequence_length + 1` — was deliberately **not** done,
because it changes shared code and every other experiment's token accounting.

### 1.9 S3b — the sink logit as a reduction instead of an einsum

`einsum("bhnd,hd->bhn", q, k_sink)` is a matrix-**vector** product per head
(reduces D=64 to one scalar per element), the shape a tensor core is worst at,
and it lands in a GEMM library kernel that inductor cannot fuse into anything.
Replaced by a broadcast multiply and a reduction over D.

**Measured in isolation (job 32776), per optimizer step:**

| merge form | fwd | bwd | total |
|---|---:|---:|---:|
| einsum, eager | 88.3 | 131.5 | 219.9 |
| mul+sum, eager | 121.6 | 159.7 | 281.3 |
| einsum, **compiled** | 41.8 | 61.8 | 103.6 |
| **mul+sum, compiled** | **14.2** | **36.7** | **50.9** |

**The ranking inverts under compile.** Eager the einsum wins by 28%; compiled
it loses by 2×. This also retired an earlier estimate of mine: the
decomposition appeared to show 241 ms of merge cost, but the real model
compiles this path and was already paying ~103.6 ms.

**Effect at 1 GPU: 33.36% → 34.17% (+2.4%)**, job 32796 vs 32772.

### 1.10 O33 — `reshard_after_forward=False`

> **Naming note.** Configs and wandb runs for this item carry the suffix
> `o29` (`h20_4gpu_mb16_mp4_o29.yaml`, `ame_h20_4gpu_align_o29`,
> `ame_h16_4gpu_mb16_o29`) because it was first labelled O29. That collided
> with the catalogue — see the note at the top of §2 — and the item is now
> **O33**. The run names are left alone so they still match wandb and slurm.

FSDP2 frees all-gathered parameters at the end of forward and gathers them
again in backward. The 4-GPU profile put `ncclDevKernel_AllGather` at 219.2
ms/step, 106.0 of it in backward. Plumbed as an optional key so no existing run
moves.

**Effect: 34.42% → 35.02% (+1.7%)**, job 32847 vs 32820. 50 ms recovered, which
is 47% of the 106 ms backward all-gather — the rest stays hidden behind
compute, exactly as the overlap analysis predicted. Costs 1.1 GiB (63.50 →
64.63), far less than feared.

### 1.11 O37 — no FSDP2 at world size 1

At 1 GPU FSDP2 shards nothing yet runs its full machinery. bf16 comes from
autocast instead.

**Effect at h20 (ga 128): 34.17% → 35.59% (+4.2%)**, job 32834 vs 32796.
**Effect at h16 (ga 32): 37.96% → 37.00% (−2.5%)**, job 32841 vs 32825.

**It reverses sign between the two.** The discriminator is `ga`: FSDP2's
per-micro-step staging is 4× smaller at h16, while autocast's own per-region
weight-cast cost is not. There is a crossover and h16 is already past it. Peak
memory *rises* (64.71 → 66.87 GiB), which is the tell — autocast keeps fp32
parameters and caches casts. **1-GPU only**; at 4 GPUs FSDP2 is what makes h20
fit.

### 1.12 Model size sweep (h12 / h16 / h20)

Each at the largest micro-batch its unsharded bank state leaves room for —
bank state at `mem_parallel_size 1` is 46.4 GB (h20), 29.7 (h16), 16.7 (h12).

| 1 GPU | micro-batch | MFU | step | peak GiB | GFLOPs/token |
|---|---:|---:|---:|---:|---:|
| h12 | 32 | 34.46% | 3.240 s | 66.54 | 1.053 |
| **h16** | **16** | **37.96%** | 5.938 s | 68.35 | 2.121 |
| h20 | 4 | 34.17% | 11.644 s | 64.71 | 3.768 |

**"Smaller is better" is wrong — there is a peak at h16.** Mechanism:
arithmetic intensity falls with H (3.768 → 2.121 → 1.053 GFLOPs/token), so each
token carries less compute to hide memory traffic behind. h16 is where the
micro-batch gain still outweighs that loss; by h12 it does not.

**And the ranking inverts with GPU count:**

| MFU/GPU | 1 GPU | 4 GPU |
|---|---:|---:|
| h20 | 34.17% | **35.02%** |
| h16 | **37.96%** | 33.30% |
| h12 | 34.46% | 30.00% |

At 1 GPU h20 is memory-starved into micro-batch 4 and h16's 4× micro-batch
wins. At 4 GPUs the banks shard, h20 gets micro-batch 16 too, and its higher
arithmetic intensity decides it — the 4-GPU order is exactly the order of
GFLOPs/token. (h12 at 4 GPUs: measured 2026-11-05 evening, job 33076, at its best
micro-batch of 32; §1.15.)

### 1.13 O11 — fused-linear-CE (lifted from the cut list 2026-11-05)

Chunks the head projection and cross-entropy so the full
`[micro_batch, seq, vocab]` logits tensor is never built. `targets` is an
argument to `TransformerHead.forward` rather than a separate method, because
FSDP2's all-gather hook fires on `__call__` — a `head.compute_loss(...)` would
read a sharded DTensor as the whole weight, which FSDP2 does not report as an
error.

**Three implementations were measured:**

| h16, 1 GPU, mb 16 | MFU | step | peak GiB |
|---|---:|---:|---:|
| no O11 | **37.96%** | 5.938 s | 68.35 |
| v1, `torch.utils.checkpoint` per chunk | 35.31% | 6.385 s | 60.65 |
| v2, gradients-in-forward, fp32 contractions | 20.87% | 10.787 s | 62.08 |
| v3, gradients-in-forward, model-dtype contractions | 30.38% | 7.408 s | — |

**Memory claim: confirmed.** 7.7 GiB recovered at 1 GPU, and h16 at 4 GPUs
micro-batch 32 now fits (62.61 GiB) where it OOMed on exactly the 6.14 GiB
`[32, 2048, 50304]` logits tensor (job 32853).

**MFU claim: not confirmed. O11 is a net loss as measured.** v1 costs 2.65
points; the micro-batch it unlocks is worth about +4.7% relative (backing the
7.0% out of the mb 32 run gives 34.87% against mb 16's 33.30%), less than the
7.0% it costs. At 4 GPUs, micro-batch 32 with O11 (32.44%) is *below*
micro-batch 16 without it (33.30%).

**v1 is the best of the three, and that is the interesting result.** v2/v3
avoid the recompute entirely by computing gradients during the forward (valid
because the loss is the final scalar, so all gradients below are linear in
`grad_output`) — and are still slower, because hand-rolled logsumexp + gather +
scatter_add + exp over a `[chunk, 50304]` buffer costs more than ATen's fused
`cross_entropy` plus the recompute. **The recompute was cheaper than replacing
the kernel.** v2's catastrophic 20.87% was my own bug: the fp32 accumulation
dtype reached the *contractions*, making two fp32 matmuls per chunk (~60
TFLOP/s against ~989 for bf16) on the largest matmul in the step.

### 1.14 Measurement and test infrastructure (no MFU effect, enabled everything above)

- **`scripts/kprof_phase.py` usage at 4 GPUs** — the first 4-GPU phase
  attribution in this work. Every earlier table was 1 GPU, where collectives
  are single-rank no-ops.
- **`main.py: isolate_triton_cache()`** — every rank hands its generated
  kernels to Triton, which caches them by content hash in a shared directory.
  Concurrent ranks racing on the same cache corrupted it. Each rank now gets
  its own cache on node-local storage, honouring an explicit `TRITON_CACHE_DIR`
  if one is set. Four Tier 1 jobs (32543/44/55/59) were cancelled and
  resubmitted to pick this up. Tested by `tests/test_triton_cache_isolation.py`.
- **Instrumentation made non-fatal** — AME's per-layer diagnostics call
  `aten.mm` on a plain tensor against a DTensor under FSDP2, which raises. It
  now disables itself per layer with a warning rather than killing the run, and
  the run says so in its log. Every multi-GPU AME log in §3 carries 20 of these
  warnings. Tested by `tests/test_instrumentation_is_not_fatal.py`.
  `run_summary.py` was corrected alongside it: it had treated the handled
  `RuntimeError` inside that warning as a failure and reported two healthy
  30-step runs as failed.
- **`scripts/run_summary.py`** — reduces a slurm log to the numbers a results
  table needs (MFU accounting line, `timers_ms` steady-state split, peak
  memory), and exits non-zero on failure signatures so "did it finish" is
  checkable. `STEADY_FROM = 15` is the convention this report follows. Extended
  here with the `bank_reduce` and `bank_step` sections. Tested by
  `tests/test_run_summary.py`.
- **`scripts/kprof_phase.py`** — splits a profiled step into forward/backward
  per kernel category by walking each GPU kernel back to the `section:` range
  LoopTimers had open. Produced every phase table in §1 and §2.
- **`tests/test_grad_sync_accumulation.py`** — pins O9's semantics: sync
  suppressed on all but the last micro-step, and the resulting gradient
  identical to the unsuppressed path.
- **`scripts/kprof_agg.py` regex fixes** — `index_add_` dispatches to
  `indexFuncLargeIndex`, matching neither `index_add` nor `embedding`, so 93.8
  ms/step was filed under `other`. Added `|indexFunc` and
  `|all_?to_?all|alltoall|send_?recv`.
- **`scripts/bench_attention.py` compiled variants** — every row it produced
  before 2026-11-05 was eager while the model runs compiled. That inverted the
  merge-form ranking (§1.9) and overstated merge cost by ~2.3×. Any future row
  should come in both forms.
- **`conftest.spawn_on_free_port()`** — `free_port()` asks the OS for a port,
  closes the socket, and hands the number to a child that binds it later. On a
  shared node with several jobs each spawning four ranks, something takes it in
  between; the failure is `EADDRINUSE` out of `init_process_group` and looks
  exactly like a correctness failure in whichever test drew the short straw. It
  cost one gate run (32836). Now retries on EADDRINUSE and nothing else.
- **`tests/test_memory_parallel.py`** — 28 tests including §1.4.1 bit-identity
  without clipping, one-fp32-ulp with clipping, the O8 divisor at
  `mem_parallel_size` 1/2/4 with a negative control that catches the wrong
  divisor as exactly 4.0×, column-shard reassembly, checkpoint round-trip,
  fused-vs-unfused equivalence.
- **`tests/test_fused_linear_ce.py`** — 14 tests, float64, tolerance of a few
  ulps with a negative control on each that misses by 1e-2 or more.

### 1.15 Micro-batch sweep — every cell (2026-11-05 evening)

Run on Maciej's own account (`mstefaniak_h100`, same node), with
`trainer.profile_timers=true` throughout. The question: is each §6.1
configuration at its best micro-batch? 24 runs, four per (model, GPU count)
cell, bracketing the known best with smaller micro-batches below and an OOM
above. Everything except micro-batch is held at the §6.1 configuration; global
batch stays 512, so `gradient_accumulation_steps = 512 ÷ (micro-batch × GPUs)`.
Three points are the reproduction runs, whose configs are identical.

**MFU/GPU, %** (bold = best in the row; jobs in §3). These are the sweep's own
aggregation, which reads 0.00–0.04 points above the W&B mean over steps ≥ 15
used everywhere else (four runs checked): h12 at 4 GPUs is 30.04% here and
30.00% by the mean, the figure §6.1 uses.

| model | GPUs | mb 1 | mb 2 | mb 4 | mb 8 | mb 16 | mb 32 | mb 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h12 | 1 | | | | 29.57 | 33.01 | **34.57** | OOM |
| h16 | 1 | | | 25.45 | 34.93 | **37.73** | OOM | |
| h20 | 1 | 16.51 | 31.13 | **35.45** | OOM | | | |
| h12 | 4 | | | | 24.66 | 28.53 | **30.04** | OOM |
| h16 | 4 | | | 24.83 | 30.34 | **33.29** | OOM | |
| h20 | 4 | | | 30.23 | 33.48 | **34.91** | OOM | |

**1. In all six cells the best micro-batch is the largest that fits.** MFU
rises at every doubling until the next one OOMs. The §6.1 configurations were
already there, so the sweep moves no best-known number; it confirms each one
is the optimum for this batch size. That is exact, not sampled: micro-batch ×
ga × GPUs = 512 = 2⁹ admits only powers of two, so no micro-batch exists
between the last that fits and the first that OOMs.

**2. Five of the six OOMs die allocating exactly the bf16 logits tensor**
`[micro-batch, 2048, 50304]`: 12.28 GiB at mb 64 (h12, both GPU counts), 6.14
GiB at mb 32 (h16, both), 1.54 GiB at mb 8 (h20, 1 GPU). The sixth, h20 at 4
GPUs and mb 32, failed on a 320 MiB allocation with 79.12 of 79.18 GiB in use,
nowhere near fitting. So the logits tensor is the wall almost everywhere, which
is what O11 targets (§1.13); §2.13 sets the price O11 would have to beat.

**3. Each doubling returns less than the one before**, 0.16–0.44× of it:

| cell | second-to-last doubling | last doubling that fits |
|---|---:|---:|
| h12, 1 GPU | 8→16: +11.6% | 16→32: +4.7% |
| h16, 1 GPU | 4→8: +37.2% | 8→16: +8.0% |
| h20, 1 GPU | 1→2: +88.6% | 2→4: +13.9% |
| h12, 4 GPU | 8→16: +15.7% | 16→32: +5.3% |
| h16, 4 GPU | 4→8: +22.2% | 8→16: +9.7% |
| h20, 4 GPU | 4→8: +10.8% | 8→16: +4.3% |

(relative MFU change). At the small end the step is overhead-bound: h20 at 1
GPU and mb 1 runs 512 micro-steps per optimizer step, and its forward (12.2 s)
costs more than its backward (11.8 s), the reverse of every other run.

**4. The MFU lost going from 1 to 4 GPUs grows as the model shrinks:** h20
−1.5%, h16 −11.8%, h12 −13.1% (same-session best points, relative). h12 has the
least compute per token (1.053 GFLOPs) to set against the bank machinery's
fixed per-step cost, so it is the weakest 4-GPU cell, as §6.4 had predicted.

**5. The time profiler finds no gaps between sections.** At every best point the
GPU-timed sections (data_h2d, forward, backward, clip, bank_reduce, optim,
bank_step) cover 99.79–99.95% of the step; backward is 68.3–69.1% of it and
forward 29.7–30.5%. Whatever MFU is left is inside the forward and backward
kernels, not in orchestration. The `log` section (46–186 ms) is a host timer
around the loss `.item()` sync, so it overlaps the GPU sections rather than
adding to them, the same reading as the `log = 151.5 ms` correction in §4.

**Reproduction.** The five cells measured in the morning re-measured within
0.23 points with identical peak memory, h16 at 4 GPUs to the millisecond:

| cell | morning | evening | Δ points | peak GiB (both) |
|---|---:|---:|---:|---:|
| h12, 1 GPU, mb 32 | 34.46% (32832) | 34.57% (33022) | +0.11 | 66.54 |
| h16, 1 GPU, mb 16 | 37.96% (32825) | 37.73% (33034) | −0.23 | 68.35 |
| h20, 1 GPU, mb 4, no FSDP2 | 35.59% (32834) | 35.45% (33051) | −0.14 | 66.87 |
| h16, 4 GPU, mb 16 | 33.30% (32856) | 33.29% (33071) | −0.01 | 44.53 |
| h20, 4 GPU, mb 16 | 35.02% (32847) | 34.91% (33067) | −0.11 | 64.63 |

The morning runs had `profile_timers` off and the evening runs had it on; its one
`torch.cuda.synchronize()` per step is inside these deltas, so it costs at most
a few tenths of a point.

**Provenance.** wandb project `ideas_cv/nano-ame`. Sweep runs are named
`ame_<model>_<GPUs>g_mb<N>_entr`; the three 1-GPU reproductions are
`ame_h12_1gpu_mb32_entr`, `ame_h16_1gpu_mb16_entr` and
`ame_h20_1gpu_mb4_nofsdp_entr`. Each one's cemetery branch is its run name plus
the submit timestamp. The h12/4-GPU base is the one new config,
`configs/ame_tests/h12_4gpu_mb16_o29.yaml`; every other point is a §6.1 config with
`trainer.gradient_accumulation_steps` overridden on the command line.

### 1.16 Dense control — the same sweep with dense attention (2026-11-05 evening)

Not an optimization: a per-cell dense reference for §1.15, in place of the
single B2. The 24 cells were rerun on the same configs with two overrides,
`common.attention.type=gqa` and `trainer.mem_parallel_size=null` (the bank
knob, which the trainer rejects when there are no banks). Everything else is
identical, including sequence length 2049, `compile_blocks_full`, O33 at 4 GPUs
and O37 for h20 at 1 GPU, so only the attention layer differs: GQA with 20
query and 10 KV heads, RoPE, SDPA. Dense was not tuned on its own; these are its
numbers on AME's recipe. Each ladder starts one doubling higher than AME's (two
for h20 at 1 GPU, none for h12 at 4 GPUs) because dense carries no bank state.
Jobs 33106–33131 (§3), wandb `dense_<model>_<GPUs>g_mb<N>_entr`, same
aggregation as §1.15. Step, tokens/s, peak memory and forward/backward time per
run are in [ame_microbatch_sweep.md](ame_microbatch_sweep.md).

**Dense MFU/GPU, %** (bold = best in the row):

| model | GPUs | mb 4 | mb 8 | mb 16 | mb 32 | mb 64 | mb 128 |
|---|---:|---:|---:|---:|---:|---:|---:|
| h12 | 1 | | | 38.77 | **40.32** | OOM | OOM |
| h16 | 1 | | 40.84 | 43.31 | **44.91** | OOM | |
| h20 | 1 | 40.97 | 43.33 | **44.98** | OOM | | |
| h12 | 4 | | 33.57 | 37.75 | **39.34** | OOM | |
| h16 | 4 | | 39.73 | 42.78 | **43.68** | OOM | |
| h20 | 4 | | 41.96 | **44.47** | OOM | OOM | |

**AME against dense, per cell** (AME from §1.15):

| cell | AME best | dense best | AME % of dense MFU | AME % of dense tokens/s | AME % of dense at equal micro-batch |
|---|---:|---:|---:|---:|---|
| h12, 1 GPU | 34.57% (mb 32) | 40.32% (mb 32) | 85.7% | 89.1% | 85.1% (mb 16), 85.7% (mb 32) |
| h16, 1 GPU | 37.73% (mb 16) | 44.91% (mb 32) | 84.0% | 88.0% | 85.5% (mb 8), 87.1% (mb 16) |
| h20, 1 GPU | 35.45% (mb 4) | 44.98% (mb 16) | 78.8% | 82.9% | 86.5% (mb 4) |
| h12, 4 GPU | 30.04% (mb 32) | 39.34% (mb 32) | 76.4% | 79.4% | 73.5% (mb 8), 75.6% (mb 16), 76.4% (mb 32) |
| h16, 4 GPU | 33.29% (mb 16) | 43.68% (mb 32) | 76.2% | 79.9% | 76.4% (mb 8), 77.8% (mb 16) |
| h20, 4 GPU | 34.91% (mb 16) | 44.47% (mb 16) | 78.5% | 82.6% | 79.8% (mb 8), 78.5% (mb 16) |

**1. Dense obeys the same rule: in all six cells the best micro-batch is the
largest that fits.** Four of the six first OOMs die allocating the logits
tensor (12.28 GiB at mb 64, 6.14 GiB at mb 32); h16's at mb 64 fail on 128–256
MiB with the card full. h12 at 1 GPU and mb 128, and h20 at 4 GPUs and mb 64,
were redundant second brackets.

**2. AME's tokens/s ratio runs 3–4 points above its MFU ratio**, because AME
does 3.8–5.0% fewer FLOPs per token (h12 1.053 vs 1.095, h16 2.121 vs 2.222,
h20 3.768 vs 3.965 GFLOPs).

**3. On 1 GPU at equal micro-batch, AME runs at 85–87% of dense.** h20's lower
78.8% best against best is memory: AME's best micro-batch is 4, dense's 16. At
mb 4 AME peaks at 66.87 GiB and dense at 21.47 GiB, and the banks' state
accounts for most of the 45.4 GiB difference: 18 bytes per bank parameter (bf16
compute copy, fp32 master, fp32 gradient accumulator, two fp32 Adam moments),
43.2 GiB at h20. At h16 dense also fits twice AME's micro-batch, on both GPU
counts.

**4. The 1→4-GPU loss is AME's own.** Going from 1 to 4 GPUs dense loses 2.4%
(h12), 2.7% (h16) and 1.1% (h20); AME loses 13.1%, 11.8% and 1.5% (§1.15,
finding 4). The difference at h12 and h16 is the bank machinery, as §6.3
attributes it. At equal micro-batch on 4 GPUs AME is at 73.5–79.8% of dense.

**5. Dense h20 at 4 GPUs and mb 16 reproduces B2**: 44.47% against 44.44% (job
32577), on the new account and at sequence length 2049 instead of 2048.

### 1.17 A0 — attribution of the 4-GPU best against its dense twin (2026-11-05/06 night)

The first step of §7. Both runs are the §6.1 h20/4-GPU configuration
(`configs/ame/h20_4gpu.yaml`) with `trainer.kernel_profile=true
trainer.profile_timers=true`; the dense twin adds `common.attention.type=gqa
trainer.mem_parallel_size=null`. Steps 12–13 are profiled; MFU is the mean over
steps ≥ 15 as everywhere, so the profiler does not touch it.

| | job | MFU/GPU | step | compute kernels busy | GPU idle | exposed NCCL |
|---|---|---:|---:|---:|---:|---:|
| AME | 33502 | **35.06%** | 2.849 s | 2615.1 ms | 71.5 ms | ~255 ms |
| dense | 33507 | **44.51%** | 2.361 s | 2311.9 ms | 39.4 ms | ~26 ms |

"Exposed" is measured on the timeline (`kprof_kernels.py` in the scratch tools,
now the method): the time an NCCL kernel runs while no compute kernel runs on
any stream. Per collective, busy / exposed ms per step:

| | AME | dense |
|---|---:|---:|
| bank all-to-all (`SendRecv`) | 196.6 / **196.6** | — |
| all-gather (FSDP2 parameters, plus AME's id all-gather) | 103.1 / 75.3 | 76.4 / 9.1 |
| reduce-scatter | 98.9 / 39.1 | 133.0 / 17.1 |

**Per category, AME minus dense (ms per step, kernel time):**

| category | AME | dense | Δ |
|---|---:|---:|---:|
| GEMM | 1138.7 | 1311.4 | **−172.7** |
| attention (flex vs flash) | 477.1 | 477.0 | **+0.1** |
| triton glue | 737.6 | 370.8 | **+366.8** |
| NCCL all-to-all | 196.8 | 0 | +196.8 |
| copy / memcpy | 70.4 | 9.4 | +61.0 |
| other eager | 107.9 | 64.0 | +43.9 |
| optimizer-category kernels | 31.4 | 13.4 | +18.0 |
| softmax / CE, FSDP2 packing, bank / embedding | 84.5 | 86.6 | −2.1 |

**Findings.**

1. **Attention is at parity.** flex with ALiBi costs exactly what dense's
   flash costs (477.1 vs 477.0 ms). Nothing is left to win from the kernel by
   matching dense.
2. **AME's GEMMs are 172.7 ms cheaper than dense's.** It has no Q/K projections
   (6.4% fewer GEMM FLOPs), and dense's N = 640 K/V projections are its least
   efficient GEMMs. So if AME's other costs equalled dense's, AME would run at
   about 2.361 − 0.173 = 2.19 s, **45.5%**: the 45% target is the AME-specific
   overhead, not the shared backbone.
3. **The bank path costs about 500 ms of critical path**: the all-to-all
   (196.6, fully exposed), the backward stack/permute of the fused pair
   (`triton_poi_fused_stack_10`, 135.6 ms, 160 launches), the copies around the
   traced collective (`triton_poi_fused_all_to_all_single_*`, 34.7 ms;
   `direct_copy_kernel` in backward, 60.4 ms, 480 launches), and the scatter
   into `main_grad` (`triton_poi_fused__to_copy_index_add_12`, 36.9 ms).
4. **AME's own attention-side math** (normalize, KV-shift mix, tau, sink merge,
   and the reductions for the per-head parameters' gradients) is the rest of
   the glue excess: about 53 ms forward and 107 ms backward over dense's RoPE.
5. **The exchange is traced, not a graph break.** The kernel names
   (`triton_poi_fused_all_to_all_single_*`, `..._index_add_*`) show dynamo
   tracing `_FusedBankGather` into the compiled block, collectives and
   `main_grad` scatter included. That retires S5's premise (§2.6): since D2 the
   fused path has no graph break. Its correctness under compile had never been
   checked; §1.18 does.

### 1.18 The compiled bank path is correct on CUDA, and wrong on CPU

`FusedBankPair.forward` carries no `@torch.compiler.disable`, so every AME run
since D2 (job 32666) has run the bank gather, the all-to-all and the in-place
`index_add_` into `main_grad` inside the compiled block. D1 (2026-11-04) had
measured that tracing the bank path gives a wrong `main_grad` (7.3e-3 on the
toy model) and blamed AOTAutograd. Rechecked:

| backend | where | main_grad vs eager |
|---|---|---|
| `aot_eager` | CPU | **exact** (0.0) |
| inductor | CPU | **wrong**, max 7.3e-3: k banks only, q banks exact |
| inductor, mp 1 | CUDA (job 33504) | **pass** (< 1e-4 relative) |
| inductor, NCCL, mp 2 | CUDA (job 33504) | **pass** |

The CPU error is an inductor C++ codegen bug in torch 2.7.1, not an AOTAutograd
one: the AOT graph functionalizes `index_add_` into
`index_put(..., accumulate=True)` plus a `copy_` back into `main_grad`, which
is right; the generated C++ kernel for the k bank's scatter indexes its target
with the wrong tile variable (`x0 + 2*x0_inner + 64*id` in a loop over
`x1_inner`), spreading each row's gradient over rows `id` and `id + 1`. It needs
`qk_norm` (the normalize backward is what gets fused into the tiled kernel) and
is independent of the shift and the sink. D1's 7.3e-3 is this bug: same toy
model, same number. **Production is CUDA, and on CUDA the path is correct.**
The gate for it is now `tests/test_bank_compile.py` (aot_eager on CPU; inductor
on CUDA at mp 1 and over NCCL at mp ≥ 2).

### 1.19 O40 — parameters unsharded across micro-steps (built, opt-in, not kept)

`trainer.keep_params_unsharded_during_accumulation=true` calls FSDP2's
`set_reshard_after_backward(False)` on every micro-step but the last, so with
O33's `reshard_after_forward=False` the backbone is all-gathered once per
optimizer step instead of once per micro-step. Gradients are still
reduce-scattered every micro-step, so nothing changes numerically:
`tests/test_grad_sync_accumulation.py` checks bit-identical gradients and that
the all-gather count falls by exactly the accumulation factor.

**Effect (job 33509 against 33502): 35.06% → 34.75% by the mean, 35.28% by the
median.** Thirteen of the fifteen steady steps ran 2823–2835 ms against the
base's 2834–2860 (forward −32 ms, backward +14 ms), and two had their forward
jump by +614 and +834 ms (steps 14 and 26). Peak memory 64.63 → 64.79 GiB. By
this report's rule (the mean, §1.1) it is a loss. Whether the spikes are O40's
own (an allocator stall, as with O9) or another job starting on the node was
not settled by one run; O40 is kept as an opt-in flag for a retest on top of
O39, where the all-gather it removes is a larger share of what is exposed.

### 1.20 O39 — the bank exchange off the critical path (built)

AME's q and k are gathers by token id, so a layer's rows depend on nothing the
layers below compute. Until now each block nevertheless waited for its own
all-to-all inside its compiled graph (§1.17: 196.6 ms per step, all of it
exposed). O39 moves the exchange out of the blocks:

- the encoder all-gathers the micro-batch's ids **once** (S6, §2.7: 8
  launches per step instead of 160) and issues each layer's gather and
  all-to-all on a separate CUDA stream, two layers ahead
  (`trainer.bank_prefetch_depth=2`, `src/core/memory_parallel.py`
  `BankPrefetcher`);
- the block receives the all-to-all's receive buffer as it lands (rank-major)
  and reassembles it through a view inside its compiled region, where the
  transpose folds into the normalize that reads it (A2, §2.2): the backward
  `stack`/permute copy (135.6 ms) disappears;
- the gradient exchange and the scatter into `main_grad` run on the side
  stream too (autograd runs a Function's backward on its forward's stream), so
  the compute stream never waits for them. `reduce()` joins the streams before
  anything reads `main_grad`.

Gates: bit-identical to the fused path on a gloo mesh in float64 at mp 2 and
4, losses, bank masters and every backbone parameter after two optimizer steps
(`tests/test_bank_prefetch.py`).

**v1 (job 33512): 35.06% → 35.79% (+2.1%), step 2.849 → 2.791 s, peak 64.61
GiB.** Forward unchanged (849 vs 851 ms), backward −56 ms. Loss trace: equal to
the base through step 17, one bf16 ulp (0.03125) apart at step 18, equal again
after; grad-norm trace drifts from step 1 with alternating sign (two runs of
the base itself match to step 16), the signature of a reassociated
fp32 accumulation, not of a different function.

**Why so little: v1's kernel profile (job 33527).**

| per step | A0 (33502) | O39 v1 (33527) |
|---|---:|---:|
| all-to-all busy / exposed | 196.6 / 196.6 | 444.6 / 59.6 |
| compute-stream GEMMs | 1138.7 | 1317.3 |
| bank gathers | ~15 (fused, Triton) | 169.8 (`indexSelectLargeIndex`, eager) |
| scatter into `main_grad` | 36.9 (Triton) | 158.6 (`indexFuncLargeIndex`, eager) + 86.0 copies |
| compute-stream gaps | — | 200.2 |

The exposure went (196.6 → 59.6 ms) and so did the backward copy, but the work
moved to the side stream ran as **eager kernels 3–11× slower** than the
compiled ones they replaced, q and k as two all-to-alls instead of one, and all
of it beside the compute stream, whose GEMMs ran **178.6 ms slower** for the
shared memory bandwidth and SMs. A side stream hides latency, not cost.

**v2** (job 33530): the side-stream gather and scatter compiled (`torch.compile`
on two small functions, CUDA only), and q and k back in one buffer and one
all-to-all per layer. An optional cap on the bank group's NCCL CTAs
(`trainer.bank_nccl_max_ctas`) to bound what the exchange takes from compute.

---

## 2. Promising, not done or not tested

> **On the O-numbering, and a collision I introduced.** O1–O31 are a catalogue
> of optimizations harvested from the pkmf framework, defined in
> [ame_optimization.md](archive/ame_optimization.md). Tiers 1–3 implemented O1–O9; the
> rest were cut or deferred, which is why the implemented set jumps from O8/O9
> straight to the thirties — the numbers in between exist as catalogue entries
> that were never built.
>
> Tier 4 originally labelled its new items **O28–O33**, which were **already
> taken** (O28 diagnostics switch, O29 periodic `gc.collect()`, O30 rank-local
> optimizer-state checkpoints, O31 per-component CUDA-event timing — and O31 is
> what Tier 1 implemented as T1.3). They have been renumbered to start after
> the catalogue:
>
> | was | now | item |
> |---|---|---|
> | O28 | **O32** | a 4-GPU kernel profile |
> | O29 | **O33** | `reshard_after_forward=False` |
> | O30 | **O34** | revisit DDP at 4 GPUs |
> | O31 | **O35** | the h-sweep at 1 GPU |
> | O32 | **O36** | raise micro-batch at 4 GPUs |
> | O33 | **O37** | no FSDP2 at world size 1 |
>
> **New items start at O38.** Note separately that the catalogue's O30
> (rank-local optimizer-state checkpoints) was in fact *implemented* by Tier 2
> as the bank sidecars — it is listed as deferred in §0 but is not.

Ranked by expected value. Estimates are flagged for certainty; the 4-GPU ones
are discounted because that step is partly communication-bound.

### 2.1 Head-parallel attention — removes 350 ms at 4 GPUs (redesign)

Column shard `r` owns heads `[5r, 5r+5)`, and heads are **independent** in
attention. The all-to-all and the reassembly permute exist only to put back
together heads that never needed putting together. Computing attention per head
group is tensor parallelism — a redesign of O6, not a tweak.

**Value: up to 350.2 ms/step at 4 GPUs** (215.6 all-to-all + 134.6 permute),
≈4.2 MFU points. **Highest-value item identified.** Not started.

> **Correction (2026-11-05, §7.5 A3).** Head-parallel attention moves as much
> data as the current exchange: V slices to the head owners and attention
> outputs back are T × 1920 values per rank per block per micro-step, the same
> as today's Q/K all-to-all. Only the permute (134.6 ms) goes away outright; the
> 350 ms is an upper bound, reached only if the exchange hides behind compute.

### 2.2 The `fused_stack` permute kernel — ~85 ms at 4 GPUs

> **Done inside O39 (§1.20):** the block takes the all-to-all's buffer as it
> lands and reassembles it through a view; the 135.6 ms backward copy
> (`triton_poi_fused_stack_10`, which A0 places in the backward) is gone.

`_FusedBankGather`'s `.permute().reshape()` after the all-to-all runs at **~400
GB/s on a 3 TB/s machine** (0.84 ms per launch moving 336 MB, 160 launches,
134.6 ms/step) because it transposes.

**A fusion I reasoned through and rejected:** merging it with `_mixed_qkv`'s
final `transpose(1,2)` would not remove a copy — the intervening elementwise
work (qk_norm, alpha mix, tau) materialises `[B, n, H, D]` contiguously and the
transpose is a *view*, so there is exactly one materialising copy today. Fusing
moves it. The defect is kernel quality, not structure. A hand-written Triton
transpose, or arranging the all-to-all so a view suffices, is the open route.

### 2.3 O10 — activation checkpointing (**still cut**)

The only remaining headroom for a larger micro-batch at both h20/4-GPU (63.5 of
80 GiB, activations ~48 of it) and h16/1-GPU (68.35 of 80). Micro-batch is the
only lever measured in MFU **points** rather than milliseconds.

**Cut deliberately and the reason is the asymmetry with O11:** it trades
compute for memory, and nothing here has measured whether the micro-batch gain
beats the recompute cost. Given that O11's recompute turned out to cost more
than its micro-batch bought (§1.13), that caution now looks well placed — O10
should be measured, not assumed.

### 2.4 GEMM efficiency

| | achieved | ceiling | value |
|---|---:|---:|---:|
| h20, 4 GPU | 76.0% | 80–85% realistic | 61–130 ms |
| h16, 1 GPU | 81.4% | ~88% realistic | ~180 ms |

An earlier estimate of mine sized this against 90% of peak and was wrong — 752
TFLOP/s is a normal figure for large cuBLAS bf16 GEMMs on an H100 and the
practical ceiling is 80–85%, not 100%. Incremental tuning, not a defect.

### 2.5 S3c — `sink_score` as a per-vocabulary gather

`q` is not computed, it is gathered: `q = q_bank(token_ids)`. So
`sink_score[b,h,n] = q_bank[ids[b,n]] · k_sink[h]` depends only on the token id.
Precompute `sink_proj = q_bank_weight.view(vocab, H, D) @ k_sink` — a
`[50304, 20]` table, 4 MB — **once per optimizer step**, and `sink_score`
becomes a 1.3 MB gather instead of a reduction over 83.9 MB of `q`. Building it
reads 128.8 MB against the 83.9 it replaces, so it pays only when amortised
across micro-steps: 128.8 once against 83.9 × 8 = 671 MB. **5.2× traffic
reduction, growing with ga.**

Catch, stated so it is not discovered later: the backward must route
`d(sink_score)` into the bank as an index_add over token occurrences rather
than into a dense `dq`. That is machinery the bank already has, but it is real
work. Untested.

### 2.6 S5 — the bank path under compile

> **Retired (2026-11-06, §1.17–§1.18).** Since D2 the fused pair has no
> `@torch.compiler.disable`; dynamo traces it, collectives and scatter
> included, with no graph break, and the result is correct on CUDA. The 160
> breaks below were the unfused path's. O39 then takes the exchange out of the
> compiled block on purpose.

`ShardedBank.forward` carries `@torch.compiler.disable` (dynamo traces it but
the in-place scatter gives a **wrong gradient**), costing a graph break per
block — 160 per step at ga 8. D1 established that this graph break, not memory
parallelism, was the real Tier 2 regression. **Unsized.** Would need the
side-effecting backward restructured so dynamo can trace it safely.

### 2.7 S6 — hoist the id all-gather

> **Done inside O39 (§1.20):** one id all-gather per micro-step.

320 NCCL launches per step → 8, by gathering ids once per micro-step rather
than once per bank per block. The payload is ~1 MB so it is nearly all latency.
**~16 ms.** Small but cheap.

### 2.8 Row-sharded banks instead of column-sharded

Each rank owns a vocabulary slice; a token's embedding lives entirely on one
rank, so the gather produces full-width vectors and **no reassembly permute is
needed** (§2.2's 134.6 ms). Redistribution becomes a scatter by token index
rather than a clean reshape. The plan's §"Why the column split fits AME better
than it fits pkmf" argues for columns; this would revisit it against the
measured permute cost, which did not exist when that was written.

### 2.9 "triton glue" — the largest row with no plan against it

731.8 ms/step at h20/4 GPU (22.7% of kernel time) and 1283.3 ms at h16/1 GPU
(21.3%). It is RMSNorm, residual adds, the SwiGLU activation and the sink
merge: memory-bound elementwise work that inductor already fuses.

**Partly proven irreducible.** SwiGLU's backward alone is 228.1 ms at h16 and
touches ~5 × 268 MB per call over 512 calls = 687 GB/step, which is 229 ms at 3
TB/s — it matches its bandwidth bound to within 0.5%. It cannot be made faster,
only *smaller*, by recomputing instead of storing, which is O10 (§2.3).

The unexplored part is whether the remaining ~1000 ms contains fusion
opportunities inductor is missing — nothing has looked kernel by kernel. Given
S3b showed inductor's fusion decisions can invert a ranking, this is worth one
careful pass.

### 2.10 Options that exist in the codebase and are deliberately off

| flag | status |
|---|---|
| `compile_ce` | **Measured harmful 2026-11-03**, before this work. Buys ~1% step time and 3 GiB but shifts the loss one-directionally: over 300 steps at h20 it sat ABOVE the uncompiled twin in 50 of the last 50 steps, last-50 mean \|dloss\| 0.110 against 0.006. A one-directional shift is not the unbiased noise fusion produces. Not retried. The schema now rejects it together with `fused_linear_ce`, since O11 never builds the logits `compile_ce` would compile over. |
| `compile_loss` | Traces the model with the loss; leaves the logits crossing a graph boundary. Not used. |
| `compile_blocks_full` | **On** in every config here; took 4-GPU FSDP2 from 33.7% to 44.3% MFU on the dense audit. |

### 2.11 Not retried, worth one measurement each

- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — would recover ~2.5
  GiB of reserved-but-unallocated memory and perhaps let h16/4-GPU reach
  micro-batch 32 without O11. Refused throughout this work because it changes
  the allocator under measurements whose purpose is comparison; legitimate for
  a max-MFU run, which would then not be comparable.
- **h18 / h14** — interpolation between measured points 3.5 apart; low value.
- **O11 + micro-batch 32 at h20/4 GPU** — never run; h20's activations, not its
  logits, are the binding constraint there.
- **A 300-step loss-trajectory comparison for O11.** The 30-step gate cannot
  distinguish reassociation from a small one-directional shift the way the
  300-step `compile_ce` comparison in the plan did — and that comparison is
  precisely what got `compile_ce` rejected. Required before O11 is used for
  real training rather than throughput measurement.

### 2.12 Explicitly cut or deferred, unchanged

**Cut:** O10 (above), O16 chunked AdamW, O17 CPU offload (ruled out outright).
**Deferred:** O14/O15 fused gather, O19/O20 per-group LR, O21–O27 schedule
machinery.

### 2.13 The price a memory saving must beat (from the §1.15 sweep)

Every cell's best micro-batch is the largest that fits, and in five of six the
next one is blocked by the logits tensor. So any memory saving (O11, O10, a
fused kernel) is worth exactly **what the next micro-batch doubling returns,
minus its own cost**. The sweep bounds the first term. Each doubling has
returned 0.16–0.44× of the one before; carrying each cell's own decay one step
further puts the next doubling at **+1.7% to +4.3% relative**: h16 at 4 GPUs
+4.3%, every other cell +1.7–2.2%. At h16/4 GPU that agrees with the +4.7%
§1.13 backed out of a measured O11 run. These are extrapolations, not
measurements.

O11 v1 costs 7.0%, which is why it loses in every cell. **A logits-free loss
path costing under ~1.7% of the step would pay in every cell; under ~4%, at
h16/4 GPU only.** The untried route is a single fused linear + cross-entropy
kernel that computes each chunk's logits, softmax and gradient in one Triton
pass — so neither §1.13 v1's recompute nor v2/v3's separate unfused ops — the
approach of Liger-Kernel's `LigerFusedLinearCrossEntropyLoss`. Untested here.
Its cost on this model is the whole question, so it needs a gate run before
anything else.

---

## 3. Experiment log

Every job submitted for this work. Times are cluster-local (UTC+1); wandb
timestamps are one hour earlier. "—" means the run produced no MFU (failed,
cancelled, or a benchmark rather than a training run).

### 2026-11-04 — Tier 1 and Tier 2/3 construction

| time | job | what it tested | MFU | outcome / conclusion |
|---|---|---|---:|---|
| 20:05 | 32576 | **B1 baseline**, AME h20 4 GPU mb16 | 26.84% | The reference every later number is read against. |
| 20:10 | 32577 | **B2 dense ceiling**, H-balanced h20 4 GPU | 44.44% | AME reaches 60.4% of dense. Also validates the instrument. |
| 20:14 | 32578 | O9, gradient-sync suppression | 26.16% | +1.75%. Marginal; Tier 1's attribution gate then fired NEGATIVE. |
| 20:19 | 32579 | O31 per-submodule attribution, AME | 26.80% | Bank path ≈8% of step. The gate result Maciej overrode. |
| 20:52 | 32601 | O31 attribution, dense | 44.41% | Matched comparison for the attribution table. |
| 21:43 | 32623 | AME attribution rerun | 26.80% | Reproducible to 0.02 points. |
| 21:48 | 32624 | dense attribution rerun | 44.41% | Reproducible. |
| 22:40 | 32632 | **Tier 2+3**, `mem_parallel_size=4`, unfused | 26.55% | −1.2% vs B1. Mathematically right, throughput-neutral at ga 8. |
| 22:46 | 32634 | Tier 2+3 with attribution | 26.46% | Showed NCCL already overlapped behind attention. |
| 22:53 | 32638 | `mem_parallel_size=2` at 4 GPU | — | **OOM.** Only `mp = world_size` fits at h20. |
| 22:58 | 32649 | `mem_parallel_size=1` at 4 GPU | — | **OOM.** Tier 2 alone is memory-infeasible at h20. |
| 23:01 | 32652 | Tier 2+3 at 1 GPU, ga 128 | 25.04% | **+38.5%** vs 18.1%. My ~2% prediction was wrong by 20×; the miss is what exposed the ga-scaling law. |
| 23:16 | 32656 | baseline at micro-batch 4 | 18.28% | Comparand for the ga-scaling curve. |
| 23:23 | 32659 | Tier 2+3 at micro-batch 4 | 22.44% | +22.1%. Confirms value ∝ ga ÷ shard factor. |
| 23:49 | 32664 | **D3**, DDP + memory-parallel, mb4 | 22.99% | Maciej's directive. Loses to FSDP2+fused by 3.0%. |
| 23:55 | 32666 | **D2**, fused Q/K pair, mb16 | 27.41% | +3.4%. The fused pair is the win; DDP is not. |

### 2026-11-05 (early) — separating D2 from D3, and the sink

| time | job | what it tested | MFU | outcome / conclusion |
|---|---|---|---:|---|
| 00:01 | 32672 | DDP at micro-batch 16 | — | **OOM** in `cross_entropy`, allocating exactly 6.14 GiB. Led to `_loss_ctx`. |
| 00:05 | 32674 | FSDP2 + fused, mb4 | 23.52% | Decisive against DDP once confounds removed. Corrected an earlier claim of mine that DDP was fastest. |
| 00:23 | 32678 | DDP mb16 retry after the CE fix | — | Still OOM. Capacity result, not a gate failure. |
| 00:27 | 32680 | DDP at micro-batch 8 | 25.23% | Fits at 53.53 GiB; still short of FSDP2 at mb16. |
| 00:35–00:40 | 32681–32685 | attention micro-benchmarks | — | Decomposed the kernel; found the sink costing 876 ms/step. |
| 00:42 | 32687 | **S2 sink merge** | 30.24% | **+10.0%.** Loss trajectory unchanged. |
| 00:48 | 32688 | bench: n=2047 vs n=2048 | — | **479.9 ms for one token.** The S3 discovery. |

### 2026-11-05 (morning) — alignment, the h-sweep, O33, O11

| time | job | what it tested | MFU | outcome / conclusion |
|---|---|---|---:|---|
| 08:16 | 32771 | 1-GPU control, model n = 2047 | 28.80% | Control for the alignment A/B. |
| 08:16 | 32772 | 1-GPU, model n = **2048** | 33.36% | **+16.1%.** 100.3% transfer from benchmark to training. |
| 08:19 | 32774 | full attention decomposition, both lengths | — | ALiBi costs 59 ms; flex sits at the flash floor; the KV-column branch costs +267 ms. |
| 08:30 | 32776 | merge forms, eager vs compiled | — | **Ranking inverts under compile.** Source of S3b. |
| 08:31 | 32777 | 1-GPU aligned + kernel profile | 31.92% | Produced the 1-GPU phase table. |
| 08:47 | 32794 | phase attribution of that trace | — | 1347 ms of "optimizer" inside fwd/bwd = FSDP2 `foreach`, not AdamW. |
| 08:49 | 32796 | **S3b** merge rewrite, 1 GPU | 34.17% | +2.8%. |
| 08:49 | 32797 | correctness gate | — | Ran, but output lost to my own `tail` filter. **Preempted lukaskoz 32611 (11 h 49 m).** |
| 09:00 | 32806 | gate retry | — | Cancelled by me. **Preempted lukaskoz 32736 (1 h 55 m).** |
| 09:24 | 32820 | **4-GPU headline**: align + S3b | 34.42% | Below my 36.1–36.4% projection. 560 ms of attention removed, 412 ms off the step — the gap is communication attention used to hide. |
| 09:24 | 32822 | gate retry 2 | — | Output lost again; ran 10 min and printed no summary. |
| 09:26 | 32825 | **h16 at 1 GPU, mb 16** | **37.96%** | **Best result of the work.** |
| 09:36 | 32832 | h12 at 1 GPU, mb 32 | 34.46% | Below h16 — "smaller is better" is false; peak at h16. |
| 09:37 | 32834 | h20 1 GPU, FSDP2 off | 35.59% | +3.8%. 421 ms, not the 2281 ms I had sized. |
| 09:37 | 32836 | gate, full capture | — | **AME suite 73/73 pass.** memory_parallel failed one test on `EADDRINUSE` — infrastructure, not correctness. |
| 09:52 | 32841 | h16 1 GPU, FSDP2 off | 37.00% | **−2.3% — reverses sign vs h20.** Crossover is `ga`. |
| 09:58 | 32843 | **4-GPU kernel profile** | 34.52% | First ever. NCCL 509.8 ms total, only ~177 ms exposed. |
| 10:07 | 32847 | **O33** `reshard_after_forward=False` | **35.02%** | +1.7%. Best 4-GPU result. |
| 10:12 | 32853 | h16 4 GPU, micro-batch 32 | — | **OOM on exactly 6.14 GiB of logits.** The measurement that justified lifting O11. |
| 10:16 | 32856 | h16 4 GPU, mb 16 + O33 | 33.30% | Below h20 at 4 GPUs — ranking inverts with GPU count. |
| 10:22 | 32859 | h16 1 GPU kernel profile | 37.38% | Three of four largest rows already at hardware limits. |
| 10:46 | 32862 | **O11 v1** gate run, h16 1 GPU | 35.31% | −2.65 points; memory 68.35 → 60.65 GiB. |
| 10:46 | 32864 | **O11 v1**, h16 4 GPU mb 32 | 32.44% | **Fits** (62.61 GiB) but below mb 16's 33.30%. |
| 11:03 | 32870 | **O11 v2**, gradients-in-forward | 20.87% | My bug: fp32 contractions, 16× matmul penalty. |
| 11:15 | 32876 | **O11 v3**, model-dtype contractions | 30.38% | Fixed, but still below v1. **The recompute was cheaper than hand-rolling the softmax.** |

### 2026-11-05 (evening) — Maciej's own account: reproduction and the micro-batch sweep

QOS `mstefaniak_h100`, same node, `trainer.profile_timers=true` on every run.
Configs are the §6.1 files, with only `trainer.gradient_accumulation_steps`
varied (shown as micro-batch × ga). **Times here are job start**; the sweep's
21 jobs were all submitted 17:37–17:45 and ran as the 8-GPU cap allowed.

| time | job | what it tested | MFU | outcome / conclusion |
|---|---|---|---:|---|
| 17:11 | 33014 | CPU-only `srun` on h100a: env, FineWeb, HF cache | — | torch 2.7.1+cu128 from the account's own pixi env; data readable. |
| 17:16 | 33022 | **repro** h12, 1 GPU, 32 × 16 | 34.57% | Morning 34.46%. Peak 66.54 GiB, identical. |
| 17:22 | 33034 | **repro** h16, 1 GPU, 16 × 32 | 37.73% | Morning 37.96%. Peak 68.35 GiB, identical. |
| 17:28 | 33051 | **repro** h20, 1 GPU, 4 × 128, no FSDP2 | 35.45% | Morning 35.59%. Peak 66.87 GiB, identical. |
| 17:37 | 33065 | sweep: h20, 4 GPU, 4 × 32 | 30.23% | |
| 17:42 | 33078 | sweep: h12, 1 GPU, 8 × 64 | 29.57% | |
| 17:42 | 33066 | sweep: h20, 4 GPU, 8 × 16 | 33.48% | |
| 17:42 | 33079 | sweep: h12, 1 GPU, 16 × 32 | 33.01% | |
| 17:46 | 33080 | sweep: h12, 1 GPU, 64 × 8 | — | **OOM**: 12.28 GiB (the logits) with 70.82 GiB in use. |
| 17:47 | 33081 | sweep: h16, 1 GPU, 4 × 128 | 25.45% | |
| 17:47 | 33082 | sweep: h16, 1 GPU, 8 × 64 | 34.93% | |
| 17:47 | 33083 | sweep: h16, 1 GPU, 32 × 16 | — | **OOM**: 6.14 GiB (the logits) with 77.08 GiB in use. |
| 17:47 | 33084 | sweep: h20, 1 GPU, 1 × 512 | 16.51% | Overhead-bound: forward 12.2 s > backward 11.8 s. |
| 17:47 | 33085 | sweep: h20, 1 GPU, 2 × 256 | 31.13% | |
| 17:48 | 33086 | sweep: h20, 1 GPU, 8 × 64 | — | **OOM**: 1.54 GiB (the logits) with 79.08 GiB in use. |
| 17:54 | 33067 | **repro + sweep** h20, 4 GPU, 16 × 8 | 34.91% | Morning 35.02%. Peak 64.63 GiB, identical. |
| 17:58 | 33068 | sweep: h20, 4 GPU, 32 × 4 | — | **OOM**: 320 MiB with 79.12 GiB in use. |
| 18:01 | 33069 | sweep: h16, 4 GPU, 4 × 32 | 24.83% | |
| 18:05 | 33070 | sweep: h16, 4 GPU, 8 × 16 | 30.34% | |
| 18:09 | 33071 | **repro + sweep** h16, 4 GPU, 16 × 8 | 33.29% | Morning 33.30%; step 1.689 s in both. |
| 18:12 | 33072 | sweep: h16, 4 GPU, 32 × 4 | — | **OOM**: 6.14 GiB (the logits) with 75.18 GiB in use. Reproduces 32853. |
| 18:15 | 33073 | **first** h12 at 4 GPU: 8 × 16 | 24.66% | |
| 18:18 | 33075 | h12, 4 GPU, 16 × 8 | 28.53% | The controlled analogue of the other 4-GPU rows; not h12's best. |
| 18:21 | 33076 | h12, 4 GPU, 32 × 4 | **30.04%** | **Best h12 at 4 GPUs.** Weakest cell of the grid. |
| 18:24 | 33077 | h12, 4 GPU, 64 × 2 | — | **OOM**: 12.28 GiB (the logits) with 78.12 GiB in use. |

Each OOM is a `torch.OutOfMemoryError` on a 79.18 GiB card, with nothing else
wrong in the log; all six are the intended upper brackets.

### 2026-11-05 (evening) — the dense control (§1.16)

Same account and node, `trainer.profile_timers=true` on every run. Configs are
the §6.1 files plus `common.attention.type=gqa trainer.mem_parallel_size=null`,
with the micro-batch × ga shown. Each run keeps the §6.2 settings that are not
AME-specific: sequence length 2049, `compile_blocks_full`, O33 at 4 GPUs, O37
for h20 at 1 GPU. Submitted serially 18:49:53–19:04:46 with 20 s gaps. One
`--test-only` probe per shape at 18:49:07 found five GPUs free (lukaskoz's
33098 on 1 GPU and wojszalek's 32991 on 2 were running), so both shapes could
start at once and neither probe named a job to preempt. **Times here are job
start.**

| time | job | what it tested | MFU | outcome / conclusion |
|---|---|---|---:|---|
| 18:49 | 33106 | dense h20, 4 GPU, 8 × 16 | 41.96% | |
| 18:53 | 33107 | dense h20, 4 GPU, 16 × 8 | **44.47%** | **Best dense h20 at 4 GPUs**, same micro-batch as AME's best. Reproduces B2 (32577, 44.44%). |
| 18:56 | 33108 | dense h20, 4 GPU, 32 × 4 | — | **OOM**: 6.14 GiB (the logits) with 73.05 GiB in use. |
| 18:57 | 33118 | dense h12, 1 GPU, 16 × 32 | 38.77% | |
| 18:58 | 33119 | dense h12, 1 GPU, 32 × 16 | **40.32%** | **Best dense h12 at 1 GPU**, same micro-batch as AME's best. |
| 18:58 | 33120 | dense h12, 1 GPU, 64 × 8 | — | **OOM**: 12.28 GiB (the logits) with 72.62 GiB in use. |
| 18:59 | 33122 | dense h12, 1 GPU, 128 × 4 | — | **OOM**: 384 MiB with 78.85 GiB in use. Redundant second bracket. |
| 19:00 | 33123 | dense h16, 1 GPU, 8 × 64 | 40.84% | |
| 19:00 | 33124 | dense h16, 1 GPU, 16 × 32 | 43.31% | AME's best micro-batch; AME is at 87.1% of this. |
| 19:01 | 33125 | dense h16, 1 GPU, 32 × 16 | **44.91%** | **Best dense h16 at 1 GPU**: twice AME's best micro-batch, which OOMs for AME. |
| 19:02 | 33126 | dense h16, 1 GPU, 64 × 8 | — | **OOM**: 128 MiB with 79.13 GiB in use. |
| 19:02 | 33127 | dense h20, 1 GPU, 4 × 128 | 40.97% | AME's best micro-batch; AME is at 86.5% of this. |
| 19:03 | 33128 | dense h20, 1 GPU, 8 × 64 | 43.33% | |
| 19:04 | 33129 | dense h20, 1 GPU, 16 × 32 | **44.98%** | **Best dense run of the sweep**: four times AME's best micro-batch. |
| 19:05 | 33131 | dense h20, 1 GPU, 32 × 16 | — | **OOM**: 6.14 GiB (the logits) with 74.24 GiB in use. |
| 19:09 | 33109 | dense h20, 4 GPU, 64 × 2 | — | **OOM**: 320 MiB with 79.10 GiB in use. Redundant second bracket. |
| 19:11 | 33110 | dense h16, 4 GPU, 8 × 16 | 39.73% | |
| 19:14 | 33111 | dense h16, 4 GPU, 16 × 8 | 42.78% | AME's best micro-batch; AME is at 77.8% of this. |
| 19:17 | 33112 | dense h16, 4 GPU, 32 × 4 | **43.68%** | **Best dense h16 at 4 GPUs**: twice AME's best micro-batch. |
| 19:20 | 33113 | dense h16, 4 GPU, 64 × 2 | — | **OOM**: 256 MiB with 78.98 GiB in use. |
| 19:22 | 33114 | dense h12, 4 GPU, 8 × 16 | 33.57% | |
| 19:25 | 33115 | dense h12, 4 GPU, 16 × 8 | 37.75% | |
| 19:27 | 33116 | dense h12, 4 GPU, 32 × 4 | **39.34%** | **Best dense h12 at 4 GPUs**, same micro-batch as AME's best. |
| 19:30 | 33117 | dense h12, 4 GPU, 64 × 2 | — | **OOM**: 12.28 GiB (the logits) with 72.73 GiB in use. |

All eight failures are `torch.OutOfMemoryError` with nothing else wrong in the
log; the last job ended at 19:32.

### 2026-11-05/06 (night) — the §7 run: Phase A, h20 on 4 GPUs

Same account and node, `trainer.profile_timers=true` on every training run.
Base for every A/B: `configs/ame/h20_4gpu.yaml` (35.06%, job 33502, this
session). **Times are submit times** (cluster clock).

| time | job | what it tested | carries | MFU | outcome / conclusion |
|---|---|---|---|---:|---|
| 23:56 | 33502 | **A0, AME**: kernel profile of the 4-GPU best | §6.1 h20/4-GPU | **35.06%** | Base of this run (34.91–35.02% before). All-to-all 196.6 ms/step, 100% exposed; bank path ≈ 500 ms of critical path (§1.17). |
| 23:58 | 33504 | gate (2 GPUs): new `test_bank_compile.py`, `test_ame.py`, `test_memory_parallel.py` | — | — | 3/3, 73/73, 28/28. **The compiled bank path is correct on CUDA**, at mp 1 and over NCCL (§1.18). |
| 00:01 | 33507 | **A0, dense twin**: kernel profile, `gqa`, no banks | same recipe | 44.51% | Attention at parity with AME's flex; dense GEMMs 172.7 ms dearer (§1.17). |
| 00:11 | 33509 | **O40**, parameters unsharded across micro-steps | base + O40 | 34.75% (median 35.28%) | Normal steps −19 ms, but two forward spikes (+614, +834 ms at steps 14 and 26). Not kept; the code stays, opt-in (§1.19). |
| 00:18 | 33512 | **O39**, bank exchange prefetched on a side stream, depth 2 | base + O39 | **35.79%** | **+2.1%** (step 2.791 s). Backward −56 ms, forward unchanged: far less than the exchange costs (§1.20). Peak 64.61 GiB, unchanged. |
| 00:19 | 33515 | gate (2 GPUs) of O39 v1: + `test_bank_prefetch.py`, O40 tests | — | — | 15 passed (CPU bit-identity of O39 at mp 2 and 4, O40, O9). Two CUDA tests failed: the new prefetch test by my own bug (an `import torch._dynamo` inside the worker shadowed `torch`), and `test_bank_compile`'s NCCL test with a `c10::Error` whose traceback my output filter cut. AME suite 73/73. Cancelled at 19 min in the memory_parallel suite: superseded by 33533. |
| 00:24, 00:25 | — | O39 kernel profile, two submits | — | — | **Not submitted**: GitHub rejected the cemetery push (`remote: fatal error in commit_refs_rules`) twice, while reporting no incident. run_exp restored the tree both times; nothing was queued. |
| 00:29 | 33527 | **O39 v1 kernel profile** | base + O39 v1 | — | Exposed all-to-all 196.6 → 59.6 ms and the backward copy gone, but the side stream ran eager gathers (169.8 ms) and scatters (158.6 + 86.0 ms), 3–11× slower than the compiled kernels they replaced, and the compute stream's GEMMs ran 178.6 ms slower beside them (§1.20). |
| 00:37 | 33530 | **O39 v2**: side-stream gather and scatter compiled, one all-to-all per layer | base + O39 v2 | | running |
| 00:38 | 33533 | gate (2 GPUs) of O39 v2 + O38(b), full logs kept | — | — | running |

### Jobs on this account NOT part of this work

Listed so the log is exhaustive and the next person does not have to re-derive
which `sacct` rows belong here.

| jobs | what | why excluded |
|---|---|---|
| 32465–32538 | `brrrrr` | Another user's runs, 09:56–19:32 on 2026-11-04, predating this work. Their checkpoints are the ~35 GB driving the disk-quota problem in §5. |
| 32541, 32542, 32543, 32544, 32555, 32559 | `ame_*`, `mfu_*` | **Cancelled before running** and resubmitted as 32576/77/78/79. They carried a pre-fix snapshot in which a DTensor diagnostics bug was deterministic for AME on multi-GPU. Never started (`Start = None`). |
| 32818, 32845 | `update_pixi` | Environment helpers, not experiments. |
| 32867, 32886, 32890 | `h16_chinchilla` | 4-GPU array job on the shared `j321m` account that **I did not submit** — it appeared mid-session and I left it alone. |
| 32682, 32683 | `ame_bench_attn` | Covered above as the 32681–32685 benchmark block. |

**On `mstefaniak` (2026-11-05), before the evening block:** 32926, 32927,
32979, 32980, 32983 (CPU-only checks), 32981 (`update_pixi`, the env build),
32982 (`envcheck`, failed in 1 s and rerun as 32983), and 32987 `mfu_h4` and
32990 `ame_h20` (1-GPU smoke tests, 15:48 and 15:52). These were account
bring-up, not this work.

**Inside the dense range:** 33121 and 33130 are not this account's jobs, and
33104 and 33105 are the IDs the two `--test-only` probes printed (no job was
created).

### Correctness gates (not throughput runs)

| gate | result |
|---|---|
| §1.4.1 CPU toy-model equivalence, Tiers 2+3 | Bit-identical without clipping (float64); one fp32 ulp with clipping, traced to `clip_gradient`'s `.float()` meeting a different summation order. |
| O8 divisor, `mem_parallel_size` 1 vs 4 | Pass, with a negative control that catches the wrong divisor as exactly 4.0×. |
| `tests/test_memory_parallel.py` | 28/28. |
| `tests/test_ame.py` on CUDA (job 32836) | **73/73**, including flex-vs-reference equivalence — the gate for S2 and S3b. |
| `tests/test_fused_linear_ce.py` | 14/14. |
| O11 loss trajectory, 30 steps | Identical from step 2; 2 bf16 ulps at steps 0–1. |
| O11 grad-norm trace | 3.4e-6 relative at step 0 → 6.0e-3 at step 29, sign-flipping (not biased). |

---

## 4. Historical log — how the plan changed, and why

### 2026-11-04, evening — Tier 1 closes with a negative gate

Tier 1 measured B1 and B2 and produced the per-submodule attribution that was
supposed to *authorise* Tiers 2–3: if the bank gradient path were not a top-two
term after O9, the plan said re-rank before building anything. It was ~8% of
step time. **The gate fired NEGATIVE.**

**Maciej overrode it explicitly** and directed both tiers be built regardless.
That override is why everything in §1.2–1.3 exists. In hindsight the gate was
right about the bank path — it is 1.0% of the step today — but Tier 3 is also
what makes h20 fit at 4 GPUs at all, so "unimportant for throughput" and
"load-bearing for capacity" were both true at once.

### 2026-11-04, late — Tiers 2+3 land throughput-neutral, and the ga law

Tier 2+3 measured 26.55% against B1's 26.84% at the pinned test bed. I predicted
~2% for the 1-GPU variant; it measured **+38.5%**, wrong by 20×. That miss was
productive: it exposed that everything the tiers delete is *per-micro-step*
work, so their value scales with `ga ÷ shard factor`. The pinned test bed (ga 8)
is close to the worst case for them.

### 2026-11-04, late — Maciej's three directives

1. *"Memory parallelism should give the biggest speedup — investigate and fix."*
   Led to D1 (the defect is the graph break from `@torch.compiler.disable`, not
   memory parallelism) and D2 (the fused Q/K pair, +3.4%).
2. *"Try running the core model in DDP and parallelise just the memory banks,
   like pkmf."* Built and measured; **loses by 3.0%** and OOMs at mb 16.
   Recorded as a negative result rather than tuned toward.
3. *"Try again DDP — it should fit, there might be a bug."* There was: the
   cross-entropy autocast upcast (§1.6), worth 3.07 GiB. DDP still did not reach
   FSDP2's mb 16, but the bug was real and the fix is in the default path.

### 2026-11-04, late — the >35% loop

*"If they won't exceed 35% MFU, work continuously in a loop... keep logs in
`interactive_ame.md`."* This created the working log and the strategy backlog
(S1–S7). It also set the pattern for the rest of the work: hypothesis →
micro-benchmark → training A/B → record, with every number gated first.

### 2026-11-05, 00:42 — the sink, and a bug class

The attention decomposition showed the **sink**, not flex and not ALiBi, was
876 ms/step. Two structural causes, both about shape rather than arithmetic: a
ragged `KV_LEN` and a `torch.where` evaluated n² times. S2 removed both (+10%).

### 2026-11-05, 00:48 — the same bug class on the other axis

S2's `KV_LEN = 2049` made me look at `Q_LEN`, and the training runs had been at
**n = 2047** all along — one token short of the block grid, in every run
including every baseline. 479.9 ms for one token. **The largest single win of
the work (+16.1% at 1 GPU), and it had been present since before Tier 1.**

### 2026-11-05, 08:49 and 09:00 — I preempted two of lukaskoz's jobs

Two jobs submitted for this work destroyed ~13 h 45 m of another user's
compute: 32611 (11 h 49 m) and 32736 (1 h 55 m), under
`PreemptMode=CANCEL`/`GraceTime=0`.

**Root cause, which is not "forgot to check".** The check was run, at 08:15,
when three GPUs were free and 1-, 2- and 3-GPU jobs all showed no `Preempts:`
line. That result was then treated as a standing property of the cluster. It was
a snapshot; another user filled the node at ~08:38. A second failure compounded
it: for job 32806 the `--test-only` and the `sbatch` were issued in the **same
shell command**, so the submit had already gone out by the time its own gate was
readable. A gate evaluated after the action it gates is not a gate.

I stopped submitting and reported it. **Maciej's response changed the
constraint:** lukaskoz auto-submits to fill free compute and preemption is
configured for exactly this, so it is acceptable here. The procedural rule
still stands and is now in memory: one check per submit, in its own tool call,
read before composing the command, no exemption for small jobs — **both jobs
that caused this were 1-GPU jobs.**

### 2026-11-05, 09:24 — the 4-GPU headline undershoots, and says why

I projected 36.1–36.4% from two independent routes. It measured **34.42%**.
Per rank the attention saving was 560 ms; the step fell 412 ms. The missing 148
ms is **communication that attention had been hiding** — Tier 1's finding
running backwards. Tier 1 measured NCCL already overlapped behind flex
attention, which is why cutting NCCL traffic 55% bought nothing; remove the
attention and the overlap stops being free.

This invalidated the whole backlog's sizing, since every estimate assumed
savings land one-for-one. It is why Tier 4 re-ranked rather than continuing.

### 2026-11-05, 09:26 — Maciej's scope change: h-sweep and a 40% target

*"Evaluate h12 and h16 on 1 GPU. Aim for 40% MFU... but test 4 GPUs on h20
still because it is the established baseline."* The goal became a **number, not
a tier list**: exceed 40%, ideally match dense 44.44%.

I added the arithmetic that frames it: dense parity requires AME to be **5%
faster than dense in wall clock**, because it is 3.768 GFLOPs/token against
3.97 and buys parameters with gathers. 40% looked reachable; 44.44% did not.

h16 then produced the best number of the work (37.96%), and h12 showed the
sweep is not monotonic.

### 2026-11-05, 10:30 — the permitted list reaches a ceiling

The h16 profile showed three of its four largest rows at hardware limits:
attention where SDPA flash also sits, SwiGLU matching its bandwidth bound to
within 0.5%, GEMM at 81.4%. 40% needed −316 ms; GEMM tuning offered ~180.

I reported that **the two items blocking 40% were precisely the two on the cut
list**, for the third independent time that day, and asked rather than
implementing — §0 says "do not reintroduce them to hit a number", so lifting
them needed to be an explicit scope change rather than a quiet one.

### 2026-11-05, 10:40 — Maciej lifts O11, keeps O10 cut

*"O11 removes a measured allocation with no compensating cost... O10 stays cut:
it trades compute for memory and nothing has measured whether the micro-batch
gain beats the recompute cost."* Plus a sizing correction I had not made: **the
4-GPU step's non-NCCL kernels are 2715.9 of 2893 ms, so today's kernels cap
4-GPU MFU near 36.8%** even with communication perfectly hidden — O11 helps by
enabling a bigger micro-batch, not by shortening the step. And: queue it
alongside the empirical items, not ahead.

**The outcome vindicated the asymmetry.** O11's memory claim held exactly; its
MFU claim did not, because its recompute cost more than the micro-batch bought
— which is the precise argument for keeping O10 cut until measured.

### 2026-11-05, 11:15 — three implementations of O11, and what the gate caught

v1 (checkpoint) cost 2.65 points. v2 (gradients-in-forward) was far worse —
20.87% — from my own bug: the fp32 accumulation dtype reached the contractions,
making the step's largest matmul fp32. v3 fixed that and still lost to v1.
**The recompute was cheaper than hand-rolling the softmax.**

The gate caught two defects that a weaker test would have passed: `.float()`
silently **downcasts** float64, so the float64 gate would have been measuring
float32 reassociation; and the "logits never materialised" probe hooked the
module while the new code calls `F.linear` directly, passing vacuously. A test
whose mechanism has drifted from the code is worse than no test.

### 2026-11-05, ~17:10 — Maciej moves the work to his own account

Maciej had a new entropy account, `mstefaniak`, configured on his name, and
asked that the work stop using the shared `j321m` one. The ask: work from a
separate worktree on `ame_optimize` so his checkout stays on `ame_main`, check
that the account works, and reproduce the MFU for h12/h16/h20 on 1 and 4 GPUs
(six runs) with the time profiler on. I made the worktree (`~/code/nano-ame`)
and checked the account from a CPU-only `srun` on h100a (job 33014: the
account's own pixi env, FineWeb readable). I also wrote the one config that did
not exist: h12 at 4 GPUs, as the controlled analogue of the other two 4-GPU
rows (mb 16, `mem_parallel_size` 4, O33), with a header saying mb 32 was the
likelier optimum. Submits went out one at a time, one `--test-only` probe each.
The three 1-GPU runs reproduced within 0.23 points with identical peak memory.

### 2026-11-05, ~17:35 — Maciej widens it: a micro-batch sweep, all at once

This came before the 4-GPU reproductions started. Submit everything at once,
and for each (model, GPU count) find the best micro-batch with a worse one on
each side, or an OOM above. At least 3 runs per cell and 18 in total, ending
with one table of every MFU. I chose 4 per cell (24 runs), reusing the three
finished reproductions because their configs were identical, and queued the
4-GPU jobs first so they would not wait behind a wall of 1-GPU ones.

Two mechanical problems on the way; neither queued anything wrong:

- **The repo's `^key` grid could not express it.** `TrainerSchema` rejects a
  `^gradient_accumulation_steps` key at compose time. So each point became a
  plain `run_exp.py` call with overrides, submitted serially, because every
  invocation commits and pushes from the same repo.
- **The first serial loop died after one attempt.** fabric forwarded the loop's
  stdin to the remote shell (`OSError: Socket is closed`, before cloning).
  Rerun with `< /dev/null`; all 21 submitted 17:37–17:45.

Under "submit all at once" the per-submit probe became one probe per job shape:
a 1-GPU and a 4-GPU `--test-only` immediately before the batch. Both named only
`lukaskoz` jobs as preemption candidates; `wojszalek`'s 32991 never appeared and
ran throughout. Preempting `lukaskoz` is acceptable here (§4, 08:49). Which of
his jobs Slurm cancelled cannot be read from this account, because `sacct`
shows only one's own jobs. Two long `px_*` jobs the probes listed, 32955
(2 h 45 m) and 32956 (≥1 h 42 m), vanished in windows when a job of mine started
on a full node.

### 2026-11-05, ~17:49 — entropy stops accepting new connections from the laptop

New SSH connections began timing out at TCP connect (`193.0.109.82:22`), while
already-open sessions kept working. That is a firewall drop, most likely set
off by this session's own burst: ~21 submissions in 8 minutes, each its own
SSH session, then one `scp` per log. The jobs were unaffected. I stopped making
new connections; Maciej switched to a VPN at ~18:03 and the new address
connected at once. After that, logs came back through one SSH session as a tar
stream.

### 2026-11-05, 18:27 — the sweep closes the micro-batch question

All six cells bracket cleanly, and every §6.1 configuration is already at the
largest micro-batch that fits (§1.15). h12 at 4 GPUs, measured for the first
time, peaks at mb 32 (30.04%). That is above the mb 16 I had configured as the
controlled analogue, and it is the weakest cell of the grid, as §6.4 predicted.
The result changes no recommended setting. It changes where the next gain has
to come from: a memory saving cheaper than the next micro-batch doubling
returns (§2.13), or the forward and backward kernels themselves.

### 2026-11-05, 18:46 — Maciej: save the tables, and the same sweep for dense

"Save that results to .md, now do the same experiments but for dense not ame",
then at 18:47 "save exactly those two tables 1:1". The two AME tables went
verbatim into [ame_microbatch_sweep.md](ame_microbatch_sweep.md), indexed in
`documents/README.md`. For dense I reused the 24 configs with two overrides
instead of writing dense configs, so the control differs from AME in the
attention layer only (§1.16), and moved each ladder up one doubling where dense's
missing bank state left room. Probes at 18:49:07, then 24 serial submits
18:49:53–19:04:46 with 20 s gaps, to stay under the connection rate that
tripped the firewall at ~17:49.

### 2026-11-05, 19:38 — "what 8 GPU cap?"

At 19:05 I had reported the remaining jobs as "queued behind my 8-GPU cap".
Maciej asked whether that meant 8 physical GPUs and whether jobs simply pend.
Both hold: h100a has 8 GPUs, and QOS `mstefaniak_h100` lets one user hold 8 at
a time (`MaxTRESPerUser gres/gpu=8`, up to 128 jobs submitted); the rest pend as
`QOSMaxGRESPerUser`. Everything had already been submitted at once, so nothing
changed.

### 2026-11-05, 19:41 — the dense sweep closes

All 24 had ended by 19:32: 16 completed and 8 OOM, every cell bracketed, no
reruns needed. Results in §1.16. While polling I once misread the clock and took
an empty queue for jobs that had vanished, and first counted 15 completed and 9
OOM; both were caught before reporting. The dense tables went into a separate
`dense_microbatch_sweep.md`. Two slips of mine in `~/code/nano`: I copied my
README over the main tree's (it held only my own earlier row, so nothing was
lost; doc changes there have been patches on top since), and
`git checkout-index --temp` left a stray `.merge_file_*` in the repo root (a copy
of the README, removed). That tree had been reorganized while the jobs ran
(test configs to `configs/ame_tests/`, six best-MFU configs in `configs/ame/`);
I left it alone.

### 2026-11-05, 19:57 — one file for both sweeps

"Have you appended dense tables also to the same .md doc?" (no, a separate
file), then at 19:58 "Merge them together into one .md". Both sweeps now sit in
`ame_microbatch_sweep.md`, all four tables unchanged, and
`dense_microbatch_sweep.md` is gone. Commits 03510a74, ce4dcad0 and dec338f3
(20:05–20:07, from Maciej's checkout) put the sweep docs, this report's §1.15 and
the config reorganization on `ame_main`. 20:11 "Tldr", answered. 20:17 the
session's context was compacted.

### 2026-11-05, 20:28 — what the dense model is

"What h20 dense models have you tested and created configs for? What
feed-forward and attention?" No dense config files exist; the dense runs are the
AME configs with overrides. Both models are the H-balanced h20 with SwiGLU
(dff = 4 × dmodel); only the attention differs. While checking
`configs/mfu_audit/` I found that the 45.57% `ame_implementation.md` quotes as
the `gqa` h20 baseline was measured with GPT-2-arch overrides (4.122
GFLOPs/token), not on this model. At 20:31 Maciej narrowed the check to the
best-MFU dense run and the `configs/ame/` files. At 20:35 he asked for active
parameters (MoE-style, a bank counting one row per token): 620.35M dense,
587.69M AME. At 20:37 that table went into `ame_implementation.md` as its last
section.

### 2026-11-05, 20:42 — where documents live and get committed

I had written that section only in `~/code/nano`. "You know where is your
worktree and branch you should work on?", then at 20:43 "find your worktree":
`~/code/nano-ame` on `ame_optimize`, and the section went there too. 20:44
"commit what?", then at 20:45 the rule: ".md documents always read and commit on
ame_main", "you can also commit on ame_optimize", "so both". Also at 20:45 a
stop-hook check of this report's brief found nothing here after 18:27.

### 2026-11-05, 20:55 — the update stays lean

"What are you doing?" and "two commits shouldn't take that long" (20:55): I had
spent the time collecting inputs before writing anything. At 20:58 Maciej set
the scope: no MFU improvement has been tested since the last update, so what
belongs here is the measured micro-batch tables. I dropped a new §2 proposal I
had prepared (a smaller bank optimizer state, to fit micro-batch 8 at h20 on 1
GPU), and added §1.16, the dense rows in §3 and these entries.

### 2026-11-05, 21:11 — Maciej: a plan to 45% at h20, 4 GPUs first

Iterative work pauses for a plan. The brief: AME h20 only; at least 45% MFU on 4
GPUs and on 1 GPU; 4 GPUs first, because there the banks and their backward are
spread over the GPUs; `ame_optimize` in its own worktree; entropy `h100`, with
`lukaskoz`'s jobs expected to be preempted; methods, leads, logs and progress
recorded here, regularly and precisely; nothing implemented or tested yet. I
wrote §7 and the text for `/goal`. Writing it turned up two things the plan has
to carry. 45% is above dense's own MFU on this recipe (44.47% and 44.98%), so
AME has to run faster than dense does today. And §2.1's head-parallel attention
moves as much data as the current exchange, so its 350 ms is an upper bound, not
an estimate.

### 2026-11-05, 22:11 — the plan runs unattended

Maciej approved §7 with one change: it is expected to run overnight with no
intervention or decisions from him. §7 now says so. §7.1 records it, §7.4's
approval list became the run's out-of-scope list, §7.8 says how the run
decides, handles trouble and ends, §7.10 collects what is left for him, and the
`/goal` text carries the same terms.

### 2026-11-05, 22:19 — the goal points at this report

Maciej: the `/goal` text did not make this report the reference he asked for,
the record of what was done and planned that the run must read and keep
updated, and it repeated what §7 already says. §7.9 now holds a short goal
that sends the run to read the whole report first and to record back into it,
and points at §7.4 and §7.8 for the rules. §7.1 says the same.

### 2026-11-05, 23:55 — the §7 run starts

Maciej set the §7.9 goal at ~23:50 and left the run unattended. Read the report
in full first, then A0 as §7.5 orders. lukaskoz held five 1-GPU jobs and
wojszalek one 2-GPU job (not preemptable); each 4-GPU probe named only
lukaskoz's jobs, so they went ahead (§7.4).

### 2026-11-05, 23:58 — is the compiled bank path even correct?

Reading `memory_parallel.py` before profiling turned up a question the record
never answered. D1 found that letting dynamo trace the bank path gives a wrong
`main_grad`, which is why `ShardedBank.forward` carries
`@torch.compiler.disable` — but D2's `FusedBankPair.forward`, which every run
since 32666 uses, does not, and a local `torch._dynamo.explain` showed it traced
with no graph break. If D1 was right, every MFU number since D2 was measured
on a model whose bank gradient is wrong. Checked locally first: `aot_eager`
exact, CPU inductor wrong on the k banks only, the miscompile visible in the
generated C++ (§1.18). Then the GPU gate (33504): correct on CUDA. The record
stands; D1's explanation goes in the corrections table, and S5's premise
(§2.6) is retired.

### 2026-11-06, 00:05 — A0: the 45% target is the bank path and AME's glue

AME against its dense twin (§1.17): attention at parity, AME's GEMMs 172.7 ms
cheaper, and about 500 ms of critical path in the bank exchange, about 160 ms
in AME's attention-side math. So 45% does not need a faster backbone than
dense's; it needs AME's own overhead brought to dense's level. Lead order
changed accordingly: the exchange first (O39, absorbing A1 and A2), O40 as a
cheap side bet on the exposed all-gather (A4), then compile autotuning and the
glue.

### 2026-11-06, 00:15 — O40 inconclusive, O39 v1 +2.1%

O40 lost by the mean and won by the median (§1.19); not kept, flag retained.
O39 v1 gained 2.1% where the exchange it moved costs ~500 ms. Rather than tune
blind, a kernel profile of O39 (§1.20).

### 2026-11-06, 00:24 — GitHub refuses the cemetery push

Two `run_exp.py` submits failed in `version_code`: `remote: fatal error in
commit_refs_rules`, with githubstatus.com reporting all systems operational.
run_exp restored the worktree both times and nothing was queued. Per §7.8 I
retried once, then wrote a fallback that ships the worktree as a tar stream and
skips the push (`submit_direct.py`, in the session scratch space; the W&B key
travels on stdin, never in an argv). It was not needed: the third attempt at
00:29 went through.

### 2026-11-06, 00:36 — O39's side stream ran the slow kernels

The profile (§1.20): exposure gone, but the gathers and the scatter ran eager
on the side stream, 3–11× slower than the compiled versions they replaced, and
slowed the compute stream's GEMMs by 178.6 ms. v2 compiles them and goes back
to one all-to-all per layer. The first gate of O39 (33515) had a bug in its own
CUDA test (an `import torch._dynamo` inside the worker shadowing `torch`) and a
failure in `test_bank_compile`'s NCCL test whose traceback my output filter
cut; I cancelled it (superseded code) and resubmitted with full logs (33533).
O38(b) — gather from the fp32 master, no bf16 bank copy — was written
meanwhile for Phase B: bit-identical on CPU after a bug of mine where
`materialize()` rebound the master each step and left the bank optimizer
stepping a tensor that never got a gradient.

### Corrections I made to my own earlier claims

Recorded because several changed what was worked on next.

| claim | correction |
|---|---|
| Tier 2 at 1 GPU would gain ~2% | Measured **+38.5%**, wrong by 20×. Exposed the ga-scaling law. |
| "DDP is the fastest configuration measured" | Confounded comparison (D2+D3 vs neither). FSDP2+fused beats DDP+fused by 3.0%. |
| "0.0625 is exactly one bf16 ulp at 7.1" | It is two; the ulp on [4,8) is 0.03125. |
| The sink merge costs 241 ms | Eager-benchmark artefact; the compiled model was paying ~103.6 ms. |
| GEMM headroom ≈190 ms (90% of peak) | 90% is not a figure cuBLAS reaches; 61–130 ms at a realistic 80–85%. |
| 4-GPU align+S3b will reach 36.1–36.4% | Measured 34.42%. Exposed that the step is communication-bound. |
| FSDP2 overhead at 1 GPU ≈2281 ms removable | 421 ms materialised, 18%. |
| `log = 151.5 ms` is a 5% opportunity | A host timer overlapping GPU timers that already account for the step. No opportunity. |
| "lukaskoz's 11-hour job has finished" | It was preempted — by my job. |
| Head-parallel attention removes 350.2 ms at 4 GPUs (§2.1) | It moves the same volume as the Q/K exchange it replaces; only the 134.6 ms permute goes away outright. §7.5 A3. |
| Tracing the bank path gives a wrong `main_grad` because "the side effect does not survive AOTAutograd" (D1, 2026-11-04) | AOTAutograd keeps it (`aot_eager` exact). The 7.3e-3 is an inductor **CPU** codegen bug in the k banks' fused scatter; on CUDA the traced path is correct (job 33504, §1.18). Every run since D2 has used the traced path. |

---

## 5. State at pause

**Code, committed on `ame_optimize` and merged to `ame_main` 2026-11-05**
(7 commits, `ed4952cc`..`04c960f3`): `src/core/memory_parallel.py` (new),
`src/core/ame.py` (sink merge, S3b), `src/core/model.py` (O11, `parameter_groups`),
`src/core/trainer.py` (Tier 2/3 hooks, `_loss_ctx`, O11 wiring),
`src/core/distributed_training.py` (DDP, O33), `src/definitions.py`,
`conftest.py`, `main.py`, plus configs and tests.

**Recommended defaults for AME today:**
`sequence_length: 2049` (always — it is a correctness fix as much as a speed
one), the S2 sink merge and S3b merge form (both in `ame.py`, unconditional),
`mem_parallel_size = world_size` at 4 GPUs, `reshard_after_forward: false`,
`fused_linear_ce: false` (memory tool, not a speed one). **Micro-batch:** the
largest that fits, which is what every §6.1 configuration already uses; the
§1.15 sweep confirmed it in all six cells.

**Cluster:** the evening block ran on `mstefaniak`, not `j321m`; this
paragraph is about the shared account. j321m's home hit its disk quota during
the session. I deleted my
own profiler traces (1.9 GB). The cemetery is **93 GB**, of which **~35 GB is
two `brrrrr_*` runs' checkpoints** from 2026-11-04 (8 × 4.4 GB `.distcp`) that
are not mine. The filesystem has 7.8 TB free — this is purely the per-user
limit, and it will recur.

**Companion documents:**
[ame_implementation.md](ame_implementation.md) — what the AME layer is;
[ame_experiments.md](ame_experiments.md) — the scientific program, unrun, and the next
thing to do;
[ame_microbatch_sweep.md](ame_microbatch_sweep.md) — all 48 micro-batch runs, AME and
dense, with step, tokens/s, peak memory and forward/backward time.

**Archived, but still the source for two things.**
[archive/ame_optimization_plan.md](archive/ame_optimization_plan.md) §3 is the ledger
these figures are computed from, and the only place carrying wandb run ids and config
paths per job; §4 is the decision log §4 here is distilled from.
[archive/ame_optimization.md](archive/ame_optimization.md) holds the evidence behind each
O-number that §1.0 only names. Also
[archive/interactive_ame.md](archive/interactive_ame.md), the working log this report was
written from, and [archive/ame_review_and_fixes.md](archive/ame_review_and_fixes.md),
the 2026-11-03 review and its fixes.

---

## 6. Best known configuration per model size

Everything below is a **measured** run, not a recommended-by-inference
combination. Figures are mean over steps ≥ 15, as everywhere else. Dense
ceiling for the % column is B2 = 44.44%.

### 6.1 The table

| model | GPUs | best MFU/GPU | step | tokens/s | micro-batch × ga | peak GiB | % of dense | job | config |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| **h12** | **1** | **34.46%** | 3.240 s | 323,631 | 32 × 16 | 66.54 | 77.5% | 32832 | `ame_tests/h12_1gpu_mb32.yaml` |
| **h12** | **4** | **30.00%** | 0.930 s | 1,127,285 | 32 × 4 | 53.22 | 67.5% | 33076 | `ame_tests/h12_4gpu_mb16_o29.yaml` + `trainer.gradient_accumulation_steps=4` |
| **h16** | **1** | **37.96%** | 5.924 s | 177,001 | 16 × 32 | 68.35 | **85.4%** | 32825 | `ame_tests/h16_1gpu_mb16.yaml` |
| **h16** | **4** | **33.30%** | 1.689 s | 621,015 | 16 × 8 | 44.53 | 74.9% | 32856 | `ame_tests/h16_4gpu_mb16_o29.yaml` |
| **h20** | **1** | **35.59%** | 11.226 s | 93,404 | 4 × 128 | 66.87 | 80.1% | 32834 | `ame_tests/h20_1gpu_mb4_nofsdp.yaml` |
| **h20** | **4** | **35.02%** | 2.852 s | 367,691 | 16 × 8 | 64.63 | 78.8% | 32847 | `ame_tests/h20_4gpu_mb16_mp4_o29.yaml` |

GFLOPs/token, which is why the MFU columns are not directly comparable across
rows: h12 **1.053**, h16 **2.121**, h20 **3.768**. A smaller model carries less
compute per token, so equal MFU does not mean equal efficiency of the same
work — and raw tokens/s across rows is a different model's tokens.

Every row was re-measured on 2026-11-05 evening on Maciej's own account, within
0.23 points and with identical peak memory; the h12/4-GPU row comes from that
block (job 33076; §1.15 quotes it as 30.04%, by the sweep's own aggregation).
Every row is also the largest micro-batch that fits: the next power of
two OOMs in all six (§1.15).

Curated copies of all six rows are in `configs/ame/` (`h12_1gpu.yaml` …
`h20_4gpu.yaml`), each identical to the config in the last column except run
name, tags and job name. Every other AME config moved to `configs/ame_tests/`
on 2026-11-05.

### 6.2 What each best configuration contains

All six carry **`sequence_length: 2049`** (S3 — the model sees 2048, so
flex_attention's row-blocks are 128-aligned), the **S2 sink merge**, the
**S3b reduction form** of the sink logit, and `compile_blocks_full: true`.
They differ in the sharding and head settings:

| | `mem_parallel_size` | sharding | O33 `reshard_after_forward` | O37 FSDP2 off | O11 fused CE |
|---|---:|---|---|---|---|
| h12, 1 GPU | 1 | FSDP2 (degenerate) | — | no | no |
| h12, 4 GPU | 4 | FSDP2 + column-sharded banks | **false** (not A/B'd at h12) | n/a | no |
| h16, 1 GPU | 1 | FSDP2 (degenerate) | — | **no** — tested, −2.5% | **no** — tested, −7.0% |
| h16, 4 GPU | 4 | FSDP2 + column-sharded banks | **false** | n/a | **no** — tested at mb 32, −2.6% |
| h20, 1 GPU | 1 | **none** (O37) | n/a | **yes**, +4.2% | no |
| h20, 4 GPU | 4 | FSDP2 + column-sharded banks | **false**, +1.7% | n/a | no |

**The two inversions are the reason this table cannot be collapsed into one
recommendation.** O37 (no FSDP2 at world size 1) helps h20 by +4.2% and *hurts*
h16 by −2.5%, because the discriminator is `ga` — h20 runs 128 micro-steps
against h16's 32, so FSDP2's per-micro-step staging is 4× smaller at h16 while
autocast's per-region weight-cast cost is not. And O11 is a memory tool
everywhere, never a speed one.

### 6.3 Choosing between them

**Best MFU overall: h16 at 1 GPU, 37.96%.** Also the best efficiency per GPU of
any configuration measured, and the only one above 37%.

**Best for a fixed 4-GPU allocation: h20, 35.02%.** Note it beats h20 at 1 GPU
(35.59%) only on wall clock, not on MFU — the 4-GPU run is 3.9× faster per step
but 0.6 points *less* efficient per GPU.

**Best wall-clock throughput: h16 at 4 GPUs, 621,015 tokens/s** — 1.7× h20's
4-GPU figure, but those are smaller tokens (2.121 vs 3.768 GFLOPs each).

**The general rule the h-sweep established:** AME's MFU *falls* with GPU count,
because the bank machinery (id all-gather, activation all-to-all, reassembly
permute — 350.2 ms/step at 4 GPUs) does not exist at world size 1 and grows
while per-rank compute shrinks. If MFU is the objective and the model fits, use
fewer GPUs; if wall clock is the objective, use more and accept the loss. The
loss is largest for the smallest model: from 1 to 4 GPUs h20 loses 1.5%, h16
11.8% and h12 13.1% (same-session best points, §1.15), because h12 has the
least compute per token to set against that fixed cost.

### 6.4 Not measured

Stated so nothing here is read as a complete sweep.

- **Micro-batches between powers of two.** At global batch 512 there are none
  (micro-batch × ga × GPUs = 2⁹), so §1.15's optima are exact for this batch
  size; testing, say, 24 or 48 would need a different global batch.
- **Any model at 2 or 3 GPUs.** The crossover between h16's 1-GPU 37.96% and
  its 4-GPU 33.30% is unlocated; 2 GPUs would bound it.
- **h12 and h16 at 4 GPUs without O33**, so their O33 deltas are inferred from
  h20's rather than measured.
- **h14, h18.** The sweep has three points 3.5 MFU apart with a clear peak at
  h16; these would interpolate rather than discover.

---

## 7. Plan — AME h20 to 45% MFU, 4 GPUs first (from 2026-11-05 evening)

*Written 2026-11-05 from Maciej's brief at 21:11. Nothing in it has been
implemented or run yet. Progress is recorded where §1–§6 record it, and
summarized in §7.7.*

### 7.1 The brief

Maciej's instructions, kept here as the plan's terms of reference:

- **Target: AME h20 at ≥ 45% MFU/GPU, on 4 GPUs and on 1 GPU.** h20 only; no
  other model size is tested under this plan.
- **4 GPUs first, then 1 GPU.** His reasoning: at 4 GPUs the memory banks'
  parameters and their backward are spread over the GPUs, so 4-GPU training has
  more room to parallelize the bank work and can potentially beat 1 GPU's MFU.
- **Code:** branch `ame_optimize`, in its own worktree (`~/code/nano-ame`).
  `~/code/nano` on `ame_main` stays Maciej's.
- **Cluster:** entropy, `h100` partition, as `mstefaniak`. Preempting
  `lukaskoz`'s jobs is expected and accepted: he auto-submits to occupy all
  free GPUs, and the QOS preemption exists for that.
- **This document is the reference and the record.** Read it in full before
  starting, and again before each new lead: what was implemented and its
  measured effect (§1), what was tried and rejected and why (§1, §3, §4 and
  the corrections table), the open leads (§2), the best configurations (§6).
  Nothing recorded here is repeated or contradicted without new evidence.
  Methods, leads, experiment logs and progress go back into it, updated
  regularly and precisely, in the same form as §1–§6.
- **Process:** iterative, as in §3–§4: measure, change one thing, A/B it, keep
  or revert, record.
- **Unattended, overnight** (added 22:11): the plan is expected to run through
  the night with no intervention or decisions from Maciej. §7.8 says how the
  run decides, handles trouble and ends; §7.10 collects what it leaves to him.

### 7.2 Where h20 stands, and what 45% means

| | best measured | step | micro-batch × ga | step needed for 45% | to remove | dense h20, same recipe (§1.16) |
|---|---:|---:|---:|---:|---:|---|
| **4 GPUs** | 35.02% (32847); 34.91% re-measured (33067) | 2.852 s | 16 × 8 | 2.219 s | 633 ms (22.2%) | 44.47% at 2.363 s, mb 16 |
| **1 GPU** | 35.59% (32834); 35.45% re-measured (33051) | 11.226 s | 4 × 128 | 8.879 s | 2.347 s (20.9%) | 44.98% at 9.347 s, mb 16; 40.97% at mb 4 |

Configs: `configs/ame/h20_4gpu.yaml` and `configs/ame/h20_1gpu.yaml`.

**45% is above dense's own MFU on this recipe.** AME does 5.0% fewer FLOPs per
token than dense (3.768 vs 3.965 GFLOPs), so matching dense's wall clock would
give it 42.26% at 4 GPUs, and 42.75% against dense's best at 1 GPU. Reaching 45%
takes two kinds of work, and neither suffices alone:

1. **Remove AME's excess over dense:** about 0.50 s per step at 4 GPUs, and
   about 1.0 s at 1 GPU at equal micro-batch (§7.3).
2. **Make the shared backbone faster than dense runs it today:** GEMM
   efficiency, the memory-bound "glue", the loss path, launch overhead. These
   would lift dense too, and nothing in §1 has tried them.

### 7.3 Where AME's extra time goes — what the existing timers already say

The time profiler's forward and backward totals (§1.15, §1.16), fitted per
micro-step as a fixed cost plus a cost per sequence. Each fit uses two
micro-batches, so these are estimates for A0 and B0 to check:

| | fixed per micro-step, fwd + bwd | per sequence, fwd | per sequence, bwd |
|---|---:|---:|---:|
| AME, 4 GPUs (mb 8 → 16) | 10.75 + 4.50 ms | 5.97 ms | 15.16 ms |
| dense, 4 GPUs (mb 8 → 16) | 11.38 + 7.75 ms | 5.09 ms | 12.11 ms |
| AME, 1 GPU (mb 2 → 4) | 4.59 + 7.58 ms | 5.57 ms | 13.19 ms |
| dense, 1 GPU (mb 4 → 8) | 2.20 + 6.44 ms | 5.58 ms | 12.27 ms |

- **4 GPUs: the excess scales with tokens, and 78% of it is in the backward.**
  Per sequence AME costs 0.88 ms more forward and 3.05 ms more backward. Over the
  128 sequences each GPU processes per step that is +112 ms and +390 ms, which
  accounts for the measured 498 ms gap (2.861 vs 2.363 s). AME's fixed cost per
  micro-step is no worse than dense's. The kernel profile names the candidates:
  the dQ/dK all-to-all (215.6 ms per step, forward and backward), the reassembly
  permute (134.6 ms), and the FSDP2 parameter all-gather. That all-gather was
  219.2 ms of kernel time before O33, which removed its 106 ms backward half;
  113.2 ms of forward all-gather remains, and how much of it is exposed is
  unmeasured.
- **What this means for 4 GPUs first.** The bank backward itself is already
  split by columns and is about 1% of the 4-GPU step (`bank / embedding`, §1.3),
  because gradients go into `main_grad` with `index_add_` and no dense
  per-micro-step gradient exists. What 4 GPUs adds is the exchange around the
  banks, and that is where the gain is. What 4 GPUs already gives is memory: each
  GPU holds a quarter of the 43.2 GiB of bank state, so h20 runs micro-batch 16
  there (4 at 1 GPU), the regime where dense reaches 44.5%.
- **1 GPU: two roughly equal parts.** AME's fixed cost per micro-step is
  12.16 ms against dense's 8.64 ms, +451 ms over 128 micro-steps. Its per-sequence
  backward is 0.92 ms more, +468 ms over 512 sequences. Together that is 0.92 s
  of the measured 1.01 s gap at micro-batch 4. Micro-batch 4 itself also costs:
  dense goes from 40.97% at mb 4 to 44.98% at mb 16.

### 7.4 Rules for every step

- **Fixed workload and accounting.** AME h20 as in `configs/ame/`, global batch
  512 sequences of 2049 tokens (the model sees 2048), 30 steps, LR 2^-12,
  FineWeb, bf16. MFU is the mean per-step MFU over steps ≥ 15 at 3.768
  GFLOPs/token and 989 TFLOP/s per GPU, as everywhere in this report. **Out
  of scope for this unattended run** (Maciej's decision, so they go to §7.10 as
  proposals): the model, the batch, the sequence length, the FLOPs accounting,
  the numerics class (fp8, for example), and the two cut items, O10 (activation
  checkpointing) and O17 (CPU offload).
- **One change per A/B**, against the current best, on the same account and
  day. `trainer.profile_timers=true` on every run; `kernel_profile` only in
  dedicated attribution runs, never in an A/B.
- **Gates before a change is kept:** `tests/test_ame.py` on CUDA (73/73 today),
  `tests/test_memory_parallel.py` (28/28), the 30-step loss trace within bf16
  noise of the base run, and a 300-step loss comparison for anything that
  changes numerics (the standard that rejected `compile_ce`, §2.10). A faster
  run that fails a gate is a failed experiment.
- **Cluster.** `mstefaniak` only. `sbatch --test-only` before every submit, 1
  GPU included. Preempting `lukaskoz` is expected and accepted. No other user's
  job is to be preempted (`wojszalek`'s were not preemptable in every probe so
  far); if a probe names one, wait. Keep new SSH connections few: one session
  where possible, ≥ 20 s between submits (§4, ~17:49).
- **Submission.** `run_exp.py` with the main checkout's env python
  (`~/code/nano/.pixi/envs/default/bin/python`), never through `pixi run`; never
  kill it mid-flight; never edit the worktree while it runs, because it commits,
  pushes and checks out.
- **Records, after every experiment block:** jobs in §3 (start time, job, what
  it tested, which improvements it carries, MFU, outcome and conclusion); the
  decision in §4, with any instruction from Maciej; the method in §1 when a
  change is kept, with its measured effect; leads re-ranked in §2; §5 and §6 when
  the best changes; the progress table in §7.7. Documents are edited in
  `~/code/nano` and committed on `ame_main` and `ame_optimize`, staging only my
  own changes. Code is committed on `ame_optimize`.

### 7.5 Phase A — h20 at 4 GPUs

Base: `configs/ame/h20_4gpu.yaml`. Needed: −633 ms per step.

| step | what | value (estimate) | status |
|---|---|---:|---|
| **A0** | **Attribution first.** Kernel profile (steps 12–13) and time profiler of the current best and of its dense twin, the same config with `common.attention.type=gqa trainer.mem_parallel_size=null`. Output: a per-kernel table of AME minus dense, so every lead below gets a measured cost instead of an estimate, and the exposed (not overlapped) part of each NCCL collective. | measurement | **done** (§1.17): bank path ≈ 500 ms of critical path, AME glue ≈ 160 ms, attention at parity, AME GEMMs 172.7 ms cheaper than dense |
| **A1** | S6, hoist the id all-gather (§2.7): 320 NCCL launches per step → 8. | ~16 ms | **done** inside O39 (§1.20) |
| **A2** | The reassembly permute (§2.2): a Triton transpose, or an all-to-all layout in which a view suffices. | ~85 ms (≤ 134.6) | **done** inside O39: a view, folded into the consumer (§1.20) |
| **A3** | **The bank exchange, redesigned.** Three candidates, compared on paper and in a microbenchmark before any is built. **(a)** Deduplicate by token id before the exchange (O15, deferred so far): a micro-batch of 32,768 tokens holds fewer distinct ids than tokens, and only distinct rows need to move, in both directions. The value scales with tokens ÷ distinct ids, which is the first thing to measure on FineWeb. **(b)** Row-sharded banks (§2.8): full-width rows, so a scatter by token index replaces the transposing permute; rows balanced across ranks by `id mod 4`. **(c)** Head-parallel attention (§2.1), which moves the same volume as today (T × 1920 values per rank per block per micro-step, 125.8 MB at mb 16), so its value is the permute plus whatever exchange it can overlap with compute. | 85–350 ms, with A2 | **in progress** as O39: the overlap route, which (c) cannot match because AME's q/k need no activations (§1.20) |
| **A4** | FSDP2 forward all-gather: whatever A0 shows exposed; explicit forward prefetch. | ≤ ~113 ms | O40 tried (§1.19): inconclusive, retest on O39 |
| **A5** | S5, the bank path under compile (§2.6): 160 graph breaks per step. | unsized (A0) | **retired**: no graph break since D2, and correct on CUDA (§1.18) |
| **A6** | S3c, `sink_score` as a per-vocabulary gather (§2.5). | unsized (A0) | not started |
| **A7** | Shared backbone: GEMM efficiency (§2.4); one kernel-by-kernel pass over the "triton glue" (§2.9, 731.8 ms, partly at its bandwidth bound); a Liger-style fused linear + cross-entropy kernel (§2.13). | 61–130 ms + unsized | not started |

Micro-batch 32 is out of reach at 4 GPUs without O10: at mb 16 activations are
~48 GiB of the 64.6 GiB peak (§2.3), and dense h20 OOMs at mb 32 on 4 GPUs too.

**Arithmetic, stated now so it is not discovered late.** The sized items at
their upper bounds (A1 + A2/A3 + A4 + A7's GEMMs: 16 + 350.2 + 113.2 + 130 ms)
remove 609 ms, which gives 44.5%: still short of 45%. At central values they
remove about 300–450 ms (39–42%). So 45% at 4 GPUs also needs the unsized items
(A5, A6, the glue) to deliver.

### 7.6 Phase B — h20 at 1 GPU

Base: `configs/ame/h20_1gpu.yaml` (no FSDP2, O37). Needed: −2.347 s per step.
Whatever A1–A7 changes in shared code applies here too; it is re-measured, not
assumed.

| step | what | value (estimate) | status |
|---|---|---:|---|
| **B0** | Attribution: time profiler and kernel profile of AME and its dense twin at mb 4, locating the +451 ms fixed and +468 ms per-sequence excess (§7.3). | measurement | not started |
| **B1** | **Micro-batch 8.** It OOMs today needing about 1.4 GiB more than the card has; by AME's own memory growth (3.4 GiB per sequence) its peak would be ~80.6 GiB. In order: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (§2.11, ~2.5 GiB of fragmentation); then a smaller bank state, **O38**. The bank state is 18 bytes per bank parameter, 43.2 GiB at h20: keep Adam's first moment in bf16 (−4.8 GiB), or gather from the fp32 master and drop the bf16 bank copy (−4.8 GiB, exact numerics), or block-wise 8-bit optimizer states (−14.4 GiB; a new dependency, so a §7.10 proposal, not part of the run). A bf16 second moment is ruled out: the bank AdamW runs at torch's default β2 = 0.999, so a step moves v by ~0.1%, less than half a bf16 step (0.2–0.4%), and v would stop changing. Micro-batch 16 would need ~108 GiB by the same growth: out of reach. | +2.2% (AME's decay, §2.13) to +5.8% (dense mb 4 → 8, §1.16): 0.24–0.62 s | not started |
| **B2** | **Fixed cost per micro-step** (12.16 ms against dense's 8.64, × 128): S5 (2,560 graph breaks per step at ga 128); CUDA graphs (`mode="reduce-overhead"`), which O37's FSDP2-free setup makes plausible once the graph breaks are gone. | up to ~0.45 s AME-specific, plus part of the shared 1.1 s | not started |
| **B3** | **Per-sequence backward excess** (+0.92 ms per sequence), located in B0 among: the bank scatter (`index_add_` after an fp32 cast; duplicate ids contend for the same rows, and pre-reducing by id is O14/O15), the sink backward and the KV-shift backward. | up to ~0.47 s | not started |
| **B4** | Shared backbone, as A7. | unsized | not started |

**Arithmetic.** B1–B3 at their estimates remove 1.16–1.53 s (39.7–41.2%), and
they are not additive: B1 halves the micro-steps B2 acts on. 45% at 1 GPU needs
another ~0.8–1.2 s. That has to come from shared-backbone work (B4), or from
micro-batch 16, which is reachable only if Maciej lifts O10 (selective
checkpointing of the memory-heavy, compute-light glue would be the variant to
propose) or O17 (CPU offload of the bank optimizer state). Both are §7.10 proposals, not
part of the run.

### 7.7 Progress

| date | step | best, 4 GPUs | best, 1 GPU | note |
|---|---|---:|---:|---|
| 2026-11-05 | plan written | 35.02% | 35.59% | nothing run under this plan yet |
| 2026-11-06 00:05 | A0 | 35.06% (re-measured, 33502) | — | dense twin 44.51% |
| 2026-11-06 00:25 | O40, O39 v1 | **35.79%** (O39 v1, 33512) | — | O40 not kept |

### 7.8 Running unattended

**The plan runs overnight with no input from Maciej.** No questions, no waiting
for confirmation. Every choice it leaves open (which lead next, which A3 design,
keep or revert, whether to repeat a run) is mine to make by §7.4's rules, and
each one goes into §4 with its reason.

**Out of scope for the run.** These need Maciej's decision, so they are written
up in §7.10 as proposals and not done: the items on §7.4's out-of-scope list; new
third-party dependencies (the 8-bit optimizer states in B1, for example); any
code change on `ame_main`.

**Operational trouble is handled, not escalated:**

| situation | response |
|---|---|
| jobs pending in the queue | wait, with a background watcher on one SSH connection; do not resubmit |
| a probe names a job of any user other than `lukaskoz` | do not submit; wait and probe again |
| my job fails for an infrastructure reason (node, NCCL timeout, preemption) | resubmit once, record both jobs in §3 |
| new SSH connections time out or are refused | back off at least 15 minutes, then try a single connection; a new IP (VPN) needs Maciej, so it is not an option |
| home or cemetery disk quota | delete only my own profiler traces and scratch output |
| a gate fails | revert the change, record it as a failed experiment (§1, §3, §4), move to the next lead |
| a failure points to a bug in code already on `ame_main` | record it prominently in §4 and §7.10, leave `ame_main` alone, work around it on `ame_optimize` if the fix is local, continue |

**Keeping the record current.** The report is updated and committed on both
branches after every experiment block and at least every 2 hours, so the state
can be read in the morning without the transcript.

**The run ends only when** 45% is reached on both 4 GPUs and 1 GPU, or when
every step in §7.5 and §7.6 has been measured or shown infeasible with evidence.
In the second case it ends by writing, in this section, the ceiling analysis
(what each lead delivered, what is left, what it would take) and completing
§7.10. One failed lead, a long queue or an SSH block is not an end condition.

### 7.9 The `/goal` text

Short on purpose: the rules live in §7.4 and §7.8, and the goal points at them.

```text
Raise AME h20 training MFU to at least 45% per GPU on entropy H100s: on 4 GPUs first, then on 1 GPU.

documents/notes/ame_optimization_report.md is the reference and the record for this work. Read it in full before starting: what was already implemented and its measured MFU effect (§1), what was tried and rejected and why (§1, §3, §4, corrections), the open leads (§2), the best configurations (§6), and the plan with its rules (§7). Follow §7, and do not repeat or contradict recorded results without new evidence. Keep the report current as you go: every experiment in §3, every decision in §4, kept methods in §1, leads in §2, progress in §7.7, committed after every experiment block and at least every 2 hours (§7.4).

This runs unattended overnight; Maciej is not available. Never stop to ask or wait: decide within §7's rules, record why in §4, and put anything that needs his approval in §7.10 instead of doing it (§7.8).

Done when the report shows a measured AME h20 run at >= 45.00% MFU/GPU on 4 GPUs and one on 1 GPU, both passing the §7.4 gates, with code and report committed (§6, §7.7). Otherwise, only after every lead in §7.5 and §7.6 has been measured or shown infeasible: when §7.8 holds the ceiling analysis and §7.10 the proposals for Maciej, committed.
```

### 7.10 For Maciej — decisions the run leaves to him

The run adds to this list instead of waiting. Each entry says what it would buy,
why it needs a decision, and where the evidence is.

| proposal | why it needs a decision | expected value | source |
|---|---|---|---|
| Lift O10 for selective checkpointing of the glue at h20, 1 GPU | O10 is cut | a route to micro-batch 16 at 1 GPU, where dense measures 44.98% (40.97% at mb 4) | §7.6 |
| O17: CPU offload of the bank optimizer state | O17 is cut | the other route to micro-batch 16 at 1 GPU | §7.6 |
| Block-wise 8-bit optimizer states for the banks (an O38 variant) | a new third-party dependency | −14.4 GiB at h20, 1 GPU | §7.6 B1 |
