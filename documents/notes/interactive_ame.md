# AME MFU improvement loop — working log

**Target: AME h20 training MFU > 35% per GPU on entropy 4xH100.**
Standing instruction (Maciej, 2026-11-05): keep trying strategies until it
clears 35%; log what was tried and what is planned here so progress is legible.

This file is the *working* log — hypotheses, attempts, dead ends, next steps.
The durable record of finished results stays in
[ame_optimization_plan.md](ame_optimization_plan.md).

---

## Where we start

| configuration | job | MFU/GPU | step | peak GiB |
|---|---|---:|---:|---:|
| AME baseline B1, mb16 | 32576 | 26.82% | 3.723 s | 60.6 |
| AME: FSDP2 + mem-par + fused, mb16 | 32666 | 27.41% | 3.643 s | 61.90 |
| **AME best: + sink merge, mb16** | **32687** | **30.23%** | **3.302 s** | **63.50** |
| AME, FSDP2 + mem-par + fused, mb4 | 32674 | 23.49% | 4.250 s | — |
| AME, DDP + mem-par + fused, mb8 | 32680 | 25.23% | 3.957 s | 53.53 |
| AME, DDP + mem-par + fused, mb4 | 32664 | 23.02% | 4.337 s | 39.75 |
| **dense H-balanced h20 (the ceiling)** | 32577 | **44.44%** | 2.364 s | 43.4 |

**Gap to target: 27.41% -> 35% needs a 27.7% relative speedup**, i.e. the step
has to come down from 3.643 s to **2.853 s — 790 ms to find.**

## Where the 790 ms could come from

From the Tier 1 matched comparison against the dense twin (jobs 32579/32601,
and the forward/backward split in 32623/32624):

| term | AME | dense | AME excess |
|---|---:|---:|---:|
| **attention kernel** | **1390.7 ms** | 506.9 ms | **+883.8 ms (2.74x)** |
| — of which forward | 730.4 | 139.2 | **+591.2 (5.25x)** |
| — of which backward | 661.0 | 366.0 | +295.0 (1.81x) |
| GEMM | 1264.4 | 1298.4 | -34 (exonerated) |
| NCCL | 936.4 | 309.7 | +626.7 — **mostly removed by Tiers 2+3** |

**One term dominates and it is attention.** +883.8 ms against the 790 ms needed:
closing even two thirds of the attention gap clears the target on its own.
Everything else on the list is small change by comparison.

**Why AME's attention is 2.74x dense:** it runs `flex_attention` rather than
FlashAttention, because it needs an ALiBi distance bias and a learned sink key,
and PyTorch's flash backend accepts neither. The twist recorded in Tier 1 is
that `learnable_slopes=False` is the **default** — so AME is paying for
differentiable slopes it is not using.

---

## Strategy backlog

Ordered by expected value. Status: PLANNED / RUNNING / DONE / REJECTED.

| # | strategy | expected | status |
|---|---|---|---|
| S1 | Measure flex vs an SDPA/flash floor at the real h20 shape | sizes the prize exactly | **DONE — found the sink, see below** |
| S2 | **LSE sink merge** — keep flex, take the sink OUT of the kv axis and merge it afterwards through the log-sum-exp | **-876 ms predicted** | **IMPLEMENTED, measuring** |
| S3 | Tune the flex path in place — block-mask caching, `kernel_options` (BLOCK_M/BLOCK_N), a cheaper `score_mod` | unknown, cheap to try | PLANNED |
| S4 | `reshard_after_forward=False` on the 587M backbone now that the banks are out of FSDP2 | removes the backward all-gather, small | PLANNED |
| S5 | Get the bank path back under compile without the side-effecting backward | recovers what is left of the graph break | PLANNED |
| S6 | Hoist the id all-gather (320 NCCL launches -> 8) | ~95 ms, overlapped so likely invisible | PLANNED |
| — | Activation checkpointing (O10), fused-linear-CE (O11), chunked AdamW (O16), CPU offload (O17) | — | **REJECTED** — cut by the original goal |

## Log

### 2026-11-05 00:35 — entering the loop
Best is 27.41%. DDP explored and parked: it fits at micro-batch 8 (53.53 GiB,
25.23%) and 4 (39.75 GiB, 23.02%), OOMs at 16, and loses to FSDP2 at every
shape where both run. The CE-inside-autocast bug found on the way is fixed and
benefits any future autocast run.

Next: S1, to size the attention prize before engineering against it.


### 2026-11-05 00:36 — S1 result: it is the sink, not flex, and not ALiBi

`scripts/bench_attention.py` on one H100 at the real shape (B=16, H=20, n=2048,
D=64, bf16), scaled by 20 layers x 8 micro-steps to one optimizer step:

| kernel | fwd ms | bwd ms | **per step** |
|---|---:|---:|---:|
| AME flex, ALiBi + sink as kv column 0 | 4.629 | 3.693 | **1331.6 ms** |
| flex, ALiBi fp32, **no sink** | 0.752 | 2.097 | **455.8 ms** |
| flex, ALiBi bf16, no sink | 0.748 | 2.096 | 455.1 ms |
| flex, no score_mod at all, causal only | 0.545 | 1.940 | 397.5 ms |
| SDPA flash, causal, no bias (the floor) | 0.629 | 2.168 | 447.5 ms |

**Three things fall out, and two of them overturn what the plan assumed.**

1. **The sink costs 876 ms per step** — more than everything else in attention
   put together. One extra key.
2. **ALiBi costs 58 ms.** The distance bias, which the whole flex backend exists
   to support, is nearly free.
3. **flex with ALiBi (455.8) already beats the flash floor (447.5) to within
   2%.** Flex is not slow. The Tier 1 conclusion that "AME pays 2.74x for
   attention to keep a capability it is not using" was pointing at the wrong
   capability: it is not the differentiable slopes, it is the sink.

Also settled: **`flash_attn` is not installed** on entropy, so the S2 I had
planned (flash with native `alibi_slopes`) was never available. It does not
matter — flex already matches flash once the sink is out.

**Why one key costs 876 ms.** Two structural reasons, neither arithmetic:
`_with_sink` makes `KV_LEN` 2049, one past flex's 128-wide block alignment, so
every block goes ragged; and the sink needs a `torch.where` inside `score_mod`,
which flex evaluates **once per score** — n^2 per head — to special-case a
single column.

### 2026-11-05 00:40 — S2 implemented: merge the sink through the log-sum-exp

Attention over `{sink} u sequence` is the softmax-weighted merge of attention
over each part, and flex returns the log-sum-exp needed to do it exactly:

    numerator   = out_seq * exp(lse_seq) + exp(q . k_sink) * v_sink
    denominator = exp(lse_seq)           + exp(q . k_sink)

shifted by the running max. O(B*H*n*D) elementwise, no n^2 term. The sink keeps
its exact semantics — no distance bias, visible to every query, `tau` already
folded into `q`.

`KV_LEN` is now 2048 and `score_mod` is two ops with no branch.

**Gate:** the repo's own backend-equivalence tests
(`test_the_two_backends_compute_the_same_function`, parametrised on
`sink` x `shift`, and `test_the_backends_still_agree_under_bf16`) pass at
rtol 1e-5 / atol 1e-6 — flex against the reference path, which materialises the
sink the old way. 64 AME tests green.

**Projected:** 3643 ms - ~870 ms = ~2773 ms/step -> **~36.0% MFU**, which would
clear the target. Measuring now.


### 2026-11-05 00:41 — S2 measured and tuned: 1331.6 -> 679.5 ms/step

Three iterations on the same idea, each measured on one H100:

| version | fwd | bwd | **per step** |
|---|---:|---:|---:|
| original: sink as kv column 0 | 740.7 | 590.9 | **1331.6** |
| LSE merge, written as the ratio | 277.2 | 732.3 | 1009.5 |
| **LSE merge, written as a lerp** | **171.2** | **508.3** | **679.5** |
| (reference: sinkless + `return_lse`) | 120.8 | 335.0 | 455.8 |

**Diagnostics that made the second iteration possible** (same job):

| | ms/step |
|---|---:|
| ragged `KV_LEN = n+1`, no branch | 949.8 — so raggedness alone is **+495** |
| ragged `KV_LEN = n+1`, with branch | 1208.9 — the `torch.where` is **+261** |
| aligned, `return_lse=True` | 455.8 — asking flex for the LSE is **free** |

So the LSE route was sound and the ratio form was the mistake: upcasting the
full `[B, H, n, D]` output to fp32 and building five tensors that size cost 156
ms forward and 403 ms backward. The lerp identity

    out = out_seq + alpha * (v_sink - out_seq),   alpha = sigmoid(s - lse)

is the same arithmetic over `[B, H, n]` for the weight — 1/64th the elements —
and leaves the big tensor in its own dtype. Then written as a broadcast
multiply-add rather than `torch.lerp`, because lerp wants matching shapes and
would expand `v_sink` to full size first.

**652 ms saved so far.** The remaining 224 ms over the sinkless floor is the
merge's own forward (+50) and backward (+173, the sigmoid and the
`bhnd,hd->bhn` einsum that feeds gradients back to `q` and `k_sink`). Note the
benchmark calls `_attend_flex` **uncompiled**, so those ops are separate eager
kernels there; under `compile_blocks_full` inductor should fuse them into the
tail of the attention kernel, so the training number should beat 679.5.

Naive projection from 652 ms: 3643 -> 2991 ms/step = **33.4% MFU**. Short of 35,
but the projection ignores the fusion above. Measuring for real now.

**Next if 35% is not cleared:** S3 (the remaining 224 ms of merge overhead — the
einsum's backward is the biggest piece and could fold into the score_mod),
then S4/S5/S6 from the backlog.


### 2026-11-05 00:47 — S2 measured in training: **27.41% -> 30.23%**

Job **32687**, the pinned test bed, sink merge on top of the fused mem-parallel banks.

| | MFU/GPU | tokens/s | step | peak GiB |
|---|---:|---:|---:|---:|
| B1 baseline | 26.82% | 281,541 | 3.723 s | 60.6 |
| + mem-parallel + fused | 27.41% | 287,729 | 3.643 s | 61.90 |
| **+ sink merge** | **30.23%** | **317,389** | **3.302 s** | **63.50** |

**+12.73% over B1 cumulatively**, and the loss trajectory is unchanged
(11.1875 -> 7.0938, the same values B1 logged). 341 ms off the step, against
652 ms the uncompiled benchmark suggested — the gap is the merge fusing into
the compiled block, and the benchmark's shape differing from the real one (see
below).

**Short of target by 4.77 points.** 35% needs the step at 2.852 s, so there are
**450 ms** left to find.

### 2026-11-05 00:48 — S3 hypothesis: the real run's Q_LEN is not 128-aligned

The benchmark has been running at `--seq 2048`. **The training run does not.**
`common.sequence_length: 2048` gives a batch width of 2048, and
`_preprocess_input` takes `batch[:, :-1]` for next-token targets, so the model
sees **n = 2047** — and `2047 = 15*128 + 127`, one short of flex's block grid.

That is the same class of defect the sink had, on the other axis: the sink made
`KV_LEN` 2049 and cost 495 ms of ragged blocks; `Q_LEN` 2047 is ragged too and
nothing has measured it.

**If it is real the fix is nearly free and arguably more correct:**
`sequence_length: 2049` makes n = 2048, which aligns the grid *and* makes the
model train on exactly the 2048 tokens per sequence the plan says it does.
Job 32688 measures seq 2047 against 2048 to size it before changing anything.

### 2026-11-05 08:10 — S3 confirmed: one token costs **480 ms per step**

Job **32688**, `scripts/bench_attention.py` at both lengths on one H100, B=16
H=20 D=64 bf16, scaled to the 160 attention calls in one optimizer step:

| | fwd | bwd | **total/step** |
|---|---:|---:|---:|
| AME flex, ALiBi + sink merge, **n = 2047** | 482.1 | 674.8 | **1156.9 ms** |
| AME flex, ALiBi + sink merge, **n = 2048** | 171.4 | 505.6 | **677.0 ms** |
| SDPA flash floor, n = 2047 | 118.4 | 352.9 | 471.3 |
| SDPA flash floor, n = 2048 | 101.7 | 347.5 | 449.2 |

**479.9 ms**, and 2.8x of it is in the forward. The hypothesis was right and the
size of it was not a guess worth making — the ragged row-block is 1/16th of the
rows and costs more than the other fifteen together in the forward.

Note the flash floor barely moves (471.3 -> 449.2, 22 ms). This is not a generic
alignment tax on attention; it is specific to how flex handles a partial
row-block, which is the same mechanism that made the sink's `KV_LEN = 2049`
cost 495 ms. Two instances of one bug class, on the two axes of the same matrix.

**480 ms against the 450 ms that 35% needs.** It is on the boundary, so it has
to be measured in training, not projected.

#### The fix, and what it costs

`sequence_length: 2049`. The model sees `batch.shape[-1] - 1 = 2048`.

It is safe to do at config level and nowhere else: `sequence_length` never
reaches the model (grep over `src/` and `main.py` finds it only in
`definitions.py` and `datasets.py`), the MFU accounting already derives its
seq-len from `batch.shape[-1] - 1` (`trainer.py:437`), and `processed_tokens`
counts `input_ids.numel()` — so both sides of the MFU ratio stay honest and the
gain is wall-clock, not accounting.

What it is not: free of consequence. Each example now spans 2049 dataset tokens
instead of 2048, so the token stream is **not** the one B1 saw and the loss
trace will be comparable in shape but not step-for-step identical. That is a
data change, not a numerics change — the section 1.4 gates are untouched by it.

The honest framing for later: `sequence_length` in the config now means batch
width, not model sequence length. The cleaner fix is for the dataloader to
fetch `sequence_length + 1`, which makes every config in the repo correct
rather than this one. Deferred on purpose — not refactoring shared dataloader
code on the strength of an unmeasured end-to-end result.

### 2026-11-05 08:16 — the 4-GPU test bed is occupied; measuring at 1 GPU

`sbatch --test-only` for the 4-GPU job returns:

```
Preempts: 32611,32736,32737,32738,32739
```

All five are **lukaskoz**'s running jobs on h100a — `j321m_h100` preempts
`lukaskoz_h100` with `PreemptMode=CANCEL` and `GraceTime=0`, so submitting
would destroy them outright. 32611 is 11 h 14 m in. **Not submitted.**

h100a has 8 H100s and lukaskoz holds 5 of them at one GPU each, time limits
24-30 h with 1-11 h elapsed; the three free GPUs take a 1-, 2- or 3-GPU job
with no `Preempts:` line at all. Waiting out the 4th GPU means waiting until
tomorrow.

So the alignment fix gets measured where it can be measured honestly: a
**1-GPU A/B**, `h20_1gpu_mb4_mp1` with `common.sequence_length` the single
overridden variable.

| job | width | model n | role |
|---|---:|---:|---|
| **32771** | 2048 | 2047 | control, current behaviour |
| **32772** | 2049 | 2048 | aligned |

This transfers cleanly rather than by luck: at 1 GPU the step does all four
ranks' micro-steps (ga 128 at micro-batch 4 = 2560 attention calls at B=4,
four times the 160-at-B=16 a single rank of the 4-GPU run does), so the
alignment saving scales with it and the *ratio* to step time is the same
~14.5% in both. What 1 GPU cannot show is whether the saving survives contact
with FSDP2's collectives, which is the only reason the 4-GPU run still matters.

`configs/ame/h20_4gpu_mb16_mp4_align.yaml` is written and ready to submit the
moment a 4th GPU frees.

### 2026-11-05 08:30 — alignment measured in training: **28.75% -> 33.39%** at 1 GPU

Jobs **32771** (control) and **32772** (aligned), `h20_1gpu_mb4_mp1` with
`common.sequence_length` the only overridden variable.

| 1 GPU, ga 128 | MFU/GPU | step | tokens/s | peak GiB |
|---|---:|---:|---:|---:|
| control, model n = 2047 | 28.75% | 13.890 s | 75,457 | 64.71 |
| aligned, model n = 2048 | **33.39%** | **11.964 s** | **87,642** | 64.72 |

**1.925 s/step saved against 1.920 s predicted** by the kernel bench (4 x 479.9,
because one GPU at ga 128 / micro-batch 4 does 2560 calls at B=4, four times the
2560 B-units a single rank of the 4-GPU run does). A **100.3% transfer** — the
whole kernel saving reaches the step.

That is worth saying plainly because the sink merge did not: S2 transferred 52%
of its benchmark gain and I expected something similar here. The difference is
that the sink merge's benefit was partly already being captured by inductor
fusing the old form into the compiled block, so the benchmark over-stated what
was left to win; raggedness is inside the flex kernel where no amount of
surrounding fusion touches it. Memory is unchanged (64.71 vs 64.72 GiB), which
is the expected signature: the ragged block was costing time, not space.

#### What this projects to on 4 GPUs -- and why it is still a projection

Per rank the saving is the same 479.9 ms (160 calls at B=16 is the same 2560
B-units), so the step goes 3.302 -> 2.822 s:

  30.23% x (3.302 / 2.822) x 1.00049  =  **35.38%**

and the 1-GPU MFU ratio taken directly gives 30.23% x 1.1616 = **35.11%**.
Both clear 35%, by 0.1-0.4 points. That is too thin to call, and this session
has already produced one projection that was wrong by 20x (the 1-GPU Tier 2
estimate). **It gets measured before it gets claimed.**

### 2026-11-05 08:34 — the full decomposition at n = 2048 (job 32774)

The earlier bench_seq grep truncated the table at three rows. The whole thing,
one H100, B=16 H=20 D=64 bf16, scaled to the 160 attention calls one rank makes
in one optimizer step:

| n = 2048 | fwd | bwd | total |
|---|---:|---:|---:|
| AME flex (ALiBi + sink merge) | 212.9 | 489.3 | **702.2** |
| flex: ALiBi, no sink, `return_lse=True` | 122.2 | 338.8 | 461.0 |
| flex: ALiBi bf16, no sink | 120.6 | 335.4 | 456.0 |
| flex: no score_mod, causal only | 89.4 | 312.3 | 401.7 |
| SDPA flash, causal, no bias (floor) | 100.8 | 346.6 | 447.4 |
| flex: KV = n+1, sink, no branch | 451.7 | 508.2 | 959.8 |
| flex: KV = n+1, sink, with branch | 668.7 | 558.2 | 1226.9 |

Three things fall out of it.

**1. ALiBi is nearly free: 59.3 ms** (461.0 over 401.7), and flex with ALiBi
(456-461) sits at the SDPA flash floor (447.4). The score_mod was never the
problem, which is now measured rather than assumed.

**2. The sink merge costs 241.2 ms** (fwd +90.7, bwd +150.5) and that is the
next target. It should not cost this. Per call the forward moves roughly 255 MB
-- read q (83.9 MB) for the einsum, read out_seq (83.9) and write out (83.9),
plus the small alpha and lse tensors -- which is 0.085 ms at H100 bandwidth
against 0.567 ms measured. **6.7x off the bandwidth floor**, so this is a shape
or fusion problem, not work that has to be paid.

Two candidate causes, not yet separated: `torch.einsum("bhnd,hd->bhn")` lowering
to a degenerate matmul (a reduction over D=64 producing one scalar per element
-- a matrix-vector product per head, the worst case for tensor cores), and a
gradient flowing back through `lse_seq` into flex's backward. The bench's
`return_lse=True` row does NOT measure the second: it returns the LSE but never
backprops through it, so its +3.4 ms of backward is the cost of handing the
tensor back, not of differentiating it.

**3. The KV-column alternative is dead, and now by measurement.** Making the
sink a real KV entry needs a branch in score_mod to stop ALiBi applying to it,
and the branch costs **+267.1 ms** (1226.9 over 959.8) because flex evaluates
score_mod for every score in the matrix. Even padding KV to 2176 to buy back
the raggedness could not pay for that. The merge is the right shape; it is the
arithmetic inside it that is wrong.

**Next: S3b**, the 241.2 ms. On a 2.822 s step that is worth ~3.3 MFU points,
which turns a 35.1-35.4% boundary call into real margin.

### 2026-11-05 08:45 — in flight, and an idea AME's own structure hands us

Running: **32776** (merge variants bench), **32777** (1-GPU aligned run with
`profile_timers` + `kernel_profile`, to find where the other ~2.1 s of the step
goes rather than guessing at it). A background watcher polls `sbatch
--test-only` for a non-preempting 4-GPU slot.

Why a profile now: at 35% the step is 2.822 s and the ideal-FLOPs content is
0.999 s (262,144 tokens/GPU/step x 3.768 GFLOPs at 989 TFLOP/s). Attention is
702 ms of that wall time and the bank path was measured at ~8%. So roughly half
the step is neither attention nor banks and nothing has looked at it.

#### S3b candidates being measured (32776)

Four merge forms, with `out_seq`/`lse_seq` as leaf tensors so flex is out of
the picture: einsum vs a mul+sum reduction over D, each eager and compiled.
The compiled pair matters beyond S3b — **every bench row so far has been eager
while the real model runs this inside a compiled block**, which is the most
likely explanation for why the sink merge transferred 52% and the alignment fix
transferred 100%. Plus one diagnostic row with `lse` detached, which is not
valid math (it drops a real gradient term) but is the only way to separate the
merge's own arithmetic from what flex pays to differentiate its LSE.

#### S3c — sink_score is a property of the token id, not the position

If the measured fixes do not close the 241 ms, there is a structural one that
only exists because this is AME.

`q` is not computed, it is **gathered**: `q = q_bank(token_ids)`. So

    sink_score[b, h, n] = q_bank[ids[b, n]] . k_sink[h]

depends on `ids[b, n]` and nothing else. The whole `[B, H, n, D]` einsum is
recomputing, 16 x 2048 times per block, a quantity that has only `vocab`
distinct values per head.

Precompute `sink_proj = q_bank_weight.view(vocab, H, D) @ k_sink` -- a
`[50304, 20]` table, 4 MB fp32 -- and `sink_score` becomes a gather of 655K
elements (1.3 MB) instead of a reduction over 83.9 MB of q.

The arithmetic on when it pays: building `sink_proj` for one block reads that
block's whole q_bank, 128.8 MB, which is *more* than the 83.9 MB of q it
replaces -- for a single micro-step. But the bank is constant across an
optimizer step, so it is built **once per step and reused across all 8
gradient-accumulation micro-steps**: 128.8 MB once against 83.9 x 8 = 671 MB.
A 5.2x reduction in traffic for this term, and it grows with ga.

The catch, stated now so it is not discovered later: the backward has to route
`d(sink_score)` back into the bank as an index_add over token occurrences
rather than into a dense `dq`. That is the same scatter the bank already does
every step, so it folds into existing machinery -- but it is real work and it
is the reason this is S3c and not S3b. It also only holds while the merge
reads q straight from the bank; any transform between gather and attention
(there is none today) would break it.

### 2026-11-05 09:00 — S3b measured, and it inverts under compile (job 32776)

The merge in isolation, `out_seq`/`lse_seq` as leaf tensors so flex is out of
the picture, per optimizer step at the h20 shape:

| merge form | fwd | bwd | total |
|---|---:|---:|---:|
| einsum (current), eager | 88.3 | 131.5 | 219.9 |
| mul+sum over D, eager | 121.6 | 159.7 | 281.3 |
| einsum, **compiled** | 41.8 | 61.8 | 103.6 |
| **mul+sum, compiled** | **14.2** | **36.7** | **50.9** |
| flex+merge, lse DETACHED (wrong math) | 212.9 | 494.2 | 707.1 |

**Eager, the einsum wins. Compiled, it loses by 2x.** A reduction over the last
axis of `[B, H, n, D]` genuinely costs more than a library GEMM when nothing
fuses it; under inductor the multiply, the reduction, the sigmoid and the final
FMA all fold into neighbouring pointwise kernels, and a cuBLAS call cannot fold
into anything. Optimising this eagerly would have picked the wrong form --
worth remembering, because every bench row before this one was eager.

That also **retires the 241 ms figure** from the decomposition above. The real
model compiles this path, so it was already paying ~103.6 ms, not 241 -- which
is the actual explanation for the sink merge transferring only 52% into
training while the alignment fix transferred 100%. S3b is worth **~53 ms**
(0.65 MFU points at a 2.822 s step), not 241. Still worth taking; not the
answer on its own.

**The detached-lse row kills its own hypothesis**, which is the useful outcome:
707.1 against 703.1 for the real thing is noise, so backprop through flex's LSE
is **free**. No numerics compromise is needed anywhere here, and that line of
attack is closed rather than left open.

Implemented in `src/core/ame.py` with the measurements in the comment.
Running as job **32796** (1-GPU, aligned), gate as **32797** (the CUDA AME and
memory_parallel suites, which is where the flex-vs-reference equivalence tests
actually execute -- locally they skip, 9 CUDA tests fail identically with and
without the change).

### 2026-11-05 09:05 — where the step actually goes (job 32794, phase attribution)

`scripts/kprof_phase.py` over the 1.33 GB trace, ms per step, 1 GPU ga 128:

| category | forward | backward | total |
|---|---:|---:|---:|
| GEMM | 1482.2 | 3430.9 | **4913.2** |
| triton glue | 743.6 | 1447.7 | 2191.4 |
| ATTENTION (flex) | 548.5 | 1375.3 | 1923.8 |
| optimizer | 558.3 | 788.9 | **1347.4** |
| other eager | 90.8 | 1108.5 | 1199.3 |
| FSDP2 packing | 130.5 | 375.3 | 505.7 |
| copy / memcpy | 37.1 | 391.4 | 428.5 |
| softmax / CE | 124.0 | 137.3 | 261.3 |
| bank / embedding | 81.0 | 5.7 | 86.7 |
| **TOTAL** | 3796.0 | 9061.0 | **12857.2** |

Raw kernels sum to 12148.7 against a 12515 ms step, so this accounts for ~97%
of the time and is not a sampling artefact.

**The anomaly: 1347 ms of "optimizer" launched inside forward and backward.**
No optimizer runs there -- the `optim` timer is 9.9 ms and `bank_step` is 80.
The category is matching `multi_tensor_apply_kernel` (12,288 launches/step),
which is FSDP2's `foreach` copy machinery for all-gather and reduce-scatter
staging, not AdamW. With FSDP2 packing (505.7) and copy/memcpy (428.5) that is
**2281 ms, 17.7% of the step, at a world size where FSDP2 shards nothing.**

**But do not bank that as a 4-GPU win, and this is the important caveat.**
Every one of those is a *per-micro-step, per-parameter* cost. One GPU runs
ga 128; each rank of the 4-GPU run does ga 8. They shrink **16x** there -- 1347
-> ~84 ms, 506 -> ~32, 428 -> ~27. The 1-GPU profile systematically overstates
exactly the costs Tier 2 was built to remove, which is the same trap the Tier 1
attribution gate fell into from the other side. The categories that scale with
*tokens* rather than micro-steps -- GEMM, triton glue, attention, CE -- divide
by 4 instead, and those are the ones that dominate at 4 GPUs:

    GEMM 1228 | triton glue 548 | attention ~500 | CE 65 | bank 22

against a 2.822 s step. **GEMM is the single largest item and runs at ~76% of
peak** (923 TFLOP of backbone matmul per GPU per step needs 933 ms at 989
TFLOP/s; 1228 ms measured). Closing that to 90% would be ~190 ms, 2.3 points.

The honest limit of all of this: **it is extrapolation from one GPU.** A 4-GPU
profile would settle it and cannot be run while the node is full. When a slot
frees, two runs go in: the aligned+S3b headline, and a profiled twin.

### Backlog, re-sized by the phase data

| | item | est. at 4 GPUs | status |
|---|---|---:|---|
| S3b | mul+sum merge under compile | ~53 ms | **done, measuring** |
| S4 | `reshard_after_forward=False` | ~23 ms | deprioritised -- smaller than it looked |
| S5 | bank path under compile | ? | graph break per block, 160/step |
| S6 | hoist the id all-gather | ~16 ms | small |
| **S7** | **GEMM efficiency, 76% -> 90%** | **~190 ms** | **new, biggest single item** |

Note on scope: O10 (activation checkpointing) and O11 (fused-linear-CE) would
both bite directly on "triton glue" and "softmax / CE", and O11 in particular
is the standard fix for exactly the CE shape this model has. Both are on the
goal's explicit CUT list, so they stay unimplemented; flagging them because
they are the obvious next lever if the permitted ones fall short, and that is
a decision for Maciej rather than one to take silently.

### 2026-11-05 09:20 — correction: the GEMM headroom is 61-130 ms, not 190

I sized S7 against 90% of peak. That is not a number cuBLAS reaches. 752
TFLOP/s on large bf16 GEMMs is 76% of an H100's 989 dense, and the practical
ceiling for shapes like these is 80-85%:

| target | 1-GPU GEMM | saved at 4 GPUs |
|---|---:|---:|
| 80% of peak | 4668 ms | 61 ms |
| 85% of peak | 4393 ms | 130 ms |
| 90% of peak | 4149 ms | 191 ms |

So S7 is worth 0.7-1.6 MFU points, not 2.3, and it is incremental tuning rather
than a defect to fix. It stays the largest single line in the budget and is no
longer obviously the next thing to do.

What that leaves, within the goal's scope, against a projected ~2.769 s step:

| item | est. | note |
|---|---:|---|
| S7 GEMM 76% -> 80-85% | 61-130 ms | incremental, uncertain |
| S4 `reshard_after_forward=False` | ~23 ms | one line |
| S6 hoist the id all-gather | ~16 ms | 320 -> 8 launches |
| S5 bank path under compile | ? | 160 graph breaks/step, unsized |

~100-170 ms, or 1.2-2.1 points, on top of a projected ~35.8%. The target is
reachable on the permitted list **if the alignment projection holds at 4 GPUs**
— which remains the one thing that has not been measured and cannot be while
h100a is full.

### 2026-11-05 09:35 — the 4-GPU headline: **34.42%**, and my projection was wrong

Job **32820**, h20 at 4 GPUs, alignment + S3b, against 32687's sink-merge
baseline:

| 4 GPU, h20 | MFU/GPU | step | tokens/s |
|---|---:|---:|---:|
| sink merge (32687) | 30.13% | 3.314 s | 316,284 |
| **+ alignment + S3b (32820)** | **34.42%** | **2.902 s** | **361,303** |

**I projected 36.1-36.4%. It came in at 34.42%, below the 35% target.** The
error is worth keeping rather than quietly updating: both projection routes
assumed the per-rank attention saving would land in the step one-for-one, as it
had at 1 GPU.

It did not. Per rank the saving is 480 (alignment) + 80 (S3b) = **560 ms**; the
step dropped **412 ms**. The missing 148 ms is **communication that attention
used to hide**. This is the Tier 1 finding running backwards: Tier 1 measured
that NCCL was already overlapped behind flex attention, which is why cutting
NCCL traffic by 55% bought nothing. Take the attention away and the same
overlap stops being free — the collectives surface.

So the two results are one result. At 4 GPUs h20 is now **communication-bound**,
and the 1-GPU transfer ratio is not a valid predictor for it. Any further
compute saving at 4 GPUs will keep arriving at a discount until the
communication underneath is dealt with.

| | 1 GPU | 4 GPU |
|---|---:|---:|
| n=2047, no S3b | 28.75% | 30.13% |
| aligned + S3b | 34.31% | 34.42% |
| gain | **+19.3%** | **+14.2%** |

### 2026-11-05 09:36 — h16 reaches **37.87%** at 1 GPU

Job **32825**. h16 at micro-batch 16 against h20 at micro-batch 4, both
aligned, both with S3b, both 1 GPU:

| 1 GPU | micro-batch | MFU | step | tokens/s |
|---|---:|---:|---:|---:|
| h20 | 4 | 34.31% | 11.644 s | 90,050 |
| **h16** | **16** | **37.87%** | **5.938 s** | 176,573 |

The mechanism is the one the config predicted: bank state at `mem_parallel_size
1` is 46.4 GB at h20 against 29.7 at h16, and the freed memory buys a 4x
micro-batch. Micro-batch is what amortises the per-micro-step costs the phase
table found -- FSDP2 staging, gradient accumulation, optimizer foreach copies.

Not a like-for-like model comparison, deliberately: the micro-batches differ
because the question is what MFU each H can reach, not what each does at h20's
shape. h12 at micro-batch 32 is job **32832**.

### In flight

| job | what | why |
|---|---|---|
| 32832 | h12, micro-batch 32, 1 GPU | most bank memory freed, so the largest micro-batch |
| 32834 | h20 1 GPU, **FSDP2 off**, bf16 from autocast | the 2281 ms of sharding machinery at world size 1 |
| 32836 | correctness gate, full capture | 32797 and 32822 both lost the summary line; 32822 ran 10 min and printed none, so it may have died rather than passed |

The gate is the one that matters for honesty: **34.31% and 34.42% are
throughput numbers whose correctness gate has not yet produced a readable
result.** Two attempts were lost to my own output filtering. Until 32836 comes
back, the S3b merge rewrite is unverified on GPU, and the section 1.4 ordering
-- gate before throughput -- is not satisfied for it.

### 2026-11-05 09:50 — the gate passes, and the h-sweep is not monotonic

**Correctness gate (job 32836), finally with full capture:**

```
AME suite exit=0            73 passed
memory_parallel exit=1      1 failed: test_fusing_the_qk_pair_changes_nothing[4]
```

The AME suite passing is the one that matters for S3b: it contains the
flex-vs-reference equivalence tests, and they clear. **The merge rewrite is
gated.**

The memory_parallel failure is **not a correctness failure** and the
distinction matters. The traceback is `DistNetworkError ... EADDRINUSE` out of
`init_process_group` — the test never reached an assertion. `conftest.free_port()`
asks the OS for a port, closes the socket, and hands the number to a child that
binds it a moment later; on a shared node with several jobs each spawning four
ranks, something else takes it in between. The same test passed in 32797, in
32822 and locally.

Fixed rather than re-run and hoped away: `conftest.spawn_on_free_port()` retries
the spawn with a new port on EADDRINUSE and on nothing else, so infrastructure
noise stops looking like a test result. All five spawn sites in
`tests/test_memory_parallel.py` use it; 28/28 pass locally.

**The h-sweep, and "smaller is better" is wrong:**

| 1 GPU | micro-batch | MFU | step | peak GiB | GFLOPs/token |
|---|---:|---:|---:|---:|---:|
| h12 | 32 | 34.46% | 3.240 s | 66.54 | 1.053 |
| **h16** | **16** | **37.87%** | 5.938 s | 68.35 | 2.121 |
| h20 | 4 | 34.31% | 11.644 s | 64.71 | 3.768 |
| h20, FSDP2 off | 4 | 35.60% | 11.223 s | 66.87 | 3.768 |

**h12 is worse than h16**, so the micro-batch story has a limit. The mechanism
is arithmetic intensity: FLOPs/token falls 3.768 -> 2.121 -> 1.053 as H
shrinks, so each token carries less compute to hide memory traffic behind. h16
is where the micro-batch gain still outweighs that loss; by h12 it does not.
There is a sweet spot, and it is not at the edge.

**No-FSDP2 returned 421 ms, not the 2281 ms I sized it at** — 18%. The phase
table's FSDP2 rows were mostly not removable this way, and autocast brings its
own per-micro-step cost: it keeps fp32 parameters and casts them per region,
which is why peak memory went *up* (64.71 -> 66.87 GiB) rather than down. A
real but modest 1-GPU-only win.

**h16 has no room for a larger micro-batch** (68.35 of 80 GiB), so the next
move is to combine the two: h16 with FSDP2 off, job **32841**, projected
37.87% x (35.60/34.31) = **~39.3%**. Short of 40% if the two gains are
independent, which they may not be.

Also submitted: job **32843**, the **4-GPU kernel profile** (T4.1/O32). Every
phase table in this work is 1 GPU, where the collectives are single-rank
no-ops. The 148 ms of exposed communication at 4 GPUs is inferred from a
subtraction and has never been looked at directly.

### 2026-11-05 10:05 — the 4-GPU profile (job 32843). First one ever taken.

Step 2893 ms, 34.52% (34.42% unprofiled -- the instrumentation is ~free).

| category | fwd | bwd | total |
|---|---:|---:|---:|
| GEMM | 371.0 | 815.6 | **1186.9** |
| triton glue | 209.6 | 522.2 | **731.8** |
| ATTENTION (flex) | 122.6 | 350.4 | 472.9 |
| NCCL all-gather | 113.2 | 106.0 | 219.2 |
| NCCL other (all-to-all) | 82.0 | 133.7 | 215.6 |
| other eager | 11.3 | 90.7 | 102.0 |
| NCCL reduce-scatter | 5.2 | 69.8 | 75.0 |
| copy / memcpy | 5.9 | 65.4 | 71.4 |
| optimizer | 31.7 | 26.3 | 58.0 |
| softmax / CE | 23.7 | 33.8 | 57.5 |
| FSDP2 packing | 9.0 | 22.4 | 31.4 |
| bank / embedding | 2.8 | 1.3 | 4.1 |
| **TOTAL kernels** | 987.9 | 2237.5 | **3225.7** |

**Kernel total exceeds the step** (3225.7 vs 2893) and that is the point: the
difference is overlap, measured rather than assumed. NCCL totals **509.8 ms**
but non-NCCL kernels alone are 2715.9, so only **~177 ms of communication is
exposed** -- which independently confirms the 148 ms inferred earlier by
subtraction from the alignment result.

My 4-GPU extrapolations from the 1-GPU profile, scored: GEMM 1228 predicted vs
1186.9 measured, attention ~500 vs 472.9 -- both good. **triton glue 548 vs
731.8 -- underestimated by a third.**

#### A phantom I nearly chased

The timers show `log=151.5 ms`, 5.2% of the step, against ~20 ms at 1 GPU. It
is **not** a cost. `log` is a *host* timer; forward/backward are *GPU* timers,
and 858.4 + 2000.4 + 4.6 + 5.2 + 1.8 + 20.7 = 2891.1 against a 2893.4 step --
the GPU sections already account for the whole thing. The 151.5 ms is the host
blocking on `.item()` for work already counted in backward. Noted because the
number is conspicuous and the next person will see it too.

#### Top kernels, and what is actually addressable

| ms/step | launches | kernel |
|---:|---:|---|
| 350.4 | 160 | `triton_tem_fused_mul_1` -- flex backward |
| 271.6 | 648 | nvjet GEMM |
| 219.2 | 512 | `ncclDevKernel_AllGather` -- FSDP2 parameters |
| **215.6** | **320** | `ncclDevKernel_SendRecv` -- **Tier 3's bank all-to-all** |
| **134.6** | **160** | `triton_poi_fused_stack_10` -- **the fused-pair permute** |
| 122.6 | 160 | `triton_tem_fused_0` -- flex forward |
| 87.1 | 160 | SiLU backward |
| 75.0 | 176 | `ncclDevKernel_ReduceScatter` |

**Tier 3 costs 350.2 ms at 4 GPUs** -- 215.6 of all-to-all plus 134.6 of the
permute that reassembles the column shards. It still pays: it replaced FSDP2's
bank all-gather, measured at 269.3 + 276.6 = 546 ms in the plan. But the margin
is thinner than the design assumed, and `bank_reduce` is down to 1.8 ms, so the
gradient traffic Tier 2 targeted is simply gone.

`triton_poi_fused_stack_10` is `_FusedBankGather`'s `.permute().reshape()`
after the all-to-all: 0.84 ms per launch moving ~336 MB, **~400 GB/s on a 3
TB/s machine**, because it transposes. It is not overlapped with anything. Best
single addressable item found.

#### O33 submitted (job 32847)

`reshard_after_forward=False`, plumbed as an optional key in
`setup_fsdp2_model` so no existing run moves. The all-gather is 219.2 ms with
106.0 in backward, and while attention was slow that hid behind it and the flag
was worth nothing. It is not hidden now. It costs memory and h20/mb16 is at
63.5 GiB of 80 -- **if it OOMs that is the answer**, not a failure.

#### Where 40% has to come from, at 4 GPUs

Step 2893 -> 2497 ms is **-396 ms**. Candidates, with honesty about certainty:

| item | est. | certainty |
|---|---:|---|
| `fused_stack` permute, better layout | ~85 ms | measured cost, unproven fix |
| O33 backward all-gather | 0-106 ms | in flight; partly overlapped |
| GEMM 76% -> 80-85% | 61-130 ms | incremental |
| triton glue fusion | ? | 731.8 ms of surface, no plan yet |

That reaches 38-40% at best on the permitted list. **40% at 4 GPUs is not
comfortably in hand**, and 44.44% is not in sight by this route.

### 2026-11-05 10:15 — O33 lands: **4-GPU h20 is over 35%**

| 4 GPU, h20 | MFU/GPU | step | tokens/s | peak GiB |
|---|---:|---:|---:|---:|
| sink merge (32687) | 30.13% | 3.314 s | 316,284 | 63.50 |
| + alignment + S3b (32820) | 34.42% | 2.902 s | 361,303 | 63.50 |
| **+ O33 reshard_after_forward=False (32847)** | **35.02%** | **2.852 s** | **367,623** | 64.63 |

50 ms, which is **47% of the 106 ms backward all-gather** the profile measured
— the rest stays hidden behind compute, exactly as the overlap analysis said it
would. It costs 1.1 GiB, far less than feared, so the memory objection to it
was wrong.

**Cumulative against B1 (26.82%): +30.6%. Against the dense ceiling: 78.8%.**

### 2026-11-05 10:16 — h16 at 4 GPUs OOMs, and the allocation names its own fix

Job **32853**, h16 micro-batch 32 / ga 4: `CUDA out of memory. Tried to
allocate 6.14 GiB ... 74.50 GiB in use`.

**6.14 GiB is exactly `[32, 2048, 50304]` in bf16** — the logits. Not the
banks, not the activations, not anything Tier 3 governs. My estimate of 75.7
GiB was close and on the wrong side of the line.

That allocation is precisely what **O11 (fused-linear-CE)** removes, and it is
the second time today the cut list has been the binding constraint rather than
anything in scope — the first being O10 and the micro-batch at h20. Recorded,
not acted on.

Resubmitted as **32856** at micro-batch 16 / ga 8, which also makes it directly
comparable to h20's 4-GPU shape, with O33 folded in.

### Where the 40% target stands

| configuration | MFU | gap to 40% |
|---|---:|---:|
| 4-GPU h20, best (32847) | **35.02%** | 355 ms |
| 1-GPU h16 (32825) | **37.87%** | ~330 ms |

Both are within ~350 ms. Remaining permitted candidates at 4 GPUs:

| item | est. | note |
|---|---:|---|
| `fused_stack` permute | ~85 ms | 400 GB/s on a 3 TB/s machine; kernel quality, not structure |
| GEMM 76% -> 80-85% | 61-130 ms | incremental |
| triton glue (731.8 ms surface) | ? | no plan yet |

**The permute fix I reasoned through and rejected:** fusing
`_FusedBankGather`'s `.permute().reshape()` into `_mixed_qkv`'s final
`transpose(1, 2)` would not remove a copy. The intervening elementwise work
(qk_norm, alpha mix, tau) materialises `[B, n, H, D]` contiguously and the
transpose is a *view*, so there is exactly one materialising copy today.
Fusing moves it rather than deleting it. The defect is that the one copy runs
at ~400 GB/s, which is kernel quality.

**The structural fix is a redesign.** Column shard `r` owns heads
`[5r, 5r+5)`, and heads are independent in attention — so the all-to-all exists
only to reassemble heads that never needed reassembling. Computing attention
per head group is tensor parallelism, i.e. a redesign of O6, not a tweak. Noted
for whoever picks this up; not started.

### 2026-11-05 10:20 — h16 at 4 GPUs: **33.41%**. The h-sweep crosses over.

Job **32856**, h16 micro-batch 16 / ga 8 with O33: 33.41%, 1.683 s/step,
623,111 tokens/s, **44.53 GiB** (so memory was never the constraint here --
mb 32 failed only on the logits).

| MFU/GPU | 1 GPU | 4 GPU |
|---|---:|---:|
| h20 | 34.31% | **35.02%** |
| h16 | **37.87%** | 33.41% |

**The ranking inverts with GPU count**, and the reason is the same arithmetic
intensity that explained h12. At 1 GPU h20 is memory-starved into micro-batch 4
and h16's 4x micro-batch wins. At 4 GPUs the banks shard, h20 gets micro-batch
16 too, and the comparison reverts to what each model is: h20 carries 3.768
GFLOPs/token against h16's 2.121, so h16 has less compute per token to hide
communication behind and loses more of it.

The general statement, which is the useful one: **AME's MFU degrades with GPU
count.** The bank machinery -- all-gather of ids, all-to-all of activations,
the reassembly permute -- is work that does not exist at world size 1, and it
grows while the compute per rank shrinks. 1 GPU h16 at 37.87% is the best
number this work has produced, and it is the configuration with no bank
communication at all.

That also reframes the target. The h-sweep is explored: h12 34.46%, h16 37.87%,
h20 34.31% at 1 GPU, with a clear peak. There is no further h to try that would
plausibly beat h16 -- h14 and h18 sit inside a span of 3.5 points and would be
interpolation, not discovery.

**So the remaining distance to 40% has to come from the step itself, and the
best configuration has never been profiled.** Job **32859** profiles h16 at
1 GPU. Everything known about where AME's time goes comes from h20 -- at 1 GPU
with ga 128 (which overstates per-micro-step costs 16x relative to a 4-GPU
rank) and at 4 GPUs with ga 8. h16 at ga 32 is a third regime and the one that
currently matters.

### 2026-11-05 10:30 — h16 profiled (job 32859). The permitted list is spent.

Step 6015.9 ms, 1 GPU, ga 32 — the regime that actually matters now, and one
nothing had measured.

| category | fwd | bwd | total | share |
|---|---:|---:|---:|---:|
| GEMM | 809.4 | 1627.0 | **2436.4** | 40.5% |
| triton glue | 437.6 | 845.6 | **1283.3** | 21.3% |
| ATTENTION (flex) | 314.7 | 923.8 | **1238.5** | 20.6% |
| other eager | 35.0 | 347.6 | 382.6 | 6.4% |
| optimizer | 118.4 | 205.1 | 323.5 | 5.4% |
| softmax / CE | 94.9 | 135.3 | 230.2 | 3.8% |
| copy / memcpy | 22.2 | 176.4 | 198.5 | 3.3% |
| FSDP2 packing | 22.1 | 55.1 | 77.2 | 1.3% |
| bank / embedding | 57.2 | 4.4 | 61.6 | 1.0% |

**Three of the four largest rows are already at a hardware limit**, and that is
the finding — not a number but a ceiling.

**Attention, 1238.5 ms.** ~20% of peak FLOPs, which looks terrible until you
note the bench measured SDPA flash at the same cost (447.4 vs 456-461 for flex
with ALiBi, job 32774). At n=2048 and D=64 that is what attention costs; flex
and the score_mod are not the problem and never were.

**GEMM, 2436.4 ms at 81.4% of peak.** 311.6M backbone x 6 x 1,048,576 tokens =
1960 TFLOP, which needs 1982 ms at 989 TFLOP/s. Better than h20's 76%, so the
GEMM lever I sized at 61-130 ms is **largely already spent at h16**. A
realistic 88% would return ~180 ms.

**SwiGLU, 362 ms (228.1 backward + 134.1 forward), is exactly bandwidth-bound.**
The backward touches ~5 x 268 MB per call over 512 calls = 687 GB/step, which
is 229 ms at 3 TB/s against 228.1 measured. It cannot be made faster. It can
only be made *smaller*, by recomputing instead of storing — which is O10.

**bank / embedding is 61.6 ms, 1.0%.** The Tier 1 attribution gate fired
NEGATIVE on the bank path and it was right; after Tiers 2+3 the path is
invisible.

#### The arithmetic of what is left

40% at h16 needs the step at 5.622 s: **-316 ms** (from the unprofiled 5.938).

| candidate | est. | status |
|---|---:|---|
| GEMM 81.4% -> 88% | ~180 ms | incremental, uncertain |
| everything else above | ~0 | at hardware limits |
| **O11 fused-linear-CE** | **~230 ms + 6.1 GiB** | **CUT** |
| **O10 activation checkpointing** | enables micro-batch 32, halving ~980 ms of per-micro-step cost | **CUT** |

**40% is not reachable for AME on the permitted list.** The honest statement is
that the two items blocking it are precisely the two on the cut list, and this
is now the third independent time today they have been the binding constraint:

1. h20 at 4 GPUs cannot raise micro-batch past 16 — activations (O10).
2. h16 at 4 GPUs OOMed at micro-batch 32 on a 6.14 GiB logits tensor (O11).
3. h16 at 1 GPU is within ~180 ms of its permitted floor, and the next 230 ms
   is the CE tensor (O11) with the one after that being micro-batch (O10).

That is a decision for Maciej, and it is not one to take silently. The §0 scope
note says "do not reintroduce them to hit a number" — which is exactly what
lifting them now would be, so it needs to be an explicit change of scope rather
than a quiet one.

### 2026-11-05 11:15 — O11 measured, and two implementations of it

Stopped here at Maciej's request. State for whoever resumes.

**O11 delivers the memory and, as first written, lost MFU.**

| h16 | MFU | step | peak GiB |
|---|---:|---:|---:|
| 1 GPU mb16, no O11 | **37.87%** | 5.938 s | 68.35 |
| 1 GPU mb16, O11 via `checkpoint` | 35.22% | 6.385 s | 60.65 |
| 1 GPU mb16, O11 gradients-in-forward **v2** | **20.85%** | 10.787 s | 62.08 |
| 1 GPU mb16, O11 **v3** (contractions in model dtype) | job 32876 | | |
| 4 GPU mb32, O11 via `checkpoint` | 32.47% | 1.732 s | 62.61 (was OOM) |
| 4 GPU mb16, no O11 | 33.41% | 1.683 s | 44.53 |

The memory claim held exactly: micro-batch 32 fits at 4 GPUs where it OOMed on
the 6.14 GiB logits, and 7.7 GiB came back at 1 GPU. The MFU claim did not.

**v1 (`torch.utils.checkpoint`)** is correct by construction but reruns each
chunk's projection *and* softmax in backward: ~144 ms of matmul + ~230 ms of CE
against +447 ms measured. Cost 2.65 points; the micro-batch it unlocks is worth
about +4.5% relative, so it is net negative.

**v2 (gradients computed in the forward, no recompute)** was much worse, and
the reason is a mistake worth not repeating: the fp32 accumulation dtype got
applied to the *contractions* as well as the log-sum-exp. `probabilities @
weight.to(acc)` on a bf16 model is an fp32 matmul -- ~60 TFLOP/s against ~989 --
twice per chunk, on the largest matmul in the step. **fp32 belongs on the
log-sum-exp (50304 terms) and nowhere else**; cuBLAS already accumulates a bf16
product in fp32 internally.

**v3** fixes that: contractions in `x.dtype`, accumulation buffers in fp32,
chunk 4096 -> 1024 because the fp32 log-softmax buffer is `[chunk, 50304]`
(824 MB at 4096, 206 at 1024). Gate 14/14. Job **32876** was running when work
stopped; its result is the one that decides whether O11 is net positive.

**§1.4 gate for O11, passed before any throughput was quoted:** 14 CPU tests
(float64, a few ulps, each with a negative control that misses by 1e-2+); loss
trajectory identical from step 2 on, 2 bf16 ulps at steps 0-1; grad-norm
relative delta 3.4e-6 at step 0 growing to 6.0e-3 by step 29, sign-flipping
rather than biased. **Caveat worth carrying:** 30 steps cannot distinguish
reassociation from a small one-directional shift the way the 300-step
`compile_ce` comparison in the plan did. If O11 is used for real training
rather than throughput measurement, that longer comparison should be run.

Two defects the gate caught in my own code, both of which would have passed a
weaker test: `.float()` silently **downcasts** float64, so the float64 gate
would have been measuring float32 reassociation; and the
"logits-never-materialised" probe hooked the module while the new code calls
`F.linear` directly, so it was passing vacuously.

**Cluster state:** j321m's home hit its disk quota mid-session. I deleted my own
profiler traces (1.9 GB, pure scratch -- every phase table is recorded here).
The cemetery is still **93 GB**, of which **~35 GB is two `brrrrr_*` runs'
checkpoints** from 2026-11-04 (8 x 4.4 GB `.distcp`). Not mine, not touched,
and the thing that will cause this again. The filesystem has 7.8 TB free, so it
is purely the per-user limit.
