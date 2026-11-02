# MFU ablation plan: six candidate fixes, one branch each, measured against baseline

2026-10-31. Follow-up to [mfu_gap_nanogpt_audit.md](mfu_gap_nanogpt_audit.md),
which ranked six gap sources behind our 27.7% h16 MFU vs tuned-stack ~40%+.
Maciej's directive: **implement each improvement in a separate tree off the
current code version, run a training with each, and compare to the baseline** —
so the result is a per-fix MFU delta that names the actual blocker, not a stack
where the first change takes credit for the rest.

**Status: EXECUTED 2026-10-31, extended 2026-11-01.** Verdict: **torch.compile
was the blocker** (+18% at every size once a guard bug was fixed); the data
path, FSDP2 overhead, optimizer and RoPE casts were all measured innocent.
Final stack (= V1 compile only): h12 29.8%, h16 32.9%, h20 36.0%; the
nanoGPT-parity combo adds ~0.5pt. The size-matched nanoGPT runs then showed
the honest remaining gap is **10.6–12.1pt at equal workload** — see the
comparison record below.

## nanoGPT vs our framework: the complete comparison record (2026-11-01)

Everything tested and identified so far, in one place. Hardware held fixed
throughout: one entropy H100 SXM (h100a), bf16, MFU on the 989 TFLOP/s basis
(nanoGPT's printed mfu is on its hard-coded A100 312e12 basis; ×312/989 to
convert). Both frameworks count FLOPs/token the same way (6·P incl. LM head +
12·s·d_att·L), so MFUs are comparable; tokens/s is the arbitration-free number.

### 1. What was measured (the full matrix)

| configuration | h16 size (~303M w/ head) | h20 size (~556M w/ head) |
|---|---:|---:|
| ours, eager baseline (B0: FSDP2 bf16) | 28.1% | 30.8% |
| ours, + compile only (`mfu-abl/stack`) | 32.9% | 36.0% |
| ours, full nanoGPT recipe (`mfu-abl/nanogpt-parity` = V1+V3+V5+V6) | **33.6%** | **36.4%** |
| ours, + whole-model compile (`mfu-abl/whole-compile`, 2026-11-01) | — | **44.4%** |
| ours, + compiled loss (`mfu-abl/compiled-loss`, 2026-11-02) | — | **46.4%** (2.2% faster wall-clock than nanoGPT) |
| nanoGPT, size-matched (same non-emb params, seq 2048, 1.05M tok/step) | **45.7%** | **47.0%** |
| nanoGPT, its own defaults (124M, seq 1024) | 37.9% | — |

Tokens/s at matched shape: h16-size 149,474 (ours) vs 194,600 (theirs) =
**1.30×**; h20-size 90,697 vs 113,150 = **1.25×**. Jobs 31677/31680 (ours),
31871/31872 (theirs); per-run detail and wandb links in the results tables
below.

### 2. Code differences tested, one branch each — verdicts

| hypothesis (nanoGPT does X, we didn't) | variant | verdict on our stack |
|---|---|---|
| whole-training `torch.compile` | V1 `mfu-abl/compile` | **the blocker: +17–19% at every size** (after fixing the per-block `log_name` guard → recompile-limit bug) |
| fused AdamW + single clip kernel | V3 `mfu-abl/fused-optim` | noise (±0.1%) |
| no sharding wrapper, plain autocast bf16 | V5 `mfu-abl/plain-bf16` | −5/−7% eager; **sign flips to +0.4–0.7pt under compile**, and compile frees enough memory that h20 fits micro 16 where eager OOMed |
| memmap-style async H2D + prefetch | V6 `mfu-abl/data-path` (+V6c random-token run) | noise; the whole input pipeline costs ≈0 at these sizes |
| (not nanoGPT features, our own hygiene) | V2 `enable_gqa`, V4 RoPE casts | V2 +2–3% eager but −1.6% under compile (reverted); V4 noise |

All four genuine nanoGPT code features live together on
`mfu-abl/nanogpt-parity` (gates green, pushed). **After it, no known
trainer-level code difference remains untested except compile scope.**

### 3. What explains the remaining 10.6–12.1pt, in testable order

1. **Compile scope** — nanoGPT compiles the whole model as one inductor graph;
   we compile attention+FF per block, leaving Residual/norm/embedding/LM-head
   glue eager (forced by the `log_name` guard bug). **Confirmed by the
   kernel-level profile (§4b) as the whole gap**; needs the Residual refactor.
2. ~~**Vocab padding**~~ — measured minor (§4b): the head GEMM sits in a GEMM
   total already *faster* than theirs; only the eager CE (230 ms) remains, as
   part of the glue item.
3. ~~**Architecture (paper choices)**~~ — measured off the list (§4b): our
   GEMMs + flash attention run 713 ms/step *faster* than nanoGPT's; RoPE and
   the gated FF live inside compiled regions whose triton total is smaller
   than theirs.
4. ~~**Trainer-loop residue**~~ — measured at parity by the section timers
   (§4): ≈12 ms of an 11,536 ms step. Closed.

### 4. Where the time goes: matched per-section timers, h20 (2026-11-01)

The same `LoopTimers` class (CUDA-event pairs for GPU sections, perf_counter
for the host batch wait, one synchronize per step) was placed at identical
points in both training loops: our `src/core/trainer.py` behind
`+trainer.profile_timers=true` (branch `mfu-abl/profile-timers`, off
`nanogpt-parity`; numerics no-op verified) and nanoGPT's `train.py`
(`--profile_timers=True`, patched entropy clone). Both h20-size, seq 2048,
16×32, same H100; means over the measurement window, ms per step (jobs
31877 ours / 31878 theirs):

| section | ours (parity h20) | nanoGPT (h20-eq) | ratio ours/theirs |
|---|---:|---:|---:|
| forward (32 micros) | 3,641 | 2,926 | **1.24×** |
| backward (32 micros) | 7,880 | 6,330 | **1.24×** |
| grad clip | 2.8 | 2.6 | 1.1× |
| optimizer step | 8.3 | 6.6 | 1.3× (both ≈0) |
| data: host wait | 0.2 | 56.9¹ | — (both overlapped) |
| data: H2D stream time | 0.5 | 0.7 | — |
| **step total (wall)** | **11,536** | **~9,250** | **1.25×** |

¹ their get_batch does memmap read + stack + pin on the host inside the micro
loop (~1.8 ms per micro); it overlaps GPU work and does not reach the step
total. Timers added no measurable overhead on either side (their step time
with timers, 9,250 ms, matches the untimed run's 9,267 ms).

**Reading**: the entire 25% wall-clock gap lives inside the model's forward
and backward kernels, in *exactly equal proportion* (1.244× fwd, 1.245× bwd;
fwd:bwd shape is identical at 2.16 on both sides). Everything around the
model — data path, clipping, optimizer, loop Python, logging — is now
measured at parity (≈12 ms of ~11,500). So the headroom is entirely
model-execution efficiency: per-block compile scope (eager Residual/norm/
embedding/LM-head glue between 25 compiled regions ×2 passes), RoPE's
elementwise work, the 3-GEMM gated FF, and the unpadded 50257 head GEMM.
Combined with the ~4% FLOPs/token difference, this reproduces the MFU ratio
(47.0/36.4 = 1.29) exactly. **Trainer-level suspects are closed; all
remaining headroom is inside the compiled model graph.**

### 4b. Kernel-level attribution: where the 2.3 s/step actually is (2026-11-01)

Same protocol, one level deeper: `torch.profiler` (CUDA activity) wrapped
around steps 12–13 in both frameworks (`+trainer.kernel_profile=true` on
`mfu-abl/profile-timers`; `--kernel_profile=True` in the patched nanoGPT),
every kernel dumped with self-time and launch count, aggregated into the same
categories by `kprof_agg.py`. h20-size, 16×32, same H100 (jobs 31882 ours /
31889 theirs), ms per step:

| kernel category | ours (per-block compile) | ours (whole-model compile) | **ours (+ compiled loss)** | nanoGPT |
|---|---:|---:|---:|---:|
| GEMMs (nvjet) | 4,699 | 4,787 | 4,801 | 4,899 |
| flash attention | 1,863 | 1,950 | 1,961 | 2,375 |
| compiled (triton: fused glue, silu/gelu, norms, CE…) | 1,157 | 1,737 | 1,871 | 1,841 |
| **eager elementwise (fp32 mul/add armies, casts)** | **2,878** | **625** | **291** | 108 |
| **eager norm pieces (pow/reduce/rsqrt)** | **438** | 0.2 | 0 | 0 |
| softmax/CE (eager until compiled-loss) | 230 | 230 | 0 | 0 |
| cat / copies / memcpy | 147 | 0.2 | 0.2 | 0.4 |
| optimizer | 10 | 10 | 10 | 9 |
| other | 33 | 8 | 8 | 6 |
| **TOTAL busy** | **11,454** | **9,348** | **8,943** | **9,238** |
| **step total (wall)** | **11,536** | **9,451** | **9,046** | **9,250** |
| kernel launches per step | 86,713 | 60,857 | 60,377 | 40,934 |
| tokens/s | 90,697 | 110,890 | **115,862** | 113,150 |
| MFU (own FLOPs basis) | 36.4% | 44.4% | **46.4%** | 47.0% |

Columns: jobs 31882 / 31905 / 32023 / 31889. The compiled-loss column
(speedup-plan item 2, branch `mfu-abl/compiled-loss`, 2026-11-02) traces
model forward + cross-entropy into one graph (`+trainer.compile_loss=true`):
the CE row and its boundary casts vanish into triton exactly as attributed,
leaving only grad-accumulation adds (291 ms) eager. **Wall-clock: 9,046 vs
nanoGPT's 9,250 ms — we are 2.2% faster at matched workload** (+2.4%
tokens/s); nanoGPT's MFU prints higher only because its FLOPs/token
numerator is 4% larger (MHA + 2-GEMM MLP). Numerics gate: per-step losses
vs job 31905 track at max |Δ| 0.0017 over 30 steps. Speedup-plan item 3
(micro 32 × ga 16) OOMs — 73.6 GiB used + 6.1 GiB more requested (job
32024) — so 16×32 stays; recorded, not retried.

**Measured verdict — the prediction ranking was wrong, and the data names the
weakness exactly:**

1. **Our hardware-efficient work is *faster* than nanoGPT's.** GEMMs 4,699 vs
   4,899 ms and flash attention 1,863 vs 2,375 ms — GQA and our GEMM shapes
   execute fine; together we are **713 ms/step ahead** on the work that
   counts. GEMM efficiency, gated-FF shape, RoPE-inside-compiled-regions and
   the padded-vocab head GEMM are all **exonerated** (the head GEMM sits in
   the GEMM line; our triton total is *smaller* than theirs).
2. **The entire gap is the eager glue between our compiled regions**:
   +2,770 ms of unfused eager elementwise (fp32 `Mul`/`Add` kernels at ~27k
   launches/step — the pre-norm scale/rsqrt muls, residual adds, autocast
   boundary casts `bfloat16_copy`/`_to_copy`), +438 ms of eager RMSNorm
   pow/reduce pieces, +230 ms eager cross-entropy, +147 ms cat/copies ≈
   **+3.6 s of glue vs their −0.7 s triton surplus that already contains all
   of it** (their layer norms, residual adds and CE are fused rows inside
   their compiled graph: `triton_red_fused..._native_layer_norm_backward`).
   Note the dtype: the glue runs in **fp32** (norms and the residual stream
   outside compiled regions), doubling its memory traffic vs bf16.
3. Both streams are saturated (idle: 82 ms ours, 12 ms theirs of ~0.7%) — it
   is not launch-latency stalling, it is real memory-bound kernel time.

**The one lever this leaves**: widen compile scope so the glue fuses —
whole-model compile (Residual `log_name` refactor), or minimally pull the
norms + residual adds + CE into compiled regions. The measured ceiling of
that fix is ~2.9 s/step ≈ the entire remaining gap to nanoGPT at equal
workload. Vocab padding drops to a minor item; RoPE and the gated FF are off
the list entirely.

**Lever pulled (2026-11-01, branch `mfu-abl/whole-compile`, job 31905,
middle column above)**: Residual's metric logging moved into a
`@torch.compiler.disable` helper so the per-block `log_name` string never
enters tracing (per-block wandb metrics unchanged), new `+compile_model=true`
knob compiles the whole model. Refactor bit-identical with the knob off;
compiled losses track eager to ~1e-7. Result at h20: step 11,536 → **9,451
ms** (−18%), tokens/s 90,697 → **110,890** (+22%), MFU 36.4% → **44.4%** —
**2.2% behind nanoGPT wall-clock** at matched workload, from 25%. The glue
collapsed as predicted by the ceiling: eager elementwise 2,878→625, eager
norms 438→0.2, cat/copies 147→0.2, absorbed into fused triton (1,157→1,737,
still smaller than theirs). GEMMs and attention remain faster than nanoGPT's.
What remains of the +201 ms: the still-eager cross-entropy (230 ms; theirs is
inside their compiled graph — ours lives in the trainer, not the model) and
~500 ms of residual eager elementwise (largely autograd's fp32 grad-
accumulation adds, theirs 108 ms) — the next and last small levers, worth at
most ~2% each.

### 5. Exonerated along the way (measured, not guessed)

FSDP2-at-world-size-1 (beats plain bf16 in eager); the tokenization/collate/
copy pipeline (V6c: random tokens ≈ +0.3%); fused-vs-unfused optimizer; RoPE
cast hygiene; nanoGPT's *absolute* superiority at its defaults (its 37.9% at
124M/seq1024 is **below** our 36.4% h20 — small models underfill the H100;
its size-matched 45.7–47.0% is the number that counts).

## Method

- **Base commit**: the `pretraining` HEAD this plan lands on. Every variant
  branches from it: branch `mfu-abl/<name>`, built in its own worktree, carrying
  **exactly one change**. Branches are pushed (entropy pulls on the login node);
  worktrees are cleaned after, branches kept (repo policy).
- **The measurement** (Maciej, 2026-10-31): every variant runs **h16 and h20,
  bf16 only** — the most efficient configuration from stage 3 — so the ablation
  is 2 trainings × 6 improvements = 12 runs, no other sizes or dtypes. Configs
  `configs/mfu_audit/{h16,h20}.yaml` exactly as in stage 3: one entropy H100,
  seq 2048, batch 512, 30 steps, micro×ga = 16×32, window steps 10–29, wandb
  project `nano-mfu-audit`, tag `mfu_ablation` + the variant name, stage-3 seed.
- **Baseline (B0)**: stage-3 bf16 numbers — h16 = 123,367 tok/s / 27.7%
  ([sfupemo2](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/sfupemo2)), h20 =
  76,858 tok/s / 30.8%
  ([32gyzfzk](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/32gyzfzk)). One
  fresh h16 re-run at the base commit confirms node parity before any delta is
  read; if it lands within noise of stage 3, the stage-3 h20 number stands as
  the h20 baseline (saving one run of shared GPU time).
- **Noise rule**: stage-3 step-time cv was ≤1.6%, so a delta under ~3% in
  tokens/s gets one repeat run before it is believed (in either direction).
- A variant that *changes memory behaviour* (V1, V2, V5) may re-run the OOM
  probe; if the micro-batch ceiling moves, both numbers are reported: same
  16×32 (like-for-like) and best-fit (what the fix actually buys).
- Entropy rules as in stage 3: single-GPU jobs, one at a time, short runs only.
- Each variant branch still passes the local gates (pytest CI filter + black),
  and where the change claims to be a numerics no-op, a seeded CPU run gated
  bit-identical proves it.

## The six variants

| # | branch | the one change | expected | numerics |
|---|---|---|---|---|
| V1 | `mfu-abl/compile` | `trainer.compile: true` knob; `torch.compile` applied per TransformerBlock (regional compile — the recommended pairing with FSDP2), knob default off | **+25–50%** | not bit-identical (kernel fusion); loss-curve overlay vs B0 must track within noise |
| V2 | `mfu-abl/enable-gqa` | SDPA `enable_gqa=True` replaces `repeat_kv` materialization in `RoPEAttention.forward` | +2–5%, plus lower activation memory | equivalence test vs repeat_kv at tolerance (different kernel path) |
| V3 | `mfu-abl/fused-optim` | `AdamW(fused=True)` + foreach gradient clipping (replace the ~150-kernel per-param loop in `clip_gradient` with multi-tensor norms over local tensors) | +1–3% | bit-identical intent; fused AdamW on DTensor params is verified on the cluster torch (2.7.1) before the run — fallback `foreach=True` if it refuses |
| V4 | `mfu-abl/rope-hygiene` | cos/sin registered in model dtype on the right device at setup; the per-forward `.to(device, dtype)` and avoidable chunk/cat work removed; math unchanged | +1–3% eager (compile would subsume it) | **bit-identical gate** on seeded CPU run |
| V5 | `mfu-abl/plain-bf16` | the nanoGPT configuration: world-size-1 path with a plain module under `torch.autocast(bf16)` instead of FSDP2 (small knob; `trainer.distributed=null` today means fp32) | isolates the FSDP2 single-GPU tax, +3–8%? — fills the dtype-matrix hole | not comparable bit-wise (different cast points); first-loss band + loss overlay |
| V6 | `mfu-abl/data-path` | `non_blocking=True` on the pinned H2D copies + prefetch the next micro-chunk during the current forward | +0–3% at h16 (bigger at h4) | bit-identical gate (copy scheduling only) |

**V6 companion, zero code**: one run of B0 with the local pre-tokenized dummy
dataset at h16 shape — bounds what the *entire* input pipeline (tokenization +
collate + copies) costs today, independent of the V6 implementation.

**Reference run, zero code of ours**: nanoGPT itself (124M, seq 1024, its
defaults, random-token `get_batch`) on the same H100 — the true same-hardware
reference number, killing the A100-vs-H100 apples/oranges permanently.

### Which of the six are actual nanoGPT code differences

Four of the six variants replicate something nanoGPT's code literally does; two
are hygiene for components nanoGPT does not have (so they measure *our*
architecture's own overhead, not a code gap):

| # | nanoGPT's code | ours before the variant | nanoGPT feature? | measured verdict |
|---|---|---|---|---|
| V1 | `torch.compile(model)` on by default (whole model, one graph) | pure eager | **yes** | **real, +17–19%** |
| V2 | plain MHA — no `repeat_kv` exists to optimize | GQA with materialized `repeat_kv` | no (our arch only) | +2–3% eager, loses under compile |
| V3 | `AdamW(fused=True)` + single `clip_grad_norm_` call | non-fused AdamW + per-param clip loop | **yes** | noise |
| V4 | learned position embedding, zero per-step cost | RoPE with per-forward cos/sin casts | no (our arch only) | noise |
| V5 | raw module + `torch.amp.autocast(bf16)` | FSDP2 wrapper at world size 1 | **yes** | **negative** −5/−7% (eager); FSDP2 exonerated |
| V6 | memmap `.bin`, pinned, `non_blocking=True`, prefetch | on-the-fly tokenization, sync `.to(device)` | **yes** | noise (V6c: whole pipeline ≈ 0) |

So nanoGPT's four genuine code advantages decompose as: one real (compile), two
worth ~0 here (fused optimizer, data path), one worse than what we had (plain
bf16 vs FSDP2 — measured in eager; see the parity combo below for the compiled
answer). The remaining untested slice of their code advantage is compile
*scope*: whole-model single graph vs our per-block compile (Residual/norm glue
stays eager because of the `log_name` guard bug).

## Execution order

Cheapest-to-implement first, so results start arriving while the larger ones
are built: **B0 re-run → V3 → V2 → V6(+companion) → V4 → V5 → V1 → nanoGPT
reference.** One job at a time; every result appended to the table below as it
lands, with the wandb link.

## The answer in one table: MFU, method × model

| method | h12 | h16 | h20 |
|---|---:|---:|---:|
| **B0 baseline** (eager, FSDP2 bf16) | 25.2% | 28.1% | 30.8% |
| V1 torch.compile | — | **33.2%** | **36.3%** |
| V2 enable_gqa (native-GQA SDPA) | — | 28.8% | 31.8% |
| V3 fused AdamW + foreach clip | — | 28.1% | 30.8% |
| V4 RoPE cast hygiene | — | 27.8% | 30.9% |
| V5 plain module + autocast bf16 | — | 26.7% | 28.6%¹ |
| V6 async H2D + prefetch | — | 27.9% | 30.9% |
| V6c baseline + free input data | — | 28.2% | — |
| **stack (final = compile)** | **29.8%** | **32.9%** | **36.0%** |
| **parity combo V1+V3+V5+V6** (nanoGPT's full recipe) | — | **33.6%** | **36.4%** |
| **whole-model compile** (parity + `compile_model`, `log_name` refactor) | — | — | **44.4%** |
| **compiled loss** (one graph incl. CE; beats nanoGPT wall-clock) | — | — | **46.4%** |
| nanoGPT reference (its defaults: 124M, seq 1024) | — | 37.9%² | — |
| **nanoGPT size-matched** (same non-emb params, seq 2048, 1.05M tok/step) | — | **45.7%**² | **47.0%**² |

¹ micro 8 (micro 16 OOMs on the plain path). ² on the same H100, converted to
the 989e12 basis. Variants ran h16+h20 only (Maciej's trim); h12 has the
baseline and the final stack. Only V1 clears the noise bar upward; V2 clears
it in eager but reverses under compile; V5 is clearly negative; V3/V4/V6/V6c
are noise. Per-run detail below.

## Results (filled as measured; each variant = one h16 row + one h20 row)

| run | tokens/s | MFU | Δ vs B0 | micro×ga | peak mem | wandb |
|---|---:|---:|---:|---:|---:|---|
| B0 h16 (stage 3) | 123,367 | 27.7% | — | 16×32 | 44.9 GiB | [sfupemo2](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/sfupemo2) |
| B0 h20 (stage 3) | 76,858 | 30.8% | — | 16×32 | 65.9 GiB | [32gyzfzk](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/32gyzfzk) |
| B0 h16 re-run | 125,052 | 28.1% | +1.4% vs stage 3 (noise) | 16×32 | 44.9 GiB | [7ms4eafl](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/7ms4eafl) |
| V1 compile h16 | 147,922 | 33.2% | **+18.3%** | 16×32 | | [ykipu3dy](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/ykipu3dy) |
| V1 compile h20 | 90,538 | 36.3% | **+17.8%** | 16×32 | | [24uvodtr](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/24uvodtr) |

V1 footnote: the first compile attempt measured −0.2% ([sw7h19j9](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/sw7h19j9))
— a real bug, not a result: block-level compile guards on Residual's per-block
`log_name` string, dynamo's recompile_limit (8) silently dropped half the blocks
to eager. Fixed on the branch (`8e2d5f6b`): compile RoPEAttention+GatedFeedForward,
Residual stays eager, zero recompile warnings after.
| V2 enable_gqa h16 | 128,107 | 28.8% | **+2.4%** | 16×32 | | [uesc9qee](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/uesc9qee) |
| V2 enable_gqa h20 | 79,324 | 31.8% | **+3.2%** | 16×32 | | [jcdm0yeq](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/jcdm0yeq) |
| V3 fused optim h16 | 125,186 | 28.1% | +0.1% (noise) | 16×32 | | [jfmkoi13](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/jfmkoi13) |
| V3 fused optim h20 | 76,887 | 30.8% | +0.0% (noise) | 16×32 | | [mh77sqta](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/mh77sqta) |
| V4 RoPE hygiene h16 | 123,976 | 27.8% | −0.9% (noise) | 16×32 | | [26sivvlj](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/26sivvlj) |
| V4 RoPE hygiene h20 | 77,108 | 30.9% | +0.3% (noise) | 16×32 | | [615r51uv](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/615r51uv) |
| V5 plain bf16 h16 | 118,899 | 26.7% | **−4.9%** (FSDP2 is faster, not a tax) | 16×32 | | [mzv4zbvu](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/mzv4zbvu) |
| V5 plain bf16 h20 | 71,323 | 28.6% | **−7.2%** (micro 16 OOMs on this path; measured at 8×64) | 8×64 | | [84utp13j](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/84utp13j) |
| V6 data path h16 | 124,270 | 27.9% | −0.6% (noise) | 16×32 | | [l106ifci](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/l106ifci) |
| V6 data path h20 | 77,082 | 30.9% | +0.3% (noise) | 16×32 | | [2e8m9tv2](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/2e8m9tv2) |
| V6c dummy-data B0 h16 | 125,428 | 28.2% | +0.3% (pipeline costs ~0) | 16×32 | | [b8934dp6](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/b8934dp6) |
| nanoGPT ref (124M s1024, compiled) | 438,278 | **37.9%** (its printed 120.25% is on the A100 312e12 basis) | n/a | 12×40 | | job 31553 log (wandb off) |
| nanoGPT h16-eq (20L/16h/1024d=303M, s2048, compiled) | 194,600 | **45.7%** (printed 144.8%) | n/a | 16×32 | | job 31871 log (wandb off) |
| nanoGPT h20-eq (25L/20h/1280d=556M, s2048, compiled) | 113,150 | **47.0%** (printed 149.0%) | n/a | 16×32 | | job 31872 log (wandb off) |
| stack V1+V2 h16 | 145,639 | 32.7% | +16.5% — **below V1 alone** | 16×32 | | [uac9scwd](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/uac9scwd) |
| stack V1+V2 h16 repeat | 145,510 | 32.7% | reproduces the −1.6% vs V1 | 16×32 | | [e7bj6zi5](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/e7bj6zi5) |
| **stack (final = V1) h12** | 269,352 | **29.8%** | **+18.6%** vs stage-3 h12 | 32×16 | | [urnucb40](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/urnucb40) |
| **stack (final) h16** | 146,650 | **32.9%** | **+17.3%** | 16×32 | | [897ald44](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/897ald44) |
| **stack (final) h20** | 89,724 | **36.0%** | **+16.7%** | 16×32 | | [kazkf06r](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/kazkf06r) |

All three final-matrix rows ran on `mfu-abl/stack@aa58bf1f` itself; `git diff`
against the V1 branch's `8e2d5f6b` is empty on every code path, and the
measurements agree with V1's own h16/h20 runs within 0.9%.

**Interaction found**: enable_gqa wins eager (+2.4/+3.2%) but loses under
compile (−1.6%, measured twice): inductor codegens the compiled repeat_kv
better than the native-GQA SDPA call. The winners' stack is therefore **V1
only** (`mfu-abl/stack` = base + compile knob, V2 reverted with rationale in
`aa58bf1f`); V1's own h16/h20 runs ARE the stack matrix for those sizes.

## After the ablation: stacking

Interactions are real — V1 (compile) largely subsumes V4 and changes what V2/V3
are worth; V5 competes with FSDP2 rather than composing with it. So once the
per-fix deltas are in: branch `mfu-abl/stack` from base, apply the winners in
descending measured-delta order, re-measuring h16 after each addition (so the
stack is also an ordered attribution), finish with the full h12/h16/h20 matrix,
and update [backbone_mfu_audit.md](backbone_mfu_audit.md) with before/after.
What merges back into `pretraining` is the reviewed stack, not the ablation
branches. (Maciej's h16+h20-only trim applies to the per-variant ablation runs;
the final stacked measurement keeps the full h12/h16/h20 matrix per the goal —
three runs, once.)

Acceptance target from the gap audit: **h16 ≥ 35–40% MFU** (≈350–400 TFLOP/s).

## nanoGPT-parity combo: V1+V3+V5+V6 together (requested 2026-10-31)

Question: if we switch on *everything nanoGPT's code does* at once — compile
(V1) + fused AdamW/foreach clip (V3) + plain module under autocast bf16 (V5) +
async H2D with prefetch (V6) — does our implementation match nanoGPT's 37.9%?
This is not the winners' stack (V5 lost in eager); it is the "their exact
recipe, our model" configuration. The open interaction it answers: V5 was only
ever measured eager — FSDP2's pre-forward all-gathers may interact differently
with compiled regions.

- Branch: `mfu-abl/nanogpt-parity` = `mfu-abl/compile` + cherry-picks of
  `ba7a274e` (V3), `1450907d` (V5), `596d7cf4` (V6). V2/V4 stay out (not
  nanoGPT features).
- Run config: `+compile_blocks=true trainer.distributed=null
  +trainer.autocast_bf16=true`; h16 at 16×32, h20 at 8×64 (micro 16 OOMed on
  the plain path at h20).
- Same measurement protocol as the ablation (stage-3 configs, 30 steps,
  single H100, `scripts/mfu_window.py`).

| run | tokens/s | MFU | vs stack (V1 only) | vs nanoGPT 37.9% | wandb |
|---|---:|---:|---:|---:|---|
| parity h16 (16×32) | 149,474 | **33.6%** | +0.7pt (stack: 32.9%) | −4.3pt | [z5klwygs](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/z5klwygs) |
| parity h20 (8×64) | 87,001 | 34.9% | −1.1pt — micro-batch confound, see next row | −3.0pt | [754pxldo](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/754pxldo) |
| parity h20 (16×32) | 90,697 | **36.4%** | +0.4pt (stack: 36.0%) | −1.5pt | [abzl2438](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/abzl2438) |
| whole-model compile h20 (16×32) | 110,890 | **44.4%** | +8.0pt vs parity (+22% tok/s) | −2.6pt (−2.2% wall) | job 31905 log (stdout) |
| compiled-loss h20 (16×32) | 115,862 | **46.4%** | +4.5% tok/s vs whole-compile | **+2.2% faster wall-clock than nanoGPT** | job 32023 log (stdout) |
| compiled-loss h20 micro 32×16 | — | OOM | 73.6 GiB used + 6.1 GiB requested; 16×32 stays | — | job 32024 log |

Measured 2026-10-31 (jobs 31677/31678/31680). Findings:

1. **Compile flips V5's sign.** Plain module + autocast(bf16), −4.9/−7.2% in
   eager, is +0.4–0.7pt *over* the FSDP2 stack once the blocks are compiled —
   the eager FSDP2 advantage came from its pre-cast bf16 compute path, which
   inductor now provides anyway. Compile also cuts activation memory enough
   that h20 fits micro 16 on the plain path (eager V5 OOMed there), which is
   where the 8×64 row's −1.1pt comes from: halved micro, not the recipe.
2. **V3+V5+V6 together are worth ~+0.5pt on top of compile** — consistent sign
   at both sizes but below the ablation's 3% noise bar; individually each was
   noise or negative. Compile remains the only change that matters.
3. **Parity with nanoGPT — corrected by the size-matched runs (2026-11-01).**
   The first conclusion ("nearly matched at h20, the rest is workload")
   compared different workloads and was too kind to us. nanoGPT run at *our*
   shape — same non-embedding params, seq 2048, 1.05M tokens/step, same H100
   (jobs 31871/31872: 20L/16h/1024d → 303M and 25L/20h/1280d → 556M) — gets
   **45.7% (h16-size) and 47.0% (h20-size)**: bigger GEMMs and longer seq
   *raise* its MFU. At matched workload the honest gap is **12.1pt / 10.6pt**,
   i.e. nanoGPT moves 1.30× / 1.25× more tokens per second than our parity
   branch at equal size. With every known trainer-level code difference now
   replicated, that gap is carried by: (a) compile scope — their whole-model
   single graph vs our per-block compile with eager Residual/norm/embedding
   glue; (b) architecture — RoPE's per-layer uncounted elementwise+cat work vs
   a free position-embedding add, 3-GEMM gated FF vs 2-GEMM MLP, unpadded
   50257 LM-head GEMM vs their 50304 (multiple of 64); (c) any remaining
   trainer-loop overhead (per-chunk Python, loss bookkeeping). Next levers, in
   testable order: whole-model compile (needs the Residual `log_name`
   refactor), vocab padding to 50304, then the architecture costs which are
   paper choices, not bugs.

## Parked for Maciej (changes deliberately NOT in the ablation)

- Fused QKV projection (changes parameter layout/init).
- No weight decay on 1-D params, weight tying (change training dynamics).
- Pre-tokenized memmap dataset (bigger build; V6c's dummy-data run measures its
  ceiling first — if the pipeline costs ~0 at h16, it stays parked until
  multi-GPU work needs it).
- Activation checkpointing (a memory knob, not an MFU fix at fixed micro-batch;
  it belongs to the stacking phase where bigger micros become reachable).
