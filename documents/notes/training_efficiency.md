# Training efficiency: the complete record (2026-10-30 → 2026-11-03)

One document, chronological: the baseline audit, the nanoGPT gap analysis, the
six-variant ablation, the compile-scope work, and what finally shipped. **Final
results are in Part 6**; everything before it is how we got there.

Merged 2026-11-03 from five notes that had grown apart and were separately
indexed (or not indexed at all): `pretraining_stage3_mfu_plan.md`,
`backbone_mfu_audit.md`, `mfu_gap_nanogpt_audit.md`, `mfu_ablation_plan.md`,
`mfu_speedup_plan.md`. All five are in git history before this commit; every
measurement, table and wandb link they held is reproduced here. Timestamped
per-session detail stays in [pretraining_reset_log.md](pretraining_reset_log.md).

**The short version.** A stage-3 audit measured the backbone at 25.2/27.7/30.8%
MFU (h12/h16/h20) on one H100 and found no efficiency anomaly — just no
`torch.compile` anywhere. nanoGPT — its own GPT-2 architecture (MHA + GELU MLP),
size-matched on non-embedding params, run on the same hardware — scored
45.7/47.0%. Six single-change branches established that compile was the only real
blocker; the data path, FSDP2, the optimizer and RoPE casts were all measured
innocent. Kernel profiling then showed the entire remaining gap was eager glue
between our compiled regions, which whole-model compile and a compiled loss
removed. Final state: **46.4% MFU at h20, 9,046 ms/step — 2.2% faster wall-clock
than nanoGPT at matched workload**, with our GEMMs and attention measurably
faster than theirs.

## Conventions (fixed once, used by every number below)

- **Standard workload**: sequence length 2048, 512 sequences per step, 30-step
  runs, one entropy H100 80GB HBM3 (`h100a`), bf16. Tokens/step = 512 × 2047 =
  **1,048,064** model-consumed tokens (see the token-accounting amendment in
  Part 1.2).
- **FLOPs/token (training)** = `6·P + 12·L·s·d` (PaLM convention: fwd 2P + 4Lsd,
  backward 2× fwd; causal masking NOT discounted), where `P = 15·L·d² + V·d`
  (all matmul params: attention 3d²/layer under GQA-H/2, gated FF 12d²/layer,
  untied head; embedding lookup excluded), `s = 2048`, `V = 50304`. Measured from
  the live model, not the config's claim. Worked example, h16 (d=1024, L=16):
  P = 303.2M → **2.22 GFLOPs/token**.
- **MFU** = (tokens/s × FLOPs/token) / peak-FLOPs(GPU, dtype). Peak table:
  A100 bf16 312e12, H100 SXM bf16 989e12, H100 PCIe bf16 756e12; fp32 peaks are
  the same tensor-core numbers at tf32 (A100 156e12, H100 SXM 495e12). Unknown
  GPU name → log tokens/s and a warning, mfu stays null rather than wrong.
- **Timing window**: steps 10–29 of a 30-step run; steps 0–9 excluded (allocator
  warmup, dataloader spin-up, cache effects).
- **Comparing against nanoGPT**: its printed MFU is on a hard-coded A100 312e12
  basis; multiply by 312/989 to convert to ours. Both frameworks count
  FLOPs/token the same way, so MFUs are comparable — but **tokens/s and wall
  clock are the arbitration-free numbers**, because the two architectures need
  different FLOPs per token (theirs ~4% more).
- **Which model each label means** — three different models get called "h20" in
  this document (likewise at h16):
  - **h20 (ours)**: the H-balanced model, H=20 — 20 blocks, dmodel 1280, **GQA**
    (20 q / 10 kv heads, `RoPEAttention`), **SiLU-gated FF** (`GatedFeedForward`,
    3 GEMMs, dff 5120), RoPE positions. 620.2M params (491.6M non-embedding),
    3.97 GFLOPs/token. Every row labeled plain `h12`/`h16`/`h20` is this
    architecture.
  - **nanoGPT h20-eq**: nanoGPT's own GPT-2 architecture, sized to match our
    h20's non-embedding params and workload — 25 layers, dmodel 1280, **MHA**
    (20 heads), **2-GEMM GELU MLP**, learned positions; 4.122 GFLOPs/token (~4%
    more than ours). Its attention and FF methods do **not** match ours: every
    cross-framework number outside Parts 4.4–4.5 compares these two *different
    architectures* at matched size and workload.
  - **GPT-2 arch clone (our framework)**: branch `mfu-abl/nanogpt-arch`,
    Part 4.4 — the h20-eq shape (25L, MHA, GELU MLP) inside our trainer, RoPE
    kept. The only architecture-matched ours-vs-nanoGPT comparison in this
    document.
- wandb project: [ideas_cv/nano-mfu-audit](https://wandb.ai/ideas_cv/nano-mfu-audit).

---

## Part 1 — Stage 3: the baseline audit (2026-10-30)

### 1.1 The plan

Redefined by Maciej 2026-10-30: before any new attention implementations,
validate the current backbone end to end with **real single-GPU trainings on
entropy** and measure implementation efficiency as **MFU**. Models h12/h16/h20,
few-step runs, a new wandb project. The point was to exercise everything that had
not run on a GPU since the reset and to catch bugs precisely, not to train
anything real.

Ground rules: single-GPU jobs only, one at a time (shared quota); direct commits
on `pretraining` with pytest + black green after each; MFU instrumentation must
be a **numerics no-op** (seeded local runs bit-identical before/after).

**Phase 0 — preflight** (local + entropy, no GPU experiments):

| # | Test | Pass criterion |
|---|---|---|
| P0.1 | Add `h12.yaml`, `h20.yaml`; extend `test_h_balanced` parametrize to 12 and 20 | exact closed-form param counts: h12 = 183.4M, h20 = 620.2M class (formula incl. norms) |
| P0.2 | New runnable configs `configs/mfu_audit/{h12,h16,h20}.yaml`: entropy, fineweb, seq 2048, batch 512, n_steps 30, cosine warmup 10, lr 2^-12, eval and save off | hygiene suite composes + resolves them |
| P0.3 | Stage the NeoX tokenizer on entropy and prove it loads with `HF_HUB_OFFLINE=1` on a compute node | tokenizer loads offline; ids ≤ 50303 |
| P0.4 | Run the full pytest suite on entropy — first suite run there since the stage-1/2 rewrite | suite green on the cluster env |
| P0.5 | Verify fineweb paths at seq 2048: compose h12 on entropy, pull 3 batches | batches shaped (…, 2048), max id ≤ 50303 |

**Phase 1 — instrumentation** (local, gated bit-identical): the MFU metric in the
trainer (FLOPs/token computed once at init from the live model, GPU peak from the
table, per-step `train/step_time` / `tokens_per_s` / `mfu` plus peak memory); a
first-loss sanity hook (fresh run's first train/loss must be ln(50304) = 10.826 ±
0.15 — wrong init, vocab mismatch or head-wiring bug all show up before a
GPU-hour is spent); and a bf16 autocast knob.

**Phase 2 — entropy experiments**, two jobs per model: an OOM probe that doubles
the micro-batch until it fails and backs off one notch, then a 30-step
measurement run at that setting. E1 h4 pipeline smoke, E2–E4 h12/h16/h20, E5 the
eval path, E6 resume through a checkpoint, E7 the attention registry (`mha`,
`mqa`, `mix: {type: mqa, layers: every_2nd}`).

E3 carried an investigation tripwire: if bf16 MFU at h16 came in below 25% (A100)
/ 20% (H100), stop and investigate before proceeding — prime suspects being a
dataloader/tokenizer stall, fp32 leakage, or unintended host-device syncs. **It
never fired.**

### 1.2 Amendments during execution

- **Token accounting**: `processed_tokens` counts tokens the model consumes —
  `batch × (width − 1)` = 512 × 2047 = **1,048,064** per step — not the raw
  512 × 2048 the plan stated. MFU is unaffected: tokens/s and FLOPs/token both use
  the consumed-token basis, so the ratio is consistent.
- **Submission**: direct sbatch from the pinned `~/nano-mfu` clone instead of
  `run_exp.py` (entropy :22 block; compute nodes cannot reach GitHub at all —
  pulls happen on the login node). Provenance: the job logs the clone's commit,
  wandb logs the config.
- **Dtype**: no autocast knob in the audit — FSDP2's hard-coded
  MixedPrecisionPolicy (bf16 compute / fp32 reduce) is the production path and is
  what was measured. The "fp32/tf32 and bf16" matrix was satisfied by fp32 control
  runs at all three sizes with `trainer.distributed=null` (a plain fp32 module),
  reported against both the tf32 tensor-core peak (495e12) and the FP32 vector
  peak (67e12).
- **tf32 cells**: measured as a separate cell (TF32 forced on via torch.backends
  flags; fp32 storage either way) at all three sizes. Reasoned deviation: the
  non-bf16 cells reuse conservative micro-batches derived from the bf16 probes
  (halved) instead of running their own probe jobs — the measured peaks prove the
  headroom (h16-fp32: 40.8 GiB of 80), and a larger micro could only improve
  numbers whose verdict (bf16 ≫ fp32) is already decided at 5.7–6.9×.

### 1.3 Results: the baseline MFU table

**Status: complete, 2026-10-30.** All runs one H100, bf16 via FSDP2's
MixedPrecisionPolicy, seq 2048, 512 sequences/step, 30 steps, window 10–29.

| run | params | GFLOPs/tok | micro×ga | tokens/s | **MFU** | step time (cv) | peak mem | wandb |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| h4 (E1 smoke) | 29.7M | 0.126 | 64×8 | 1,242,950 | **15.8%** | 0.843s (1.0%) | 45.2 GiB | [7dabc7pw](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/7dabc7pw) |
| h12 | 183.4M | 1.095 | 32×16 | 227,158 | **25.2%** | 4.614s (0.7%) | 55.5 GiB | [2xyxl874](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/2xyxl874) |
| h16 | 354.6M | 2.22 | 16×32 | 123,367 | **27.7%** | 8.498s (1.6%) | 44.9 GiB | [sfupemo2](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/sfupemo2) |
| h20 | 620.2M | 3.97 | 16×32 | 76,858 | **30.8%** | 13.636s (0.1%) | 65.9 GiB | [32gyzfzk](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/32gyzfzk) |
| h16 mha | 371.4M | 2.322 | 16×32 | 123,950 | 29.1% | 8.456s (0.1%) | — | [2f6nfosg](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/2f6nfosg) |
| h16 mqa | 339.9M | 2.133 | 16×32 | 133,322 | 28.8% | 7.863s (1.5%) | — | [2fp8a1w7](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/2fp8a1w7) |
| h16 gqa+mqa every_2nd | 347.3M | 2.177 | 16×32 | 127,991 | 28.2% | 8.191s (1.6%) | — | [alj69rdy](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/alj69rdy) |
| h12 **fp32** (no FSDP policy) | 183.4M | 1.095 | 8×64 | 40,066 | 4.4%* | 26.158s (0.1%) | — | [wp0vzux5](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/wp0vzux5) |
| h16 **fp32** (no FSDP policy) | 354.6M | 2.221 | 8×64 | 20,636 | 4.6%* | 50.788s (0.0%) | 40.8 GiB | [cle0ktz3](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/cle0ktz3) |
| h20 **fp32** (no FSDP policy) | 620.2M | 3.97 | 4×128 | 11,128 | 4.5%* | 94.181s (0.0%) | — | [kzruq6zo](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/kzruq6zo) |
| h12 **tf32** | 183.4M | 1.095 | 8×64 | 90,930 | 10.1%* | 11.526s (0.1%) | — | [42rph6xn](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/42rph6xn) |
| h16 **tf32** | 354.6M | 2.221 | 8×64 | 52,131 | 11.7%* | 20.105s (0.1%) | — | [t2qhpdsf](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/t2qhpdsf) |
| h20 **tf32** | 620.2M | 3.97 | 4×128 | 30,013 | 12.0%* | 34.920s (0.1%) | — | [igtr237s](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/igtr237s) |

MFU grows with model size as utilization should — **no efficiency anomaly found**.
Readings: mqa buys h16 +8% tokens/s; ~28–31% without `torch.compile` or activation
checkpointing is a healthy baseline, and both were the obvious future levers.

The fp32 controls are **5.7–6.9× slower** than the FSDP2-bf16 production path,
achieving a near-constant **43.9/45.8/44.2 TFLOP/s at h12/h16/h20**. *Starred MFU
is against the bf16 peak for column comparability; the proper bases for that
~45 TFLOP/s are **~9% of the tf32 tensor-core peak (495e12)** — proving TF32 is off
by default (PyTorch ships `allow_tf32=False`) — and **66–68% of the FP32 vector
peak (67e12)**: the plain path is compute-bound at fp32-vector rates, uniformly
across sizes. The tf32 cells sit cleanly between: at h16 the hierarchy is
**bf16 : tf32 : fp32 = 123.4k : 52.1k : 20.6k tokens/s (6.0 : 2.5 : 1)**, tf32
achieving 100–119 TFLOP/s = 20–24% of its 495e12 peak, with first losses identical
to fp32's (storage unchanged). **Verdict: bf16 stays the default, now by
measurement over the full {fp32, tf32, bf16} × {h12, h16, h20} matrix.**

### 1.4 The E7 ordering tripwire, resolved with evidence

The plan expected `mqa ≥ gqa ≥ mha` in tokens/s and named the kv wiring suspect on
a violation. Measured: gqa 123,367 < mha 123,950 — a 0.47% inversion, so the check
was run. The GFLOPs/token each run measured **from its own live parameters** are
gqa 2.221, mha 2.322, mqa 2.133, mixed 2.177 — matching the closed forms for
*correct* kv widths to three digits (mha carries exactly +2·8·64·1024 k/v rows per
layer, mqa exactly the fewer, and the mixed stack is exactly the gqa/mqa midpoint,
eight layers each). The wiring is therefore proven right on the cluster; the
inversion is wall-clock noise (0.47%, inside the windows' cv — gqa's was the
noisiest at 1.6%), reflecting that gqa's kv saving is ~0.3% of FLOPs at this size:
its real win is kv-cache memory at inference, not training speed.

### 1.5 Correctness results (the "does everything still work" half)

| check | result |
|---|---|
| Full test suite on entropy's cluster env | **172 passed, 0 failed** (first run since the reset) |
| Fineweb at seq 2048 through the real configs | batches (·, 2048), all ids < 50304 |
| NeoX tokenizer offline on compute nodes | staged into `/storage_nvme_4/nano/hf_cache`, len 50277 |
| First loss of a fresh model | 11.19–11.25 ≈ CPU-fp32 control 11.2235 (≈ln V + 0.4 at seq 2048; GPU is one bf16 ulp away) |
| Token accounting | exact: 30 × 1,048,064 on every run |
| Periodic eval + final_ce at seq 2048 (E5) | fired at steps 10/20 (8.125→7.25); final_ce over **exactly** 8192 holdout samples: 7.107 ([7y2zup9q](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/7y2zup9q)) |
| Resume through a dcp checkpoint on GPU (E6) | **bit-identical**: max loss delta 0.0 over 14 overlapping steps — measured on h12 first and then re-run on the plan's h16 with the same result (job 31210) |
| Pattern-mixed stack under a real optimizer | gqa+mqa every_2nd trains; loss 11.25→7.19 |

### 1.6 Bugs found and fixed (each with a regression test)

1. **Hardcoded rendezvous ports** — 18 tests failed on the shared node, zero
   locally: fixture/subprocess tests pinned MASTER_PORT and leaked it via
   `setdefault`. Now `conftest.free_port()` asks the OS. (`a83d48f8`)
2. **`int(os.environ.get("WORLD_SIZE"))` → `int(None)`** in `AbstractDataset` — a
   TypeError naming neither variable nor fix for any dataloader built outside
   main.py. Defaults to 1/0. (`393d200`)
3. **Same bug one layer up** — `get_dataloader` used `os.environ[...]`, bare
   KeyError. (`855b00c`)
4. **Throughput stub test probed CUDA on GPU nodes** — found by the first entropy
   suite run. (`ff673c4e`)
5. **Checkpoint-artifact test assumed the CPU shape** — a CUDA node writes the
   world-size-1 model as a dcp shard (`__0_0.distcp`), which resumes perfectly;
   the test now accepts both forms. (`393d2000`)
6. **First-loss criterion mis-calibrated** — fresh-model CE at seq 2048 is
   ~ln(V)+0.4 on any device (proved by CPU control); band recalibrated with the
   calibration written down. (`ad85c17e`)

### 1.7 Measurement infrastructure added

[src/core/mfu.py](../../src/core/mfu.py) (measured FLOPs from the live model, bf16
peak table, None-not-wrong for unknown GPUs), per-step `train/step_time` /
`tokens_per_s` / `mfu` / peak-memory logging (numerics-no-op, gated bit-identical),
the first-loss sanity in the e2e tests, and
[scripts/mfu_window.py](../../scripts/mfu_window.py) (wandb run → a table row plus
pass/fail verdicts).

---

## Part 2 — The nanoGPT gap audit (2026-10-31)

Prompted by: nanoGPT reports 37–45% MFU (124M GPT-2, A100, bf16, torch.compile,
flash SDPA) where our audit measured 25.2/27.7/30.8%. This part audits
karpathy/nanoGPT (model.py + train.py, cloned at HEAD) against our stack.

### 2.1 First, the denominator: we are not actually 1.5× slower

MFU divides by the GPU's peak, and the peaks differ by 3.17×:

| | achieved TFLOP/s | basis |
|---|---:|---|
| nanoGPT 124M, A100, 37–45% | **115–140** | 312e12 |
| ours h12 (183M), H100, 25.2% | **249** | 989e12 |
| ours h16 (355M), H100, 27.7% | **274** | 989e12 |
| ours h20 (620M), H100, 30.8% | **305** | 989e12 |

Per GPU we push ~2× nanoGPT's cited absolute throughput. H100s are also harder to
fill than A100s (same code scores lower MFU% on the bigger peak), and nanoGPT
measures at seq 1024 where the lower-efficiency attention share of wall-clock is
half of ours at 2048. So "outperformed by simple nanoGPT" is mostly the
denominator — **but** tuned H100 implementations at these sizes do reach ~40%+, so
roughly 10–15 points were judged real and recoverable.

### 2.2 Gap sources, ranked (audited line by line)

1. **No torch.compile anywhere** (`grep torch.compile` over main.py/trainer: zero
   hits). nanoGPT's headline is *with* compile; its README credits compile with
   cutting iteration time ~1.8×. Everything below is eager kernel-chain overhead
   (RoPE, gate, norms, CE) — exactly what inductor fuses. Expected +25–50%.
   **The single biggest item.**
2. **Eager RoPE per layer**: chunk/cat/neg/mul/mul/add plus a per-forward
   `.to(device, dtype)` of cos/sin — ~8 extra memory-bound kernel passes over Q
   and K per layer per micro-forward. nanoGPT's positional cost is one embedding
   add for the whole model.
3. **`repeat_kv` materialization before SDPA**: K/V physically expanded to full
   query width, costing bandwidth and activation memory (directly lowering the
   OOM-probe micro-batch ceiling). Torch 2.7.1 has SDPA `enable_gqa=True`.
4. **FSDP2 wrapped at world size 1**: every block + embedding + head is
   `fully_shard`-ed even single-GPU, so each micro-forward pays per-module hooks,
   DTensor dispatch, and the fp32→bf16 parameter recast — at ga=32–64 the whole
   model is recast dozens of times per step with no communication to hide behind.
   Our dtype matrix had a hole here: FSDP2-bf16, plain-fp32, plain-tf32 measured,
   **never plain + autocast-bf16**, the exact nanoGPT configuration.
5. **Optimizer and clipping, non-fused**: plain `AdamW` (nanoGPT: `fused=True`),
   and our FSDP2 grad-clip path launches one `.norm()` kernel per parameter
   (~150 serial launches/step) where `clip_grad_norm_`'s foreach path uses a
   handful of multi-tensor kernels.
6. **Data path**: on-the-fly NeoX tokenization in `num_workers: 2`, int64 collate,
   synchronous `.to(device)` per micro-chunk (pinned but no `non_blocking=True`,
   no prefetch overlap) — vs nanoGPT's pre-tokenized uint16 memmap with async
   prefetch issued *during* the forward.
7. **Not inefficiencies, but MFU% depressors to remember when comparing**: seq
   2048 doubles the attention share (flash runs below dense-GEMM efficiency); GQA
   cuts counted FLOPs ~proportionally more than wall-clock; gated FF = 3 smaller
   GEMMs vs MLP's 2 bigger; separate q/k/v projections vs nanoGPT's single fused
   `c_attn` GEMM.

Exonerated by this audit before any run: the per-layer Residual norm
instrumentation (gated to every 100 steps — fires at step 0 only in a 30-step run,
outside the window), and `F.cross_entropy` usage (same shape as nanoGPT's).

### 2.3 The plan it produced

**Phase A — equal-footing measurements**: A1 h16 plain module + autocast(bf16) with
no FSDP2 (isolates FSDP2's single-GPU overhead); A2 h16 FSDP2-bf16 + torch.compile;
A3 **nanoGPT itself on our H100** (random-token `get_batch` so no openwebtext prep)
— the true reference number on our hardware, killing the A100 apples/oranges.

**Phase B** — one committed change per item, ranked by expected win: compile knob;
`enable_gqa`; fused AdamW + foreach clip; RoPE hygiene; plain+autocast for
world_size==1; activation checkpointing; pre-tokenized memmap + async H2D.

**Phase C — acceptance**: re-run the h12/h16/h20 matrix with the winners enabled.
Target **h16 ≥ 35–40% MFU** (≈350–400 TFLOP/s).

---

## Part 3 — The six-variant ablation (2026-10-31)

Maciej's directive: **implement each improvement in a separate tree off the current
code version, run a training with each, and compare to the baseline** — so the
result is a per-fix MFU delta that names the actual blocker, not a stack where the
first change takes credit for the rest.

### 3.1 Method (the protocol every number in Parts 3–5 follows)

- **Base commit**: the `pretraining` HEAD this plan landed on. Every variant
  branches from it: branch `mfu-abl/<name>`, built in its own worktree, carrying
  **exactly one change**. Branches are pushed (entropy pulls on the login node);
  worktrees cleaned after, branches kept (repo policy).
- **The measurement**: every variant runs **h16 and h20, bf16 only** — the most
  efficient configuration from stage 3 — so the ablation is 2 trainings × 6
  improvements = 12 runs. Configs `configs/mfu_audit/{h16,h20}.yaml` exactly as in
  stage 3: one entropy H100, seq 2048, batch 512, 30 steps, micro×ga = 16×32,
  window steps 10–29, wandb project `nano-mfu-audit`, tag `mfu_ablation` + the
  variant name, stage-3 seed.
- **Baseline (B0)**: stage-3 bf16 numbers. One fresh h16 re-run at the base commit
  confirms node parity before any delta is read; if it lands within noise of stage
  3, the stage-3 h20 number stands as the h20 baseline (saving shared GPU time).
- **Noise rule**: stage-3 step-time cv was ≤1.6%, so **a delta under ~3% in
  tokens/s gets one repeat run before it is believed**, in either direction.
- A variant that *changes memory behaviour* (V1, V2, V5) may re-run the OOM probe;
  if the micro-batch ceiling moves, **both numbers are reported**: same 16×32
  (like-for-like) and best-fit (what the fix actually buys).
- Each variant branch passes the local gates (pytest CI filter + black), and where
  the change claims to be a numerics no-op, **a seeded CPU run gated bit-identical
  proves it**.
- Entropy rules: single-GPU jobs, one at a time, short runs only.

**Execution order** was cheapest-to-implement first, so results start arriving
while the larger ones are built: B0 re-run → V3 → V2 → V6(+companion) → V4 → V5 →
V1 → nanoGPT reference.

### 3.2 The six variants

| # | branch | the one change | expected | numerics |
|---|---|---|---|---|
| V1 | `mfu-abl/compile` | `torch.compile` applied per TransformerBlock (regional compile — the recommended pairing with FSDP2), knob default off | **+25–50%** | not bit-identical (kernel fusion); loss-curve overlay vs B0 must track within noise |
| V2 | `mfu-abl/enable-gqa` | SDPA `enable_gqa=True` replaces `repeat_kv` materialization in `RoPEAttention.forward` | +2–5%, plus lower activation memory | equivalence test vs repeat_kv at tolerance |
| V3 | `mfu-abl/fused-optim` | `AdamW(fused=True)` + foreach gradient clipping (replaces the ~150-kernel per-param loop) | +1–3% | bit-identical intent; fused AdamW on DTensor params verified on cluster torch 2.7.1 first |
| V4 | `mfu-abl/rope-hygiene` | cos/sin registered in model dtype on the right device at setup; per-forward `.to(device, dtype)` and avoidable chunk/cat removed; math unchanged | +1–3% eager | **bit-identical gate** on seeded CPU run |
| V5 | `mfu-abl/plain-bf16` | the nanoGPT configuration: world-size-1 plain module under `torch.autocast(bf16)` instead of FSDP2 | isolates the FSDP2 single-GPU tax, +3–8%? | not comparable bit-wise; first-loss band + loss overlay |
| V6 | `mfu-abl/data-path` | `non_blocking=True` on the pinned H2D copies + prefetch the next micro-chunk during the current forward | +0–3% at h16 | bit-identical gate (copy scheduling only) |

**V6 companion, zero code**: one run of B0 with the local pre-tokenized dummy
dataset at h16 shape — bounds what the *entire* input pipeline (tokenization +
collate + copies) costs, independent of the V6 implementation.

**Reference run, zero code of ours**: nanoGPT itself on the same H100.

### 3.3 Which of the six are actual nanoGPT code differences

Four of six replicate something nanoGPT's code literally does; two are hygiene for
components nanoGPT does not have (so they measure *our* architecture's overhead,
not a code gap):

| # | nanoGPT's code | ours before the variant | nanoGPT feature? | measured verdict |
|---|---|---|---|---|
| V1 | `torch.compile(model)` on by default (whole model, one graph) | pure eager | **yes** | **real, +17–19%** |
| V2 | plain MHA — no `repeat_kv` exists to optimize | GQA with materialized `repeat_kv` | no (our arch only) | +2–3% eager, loses under compile |
| V3 | `AdamW(fused=True)` + single `clip_grad_norm_` | non-fused AdamW + per-param clip loop | **yes** | noise |
| V4 | learned position embedding, zero per-step cost | RoPE with per-forward cos/sin casts | no (our arch only) | noise |
| V5 | raw module + `torch.amp.autocast(bf16)` | FSDP2 wrapper at world size 1 | **yes** | **negative** −5/−7% (eager); FSDP2 exonerated |
| V6 | memmap `.bin`, pinned, `non_blocking=True`, prefetch | on-the-fly tokenization, sync `.to(device)` | **yes** | noise (V6c: whole pipeline ≈ 0) |

nanoGPT's four genuine code advantages decompose as: one real (compile), two worth
~0 here, one worse than what we had. The remaining untested slice was compile
*scope* — addressed in Part 4.

### 3.4 Per-variant results

| run | tokens/s | MFU | Δ vs B0 | micro×ga | peak mem | wandb |
|---|---:|---:|---:|---:|---:|---|
| B0 h16 (stage 3) | 123,367 | 27.7% | — | 16×32 | 44.9 GiB | [sfupemo2](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/sfupemo2) |
| B0 h20 (stage 3) | 76,858 | 30.8% | — | 16×32 | 65.9 GiB | [32gyzfzk](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/32gyzfzk) |
| B0 h16 re-run | 125,052 | 28.1% | +1.4% vs stage 3 (noise) | 16×32 | 44.9 GiB | [7ms4eafl](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/7ms4eafl) |
| V1 compile h16 | 147,922 | 33.2% | **+18.3%** | 16×32 | | [ykipu3dy](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/ykipu3dy) |
| V1 compile h20 | 90,538 | 36.3% | **+17.8%** | 16×32 | | [24uvodtr](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/24uvodtr) |
| V2 enable_gqa h16 | 128,107 | 28.8% | **+2.4%** | 16×32 | | [uesc9qee](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/uesc9qee) |
| V2 enable_gqa h20 | 79,324 | 31.8% | **+3.2%** | 16×32 | | [jcdm0yeq](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/jcdm0yeq) |
| V3 fused optim h16 | 125,186 | 28.1% | +0.1% (noise) | 16×32 | | [jfmkoi13](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/jfmkoi13) |
| V3 fused optim h20 | 76,887 | 30.8% | +0.0% (noise) | 16×32 | | [mh77sqta](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/mh77sqta) |
| V4 RoPE hygiene h16 | 123,976 | 27.8% | −0.9% (noise) | 16×32 | | [26sivvlj](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/26sivvlj) |
| V4 RoPE hygiene h20 | 77,108 | 30.9% | +0.3% (noise) | 16×32 | | [615r51uv](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/615r51uv) |
| V5 plain bf16 h16 | 118,899 | 26.7% | **−4.9%** (FSDP2 is faster, not a tax) | 16×32 | | [mzv4zbvu](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/mzv4zbvu) |
| V5 plain bf16 h20 | 71,323 | 28.6% | **−7.2%** (micro 16 OOMs on this path) | 8×64 | | [84utp13j](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/84utp13j) |
| V6 data path h16 | 124,270 | 27.9% | −0.6% (noise) | 16×32 | | [l106ifci](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/l106ifci) |
| V6 data path h20 | 77,082 | 30.9% | +0.3% (noise) | 16×32 | | [2e8m9tv2](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/2e8m9tv2) |
| V6c dummy-data B0 h16 | 125,428 | 28.2% | +0.3% (pipeline costs ~0) | 16×32 | | [b8934dp6](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/b8934dp6) |
| stack V1+V2 h16 | 145,639 | 32.7% | +16.5% — **below V1 alone** | 16×32 | | [uac9scwd](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/uac9scwd) |
| stack V1+V2 h16 repeat | 145,510 | 32.7% | reproduces the −1.6% vs V1 | 16×32 | | [e7bj6zi5](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/e7bj6zi5) |
| **stack (final = V1) h12** | 269,352 | **29.8%** | **+18.6%** vs stage-3 h12 | 32×16 | | [urnucb40](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/urnucb40) |
| **stack (final) h16** | 146,650 | **32.9%** | **+17.3%** | 16×32 | | [897ald44](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/897ald44) |
| **stack (final) h20** | 89,724 | **36.0%** | **+16.7%** | 16×32 | | [kazkf06r](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/kazkf06r) |
| nanoGPT ref (124M s1024, compiled) | 438,278 | **37.9%** (printed 120.25% on the A100 basis) | n/a | 12×40 | | job 31553 log (wandb off) |
| nanoGPT h16-eq (20L/16h/1024d=303M, s2048) | 194,600 | **45.7%** (printed 144.8%) | n/a | 16×32 | | job 31871 log |
| nanoGPT h20-eq (25L/20h/1280d=556M, s2048) | 113,150 | **47.0%** (printed 149.0%) | n/a | 16×32 | | job 31872 log |

All three final-matrix rows ran on `mfu-abl/stack@aa58bf1f`; `git diff` against the
V1 branch's `8e2d5f6b` is empty on every code path, and the measurements agree with
V1's own h16/h20 runs within 0.9%.

**Model identity in the reference rows**: the two `*-eq` rows are **nanoGPT's
GPT-2 architecture** (MHA, 2-GEMM GELU MLP, learned positions) size-matched to
our models' non-embedding params — they do *not* use our attention (GQA) or FF
(SiLU-gated) methods. The architecture-matched comparison is Part 4.4.

**The V1 bug worth remembering.** The first compile attempt measured −0.2%
([sw7h19j9](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/sw7h19j9)) — a real bug,
not a result: **block-level compile guards on Residual's per-block `log_name`
string, so dynamo's recompile_limit (8) silently dropped half the blocks back to
eager.** Fixed on the branch (`8e2d5f6b`) by compiling RoPEAttention +
GatedFeedForward instead, leaving Residual eager; zero recompile warnings after.
*Any future per-module compile work must check for recompile warnings — this
failure is silent and looks like "compile doesn't help here".*

**Interaction found**: `enable_gqa` wins eager (+2.4/+3.2%) but loses under compile
(−1.6%, measured twice) — inductor codegens the compiled `repeat_kv` better than
the native-GQA SDPA call. The winners' stack is therefore **V1 only** (V2 reverted
with rationale in `aa58bf1f`).

**The general lesson: per-variant deltas do not add up, so a stack must be
re-measured, not summed.** Interactions here were the rule rather than the
exception — compile largely *subsumes* V4 (RoPE casts) and changes what V2 and V3
are worth, V5 competes with FSDP2 rather than composing with it, and V2's sign
flips outright. The stack was therefore built by applying winners in descending
measured-delta order and re-measuring after each addition, so the stack is also an
ordered attribution. Acceptance target carried from the gap audit was **h16 ≥
35–40% MFU**; the eventual single-GPU default cleared it at h16 ≈ 43.7%.

### 3.5 nanoGPT-parity combo: V1+V3+V5+V6 together

Question: switch on *everything nanoGPT's code does* at once — does our
implementation match it? This is not the winners' stack (V5 lost in eager); it is
the "their exact recipe, our model" configuration. Branch `mfu-abl/nanogpt-parity`
= `mfu-abl/compile` + cherry-picks of `ba7a274e` (V3), `1450907d` (V5), `596d7cf4`
(V6). V2/V4 stay out (not nanoGPT features).

| run | tokens/s | MFU | vs stack (V1 only) | wandb |
|---|---:|---:|---:|---|
| parity h16 (16×32) | 149,474 | **33.6%** | +0.7pt (stack: 32.9%) | [z5klwygs](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/z5klwygs) |
| parity h20 (8×64) | 87,001 | 34.9% | −1.1pt — micro-batch confound, see next row | [754pxldo](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/754pxldo) |
| parity h20 (16×32) | 90,697 | **36.4%** | +0.4pt (stack: 36.0%) | [abzl2438](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/abzl2438) |

Measured 2026-10-31, jobs 31677/31678/31680.

1. **Compile flips V5's sign.** Plain module + autocast(bf16), −4.9/−7.2% in eager,
   is +0.4–0.7pt *over* the FSDP2 stack once blocks are compiled — the eager FSDP2
   advantage came from its pre-cast bf16 compute path, which inductor now provides
   anyway. Compile also cuts activation memory enough that h20 fits micro 16 on the
   plain path (eager V5 OOMed there); that is where the 8×64 row's −1.1pt comes
   from — halved micro, not the recipe.
2. **V3+V5+V6 together are worth ~+0.5pt on top of compile** — consistent sign at
   both sizes but below the 3% noise bar. Compile remains the only change that
   matters.
3. **At matched workload the honest gap was 12.1pt / 10.6pt**: nanoGPT run at *our*
   shape (same non-embedding params, seq 2048, 1.05M tokens/step, same H100) gets
   45.7% and 47.0% — bigger GEMMs and longer seq *raise* its MFU. Tokens/s at
   matched shape: h16-size 149,474 (ours) vs 194,600 (theirs) = **1.30×**;
   h20-size 90,697 vs 113,150 = **1.25×** (jobs 31677/31680 ours, 31871/31872
   theirs). An earlier reading — "nearly matched at h20, the rest is workload" —
   compared *different* workloads and was too kind to us; the size-matched runs
   corrected it.

---

## Part 4 — Closing the gap: where the time actually goes (2026-11-01 → 11-02)

With every trainer-level code difference replicated, four candidates remained for
the 10.6–12.1pt: compile scope, vocab padding, architecture, trainer-loop residue.
They were resolved by measurement, not argument — and **the prediction ranking
turned out to be wrong**, which is the main methodological lesson of this part.

### 4.1 Matched per-section timers, h20

The same `LoopTimers` class — **CUDA-event pairs for GPU sections, `perf_counter`
for the host batch wait, one synchronize per step** — was placed at identical
points in both training loops: ours behind `trainer.profile_timers=true` (branch
`mfu-abl/profile-timers`, numerics no-op verified) and nanoGPT's `train.py`
(`--profile_timers=True`, patched entropy clone). Both h20-size but **different
architectures** — ours is the parity-combo h20 (GQA + SiLU-gated FF), theirs the
GPT-2-shape h20-eq (MHA + GELU MLP; see Conventions) — seq 2048, 16×32,
same H100; means over the measurement window, ms/step (jobs 31877 ours / 31878
theirs):

| section | ours (parity h20) | nanoGPT (h20-eq) | ratio ours/theirs |
|---|---:|---:|---:|
| forward (32 micros) | 3,641 | 2,926 | **1.24×** |
| backward (32 micros) | 7,880 | 6,330 | **1.24×** |
| grad clip | 2.8 | 2.6 | 1.1× |
| optimizer step | 8.3 | 6.6 | 1.3× (both ≈0) |
| data: host wait | 0.2 | 56.9¹ | — (both overlapped) |
| data: H2D stream time | 0.5 | 0.7 | — |
| **step total (wall)** | **11,536** | **~9,250** | **1.25×** |

¹ their `get_batch` does memmap read + stack + pin on the host inside the micro
loop (~1.8 ms per micro); it overlaps GPU work and does not reach the step total.
**Timers added no measurable overhead on either side** (their step time with
timers, 9,250 ms, matches the untimed run's 9,267 ms) — the methodological check
that makes these numbers usable.

**Reading**: the entire 25% wall-clock gap lived inside the model's forward and
backward kernels, in *exactly equal proportion* (1.244× fwd, 1.245× bwd; fwd:bwd
shape identical at 2.16 on both sides). Everything around the model — data path,
clipping, optimizer, loop Python, logging — measured at parity (≈12 ms of
~11,500). **Trainer-level suspects closed; all remaining headroom was inside the
model graph.**

### 4.2 Kernel-level attribution

Same protocol one level deeper: `torch.profiler` (CUDA activity) wrapped around
**steps 12–13** in both frameworks (`trainer.kernel_profile=true`;
`--kernel_profile=True` in the patched nanoGPT), every kernel dumped with
self-time and launch count, aggregated into the same categories by `kprof_agg.py`.
h20-size, 16×32, same H100, ms/step. Columns: jobs 31882 / 31905 / 32023 / 31889.
The three "ours" columns are our h20 (GQA + gated FF) on the **plain-module +
autocast-bf16 path (no FSDP2)** — the first column is Part 3.5's parity config,
the next two widen only the compile scope; the nanoGPT column is its
GPT-2-shape h20-eq (MHA + GELU) — per-category gaps therefore mix framework and
architecture effects until Part 4.4 separates them, and nothing in this table
measures the FSDP2 path.

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

**Measured verdict — the data named the weakness exactly:**

1. **Our hardware-efficient work is *faster* than nanoGPT's.** GEMMs 4,699 vs
   4,899 ms and flash attention 1,863 vs 2,375 ms — GQA and our GEMM shapes execute
   fine; together **713 ms/step ahead** on the work that counts. GEMM efficiency,
   gated-FF shape, RoPE-inside-compiled-regions and the padded-vocab head GEMM all
   **exonerated**.
2. **The entire gap was the eager glue between compiled regions**: +2,770 ms of
   unfused eager elementwise (fp32 `Mul`/`Add` at ~27k launches/step — pre-norm
   scale/rsqrt muls, residual adds, autocast boundary casts), +438 ms eager RMSNorm
   pieces, +230 ms eager cross-entropy, +147 ms cat/copies ≈ **+3.6 s of glue** vs
   their triton surplus that already contains all of it (their layer norms,
   residual adds and CE are fused rows inside their compiled graph). Note the
   dtype: **the glue ran in fp32** (norms and the residual stream outside compiled
   regions), doubling its memory traffic vs bf16.
3. Both streams saturated (idle: 82 ms ours, 12 ms theirs, ~0.7%) — not
   launch-latency stalling, real memory-bound kernel time.

### 4.3 The lever: whole-model compile (branch `mfu-abl/whole-compile`, job 31905)

Residual's metric logging moved into a `@torch.compiler.disable` helper so the
per-block `log_name` string never enters tracing (per-block wandb metrics
unchanged); new `compile_model=true` knob compiles the whole model. Refactor
bit-identical with the knob off; compiled losses track eager to ~1e-7.

Result at h20: step 11,536 → **9,451 ms** (−18%), tokens/s 90,697 → **110,890**
(+22%), MFU 36.4% → **44.4%** — from 25% behind nanoGPT to 2.2%. The glue
collapsed exactly as the ceiling predicted: eager elementwise 2,878→625, eager
norms 438→0.2, cat/copies 147→0.2, absorbed into fused triton (1,157→1,737, still
smaller than theirs). Peak memory *dropped* 66.9 → 60.5 GiB at h20 micro 16.

### 4.4 Architecture ablation: GPT-2's shape inside our framework (2026-11-02)

Maciej's question: the shape differences are GELU-vs-SwiGLU, MHA-vs-GQA,
25-vs-20 layers — so run *their* shape in *our* framework. Branch
`mfu-abl/nanogpt-arch` adds `GeluFeedForward` (2-matrix GELU MLP); MHA and depth
are config overrides (`common.kv_heads=q_heads`, `common.n_blocks=25`). Same 491M
block params; the trainer's FLOPs accounting lands at **4.122 GF/token — identical
to nanoGPT's** (our vocab was already 50304-padded, so that difference never
existed). h20 workload, compiled loss, 16×32, job 32064:

| config (same 491M, same H100) | step ms | tokens/s | MFU |
|---|---:|---:|---:|
| **our arch, our framework** (compiled-loss) | **9,046** | **115,862** | 46.4% |
| GPT-2 arch, our framework | 9,578 | 109,425 | 45.6% |
| GPT-2 arch, nanoGPT framework | 9,250 | 113,150 | 47.0% |

Kernel profile at **matched architecture** — same categories, aggregation and
format as the Part 4.2 table (steps 12–13, [scripts/kprof_agg.py](../../scripts/kprof_agg.py)),
ms/step. Columns: jobs 32023 / 32064 / 31889. The first column repeats Part
4.2's compiled-loss column so the architecture comparison (col 1 vs 2, same
framework) and the framework comparison (col 2 vs 3, same architecture) share
one table. Re-aggregated 2026-11-03 from the saved job logs (`~/mfu_32064.log`,
`~/nanogpt_ref_31889.log` on entropy); the nanoGPT column reproduces Part 4.2's
to the digit, validating the re-aggregation.

| kernel category | ours (our arch, compiled loss) | **GPT-2 arch, our framework** | **GPT-2 arch, nanoGPT** |
|---|---:|---:|---:|
| GEMMs (nvjet) | 4,801 | 4,711 | 4,899 |
| flash attention | 1,961 | 2,418 | 2,375 |
| compiled (triton: fused glue, act., norms, CE…) | 1,871 | 1,982 | 1,841 |
| eager elementwise | 291 | 345 | 108 |
| eager norm pieces (pow/reduce/rsqrt) | 0 | 0 | 0 |
| softmax/CE (eager) | 0 | 0 | 0 |
| cat / copies / memcpy | 0.2 | 0.2 | 0.4 |
| optimizer | 10 | 10 | 9 |
| other | 8 | 9 | 6 |
| **TOTAL busy** | **8,943** | **9,475** | **9,238** |
| **step total (wall)** | **9,046** | **9,578** | **9,250** |
| kernel launches per step | 60,377 | 66,437 | 40,934 |
| tokens/s | 115,862 | 109,425 | 113,150 |
| MFU (own FLOPs basis: 3.97 / 4.122 / 4.122 GF/tok) | 46.4% | 45.6% | 47.0% |

Reading the matched-architecture pair (columns 2 vs 3): GEMMs 4,711 vs 4,899 —
**ours faster at identical shapes**; flash attention 2,418 vs 2,375 (par —
confirming the earlier flash advantage was GQA, now gone); triton 1,982 vs 1,841
(+141: RoPE's rotation work, which we kept — their learned positions are a free
add); eager elementwise 345 vs 108 (+237: grad-accumulation granularity, 25 layers
of many small tensors). Peak memory 52.0 GiB.

1. **Our architecture is genuinely faster than GPT-2's at equal parameters in the
   same framework**: +5.9% tokens/s (115,862 vs 109,425). GQA + fewer, fatter
   gated-FF layers beat MHA + more, thinner GELU layers on wall-clock — while
   scoring *lower* on MFU (46.4 vs 47.0), the bookkeeping artifact being that the
   GPT-2 shape needs ~4% more FLOPs per token.
2. **Framework-vs-framework at matched architecture: ours is +328 ms (3.5%)
   behind**, fully accounted by what we deliberately kept (RoPE, +141 triton) and
   accumulation granularity (+237), offset by faster GEMMs (−189).
3. The comparison is closed in all three directions: code (parity + compile scope),
   workload (size-matched), architecture (this run).

### 4.5 Variance study — how much of a difference is real

Jobs 32094–32098 (ours) / 32099–32103 (nanoGPT): each side re-run **5× with the
job-32064 / job-31889 commands verbatim** → 6 measurements per framework, all 12
on the same node (h100a). **Uniform reducer: mean wall/step over steps 15–29**
(post-warmup, post-profiler), applied identically to both sides.

| | wall/step | tokens/s | MFU |
|---|---:|---:|---:|
| ours (GPT-2 arch clone) | 9,627 ± 55 ms (cv 0.58%) | 108.9k ± 0.6k | 45.40 ± 0.26% |
| nanoGPT | 9,277 ± 64 ms (cv 0.68%) | 113.0k ± 0.8k | 47.11 ± 0.32% |

Framework gap at matched architecture: **+350 ± 34 ms (3.8%, ~10σ) — real**, and
consistent with the kernel attribution (+237 grad-accum granularity, +141 RoPE,
−189 GEMMs). The uniform window reads ~0.5% slower than the single-run headline
numbers (9,578 / 9,250, the trainer's own window) — within-run drift, applied
equally to both sides. **Takeaway for future comparisons: single runs carry
~0.6% cv, so differences below ~2% need repeats, and the reduction window must be
identical on both sides or it shifts results by ~0.5%.**

---

## Part 5 — Speedup plan and implementation (2026-11-02 → 11-03)

Starting point: whole-model compile at **44.4%** MFU (9,451 ms/step) vs nanoGPT
**47.0%** (9,250 ms). Our GEMMs and attention were *faster* (−537 ms combined);
the remaining +201 ms was eager cross-entropy outside the compiled graph (230 ms),
its bf16↔fp32 boundary casts over the logits (~270 ms, inside the elementwise
row), and grad-accumulation adds running ~4× off roofline (291 vs their 76 ms).

**1. Adopt what is proven.** Merge `mfu-abl/whole-compile` into `pretraining`: V1
compile knobs, V3 fused AdamW + foreach clipping, V6 async H2D + prefetch, V5
autocast knob, the Residual `log_name` refactor, and the whole-model compile knob.

**2. Compile forward+loss as one graph (~+5%).** nanoGPT computes CE inside its
compiled model; we computed it eager in the trainer, so the 16×2048×50257 logits
(≈6.6 GB fp32) crossed the graph boundary every micro-step and paid twice. Fix:
wrap the trainer's forward+CE computation in `torch.compile` so model and loss are
one graph — no model-API change. Watch for: graph breaks at the autocast context,
the backward staying fused, the eval path triggering one extra compile.

**3. Raise the micro-batch at h20 (~+2–3%).** Compile dropped peak memory to
60.5 GiB at micro 16, so micro 32 × ga 16 plausibly fits.

### 5.1 Execution record (2026-11-02 → 11-03): all three items done

1. **Merged** — `mfu-abl/whole-compile` → `pretraining` (`34bf3c17`); gates green,
   knobs opt-in, FSDP2 configs keep `compile_blocks`.
2. **Compiled loss** — branch `mfu-abl/compiled-loss` (`733f455`), merged to
   `pretraining` on 2026-11-03 (`438d55c5`). h20 step 9,451 → **9,046 ms**,
   115,862 tok/s, **46.4% MFU — 2.2% faster wall-clock than nanoGPT's 9,250 ms**
   at matched workload (job 32023). The CE row and its boundary casts vanished
   into triton exactly as attributed, leaving only grad-accumulation adds (291 ms)
   eager. Numerics gate: per-step losses vs job 31905 track at max |Δ| 0.0017 over
   30 steps.
3. **Micro 32** — OOMs at h20 (73.6 GiB used + 6.1 GiB more requested, job 32024);
   **16×32 stays**. Recorded, not retried.

### 5.2 Numerics overlay — the question that had to be closed

Jobs 32025/32031: 300 steps at h16, compiled (`compile_loss`) vs eager twin, same
seed, loss 11.2 → 5.17. **Mean |Δloss| 0.0074, max 0.066** (step 148, the steep
mid-run region), last-50 mean 0.0036 (0.07% of final CE), **no divergence trend** —
benign bf16 fusion-order noise. Numerics question closed; the compiled single-GPU
default is safe to adopt.

Incidental datum from the overlay: compiled-loss at h16 runs ~5.4 s/step ≈ 194k
tok/s ≈ **43.7% MFU** (vs 33.6% parity) — the only h16 number for the final stack.

---

## Part 6 — Final results

### 6.1 The whole journey in one table (MFU, method × model)

| method | h12 | h16 | h20 |
|---|---:|---:|---:|
| **B0 baseline** (eager, FSDP2 bf16) | 25.2% | 28.1% | 30.8% |
| V1 torch.compile (per block) | — | 33.2% | 36.3% |
| V2 enable_gqa (native-GQA SDPA) | — | 28.8% | 31.8% |
| V3 fused AdamW + foreach clip | — | 28.1% | 30.8% |
| V4 RoPE cast hygiene | — | 27.8% | 30.9% |
| V5 plain module + autocast bf16 | — | 26.7% | 28.6%¹ |
| V6 async H2D + prefetch | — | 27.9% | 30.9% |
| V6c baseline + free input data | — | 28.2% | — |
| **stack (= compile only)** | **29.8%** | **32.9%** | **36.0%** |
| **parity combo V1+V3+V5+V6** (nanoGPT's full recipe) | — | 33.6% | 36.4% |
| **whole-model compile** (+ `log_name` refactor) | — | — | **44.4%** |
| **compiled loss** (one graph incl. CE) | — | 43.7%³ | **46.4%** |
| GPT-2 arch clone in our framework (MHA+GELU+25L) | — | — | 45.6% |
| nanoGPT reference (its defaults: 124M, seq 1024) | — | 37.9%² | — |
| **nanoGPT size-matched** (same non-emb params, seq 2048) | — | **45.7%**² | **47.0%**² |

¹ micro 8 (micro 16 OOMs on the plain path). ² nanoGPT's GPT-2 architecture
(MHA + GELU, learned positions), size-matched on non-embedding params — not our
attention/FF methods; on the same H100, converted to the 989e12 basis. ³ from
the numerics-overlay run, not a dedicated measurement.
Variants ran h16+h20 only (Maciej's trim); h12 has the baseline and the stack.
Only V1 clears the noise bar upward; V2 clears it in eager but reverses under
compile; V5 is clearly negative; V3/V4/V6/V6c are noise.

### 6.2 Where we landed

**h20, standard workload, one H100: 9,046 ms/step, 115,862 tok/s, 46.4% MFU —
2.2% faster wall-clock than nanoGPT's 9,250 ms at matched workload.** From the
30.8% eager baseline that is **+51% tokens/s**. nanoGPT's MFU still prints higher
(47.0%) only because its FLOPs/token numerator is ~4% larger (MHA + 2-GEMM MLP);
on wall clock, the number that pays for GPU hours, we are ahead.

Decomposition of the final state against nanoGPT, from the kernel profile: we are
ahead on GEMMs (−98 ms) and flash attention (−414 ms), behind on eager
elementwise (+183 ms, almost all fp32 grad-accumulation adds). Everything else is
at parity. The flash-attention advantage is architecture (our GQA vs its MHA),
not framework — at matched architecture flash is par (Part 4.4); the GEMM
advantage survives at identical shapes.

### 6.3 What is in `pretraining` today

The knobs below are declared in `src/definitions.py` (`TrainerSchema`) as of the
typed-config rewire (`b973c0bf`), so they take **plain `trainer.X=` overrides**;
the `+trainer.X=` spelling used throughout this document's historical records was
correct when those runs were made but no longer composes.

- Single-GPU pretraining default: `trainer.distributed=null
  trainer.autocast_bf16=true +compile_model=true`
  (`compile_model` is a top-level key, so it keeps its `+`).
- For the measured-best single-GPU configuration add `trainer.compile_loss=true`.
- **FSDP2 multi-GPU runs keep `+compile_blocks=true`** — regional compile is the
  pairing FSDP2 documents; whole-model compile under FSDP2 is **untested** (Part 8).
- Measurement knobs: `trainer.profile_timers=true` (section timers),
  `trainer.kernel_profile=true` (per-kernel dump at steps 12–13).

### 6.4 Exonerated — measured innocent, do not re-litigate without new evidence

FSDP2 at world size 1 (beats plain bf16 in eager; loses by ~0.5pt under compile);
the tokenization/collate/copy pipeline (V6c: random tokens ≈ +0.3% — the whole
input pipeline costs ≈0 at these sizes); fused-vs-unfused optimizer and clipping;
RoPE cast hygiene; vocab padding (our vocab was already 50304); the GEMM shapes,
the gated FF, and GQA (all faster than nanoGPT's equivalents); the per-layer
Residual norm instrumentation; `F.cross_entropy` usage; and nanoGPT's *absolute*
superiority at its own defaults (37.9% at 124M/seq1024 is **below** our 36.4% h20
— small models underfill the H100).

### 6.5 Parked: the diminishing-returns shelf

- **Liger-style fused linear+CE** — never materializes the logits at all; beats
  nanoGPT's design and saves memory, but adds a kernel dependency.
- **Flat/bucketed grad accumulation** (DDP-style contiguous grad buffer) — worth
  ~150–200 ms, the largest remaining single-GPU item; real plumbing.
- **Activation checkpointing** — purely a memory knob to buy larger micro-batches
  at h24+, not an MFU fix at fixed micro-batch.
- **Micro-batch 32 at h20** — OOMs today (job 32024); revisit if memory drops.
- Changes deliberately excluded because they alter parameters or dynamics: fused
  QKV projection, no weight decay on 1-D params, weight tying.
- **Pre-tokenized memmap dataset** — stays parked until multi-GPU work needs it
  (V6c proved the pipeline costs ~0 at these sizes on one GPU; 2 workers will not
  feed a multi-GPU run).

---

## Part 7 — Reproducing a measurement (the protocol, condensed)

Everything above was produced this way; a new measurement is comparable only if it
follows the same rules.

1. **Workload**: `configs/mfu_audit/{h12,h16,h20}.yaml` unchanged — seq 2048,
   batch 512, 30 steps, window steps 10–29, wandb project `nano-mfu-audit`. Invoke
   as `--config-path configs/mfu_audit --config-name h20` (the repo convention;
   `--config-name mfu_audit/h20` nests the tree under the directory name and does
   not work).
2. **One change per branch**, `mfu-abl/<name>` in its own worktree, pushed
   (entropy pulls on the login node; compute nodes have no GitHub access).
3. **Micro-batch**: from an OOM probe — a 5-step run doubling the per-forward
   micro-batch until it fails, then backing off one notch. If a change moves the
   memory ceiling, report **both** the like-for-like 16×32 number and the best-fit
   number; a changed micro-batch is a confound, not a result (see the parity h20
   8×64 row in Part 3.5).
4. **Noise bar**: step-time cv is 0.6–1.6%, so **a delta under ~3% in tokens/s
   needs a repeat run before it is believed**, in either direction. For
   framework-level comparisons, 5–6 repeats per side give ±0.3% on the mean.
5. **Reduction window must be identical on both sides** of any comparison — the
   trainer's own window and a uniform steps-15–29 reducer differ by ~0.5%.
6. **Numerics gate**: a change claiming to be a no-op proves it with a seeded CPU
   run that is bit-identical; a change that fuses kernels (compile) instead runs a
   loss overlay against its eager twin, same seed, and reports mean/max |Δloss|
   over a few hundred steps (Part 5.2 is the template).
7. **Profiling**: `trainer.profile_timers=true` for section timers (CUDA-event
   pairs; verify they add no measurable overhead by comparing against an untimed
   run), `trainer.kernel_profile=true` for the per-kernel dump at steps 12–13,
   aggregated by category. Watch for **silent recompiles** — check for dynamo
   recompile warnings whenever compile scope changes.
8. **Reading a run**: `scripts/mfu_window.py` turns a wandb run into a table row
   plus pass/fail verdicts. FLOPs/token come from the live model via
   `src/core/mfu.py`, not from the config.

---

## Part 8 — Baseline for multi-GPU work

**Every number in this document was measured on one GPU.** No multi-GPU
throughput number exists for this codebase. Two code findings below were verified
while writing this note (2026-11-03, at `b973c0bf`); they are **recorded here, not
fixed** — multi-GPU work belongs to a separate effort.

### 8.1 Two code findings that gate any multi-GPU measurement

1. **MFU will be overreported by ×world_size.** `_update_processed_tokens`
   ([src/core/trainer.py:587](../../src/core/trainer.py)) counts *global* tokens
   (`batch.numel() * WORLD_SIZE`), while the MFU line
   ([trainer.py:634](../../src/core/trainer.py)) divides throughput by **one**
   GPU's peak (`peak_flops_bf16`, [src/core/mfu.py](../../src/core/mfu.py)).
   Latent so far because world_size has always been 1. **Fix this before trusting
   any multi-GPU MFU figure** — otherwise a 4-GPU run reports ~4× the true
   utilization.
2. **Gradient sync fires on every micro-step.** `calculate_loss` calls
   `loss.backward()` per micro-chunk ([trainer.py:508](../../src/core/trainer.py))
   and nothing in the trainer or `distributed_training.py` calls
   `set_requires_gradient_sync(False)` / `no_sync` between chunks. Under FSDP2 at
   ga 16–64 that is 16–64× the necessary reduce-scatters per step. This is the top
   multi-GPU throughput suspect, and it is invisible in every measurement above.

### 8.2 What carries over, and what does not

Carries over: the FLOPs/token convention and `src/core/mfu.py`; the workload and
window; the profiling toolchain (both knobs work regardless of world size); the
noise bar and repeat discipline; the finding that the input pipeline costs ≈0 —
**though with the caveat that 2 dataloader workers will not feed a multi-GPU run**.

Does **not** carry over, and must be re-measured:

- **Whole-model compile and `compile_loss` under FSDP2 are untested.** Single-GPU
  runs use `compile_model`; FSDP2 configs still use regional `compile_blocks`
  because that is the documented pairing. Whether one graph survives FSDP2's
  pre-forward all-gathers is an open experiment.
- **V5's verdict is world-size-1 only.** "FSDP2 beats a plain module" is a
  statement about one GPU with no communication to hide behind; at world size > 1
  FSDP2 is not optional and the comparison is meaningless.
- **`reshard_after_forward` default** has never been tuned.
- **Grad-accumulation granularity** (+237 ms at 25 layers, already the largest
  eager item on one GPU) interacts directly with sync batching — item 1 above.
- **Checkpointing**: a real multi-GPU dcp round-trip has never been run, and
  resume does not record `world_size`, so resuming on a different GPU count
  silently changes the data each rank sees (the dataloader shards per rank — a run
  is reproducible at fixed (seed, world_size), not across world sizes).

### 8.3 Suggested order for whoever picks this up

MFU accounting fix → a clean 4×H100 h20 baseline with timers and kernel profile →
batched gradient sync plus a numerics overlay → the compile-scope verdict under
FSDP2 → reshard and checkpoint hardening. New trainer knobs are declared in
`src/definitions.py` first (the typed-config policy), then used as plain
`trainer.X=` overrides.
