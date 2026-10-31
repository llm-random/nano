# Backbone audit: everything runs, and this is how fast (stage 3 results)

**Status: complete, 2026-10-30.** Plan: [pretraining_stage3_mfu_plan.md](pretraining_stage3_mfu_plan.md)
(amendments at its end); every timestamped execution detail is in
[pretraining_reset_log.md](pretraining_reset_log.md). All runs: one NVIDIA H100 80GB
HBM3 on entropy (`h100a`), bf16 compute via FSDP2's MixedPrecisionPolicy, sequence
length 2048, **512 sequences per step** (1,048,064 model-consumed tokens), 30 steps,
MFU window = steps 10–29. wandb project:
[ideas_cv/nano-mfu-audit](https://wandb.ai/ideas_cv/nano-mfu-audit).

## The MFU table

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

Conventions: FLOPs/token = `6·P + 12·s·datt·L` measured from the live model (PaLM,
no causal discount); peak = 989 TFLOP/s dense bf16; micro-batches from per-model OOM
probes (largest that fits). MFU grows with model size as utilization should —
**no efficiency anomaly found**; the 20%-on-H100 investigation tripwire never fired.
Readings: mqa buys h16 +8% tokens/s; ~28–31% without torch.compile or activation
checkpointing is a healthy baseline, and both are the obvious future levers.
The fp32 controls (`trainer.distributed=null`, plain module) are **5.7–6.9×
slower** than the FSDP2-bf16 production path, achieving a near-constant
**43.9/45.8/44.2 TFLOP/s at h12/h16/h20**. *Starred MFU is against the bf16 peak
for column comparability; the proper bases for that ~45 TFLOP/s are **~9% of the
plan's tf32 tensor-core peak (495e12)** — proving TF32 is off by default
(PyTorch ships `allow_tf32=False`) — and **66–68% of the FP32 vector peak
(67e12)**: the plain path is compute-bound at fp32-vector rates, uniformly across
sizes. The tf32 cells (same fp32 storage, tensor-core matmuls via the
torch.backends flags) sit cleanly between: at h16 the hierarchy is
**bf16 : tf32 : fp32 = 123.4k : 52.1k : 20.6k tokens/s (6.0 : 2.5 : 1)**, tf32
achieving 100–119 TFLOP/s = 20–24% of its 495e12 peak, with first losses
identical to fp32's (storage unchanged). Verdict of the dtype comparison: bf16
stays the default, now by measurement over the full {fp32, tf32, bf16} ×
{h12, h16, h20} matrix. The fp32/tf32 runs' un-quantized first losses
(11.20–11.22) also pin the CPU-fp32 control (11.2235).

### Ablation follow-up (2026-10-31): the blocker was torch.compile

The six-variant ablation ([mfu_ablation_plan.md](mfu_ablation_plan.md), one
branch per candidate fix, each measured against this table's baselines on the
same H100) decomposed the gap to nanoGPT-class MFU. Before/after, bf16 FSDP2
path, same 30-step protocol:

| model | eager (this audit) | + torch.compile (`mfu-abl/stack`) | gain |
|---|---:|---:|---:|
| h12 | 227,158 tok/s / 25.2% | 269,352 / **29.8%** ([urnucb40](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/urnucb40)) | +18.6% |
| h16 | 123,367 / 27.7% | 147,922 / **33.2%** ([ykipu3dy](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/ykipu3dy)) | +18.3%* |
| h20 | 76,858 / 30.8% | 90,538 / **36.3%** ([24uvodtr](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/24uvodtr)) | +17.8% |

*vs the fresh B0 re-run (125,052 / 28.1%, [7ms4eafl](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/7ms4eafl)).
Reference: nanoGPT itself (124M, seq 1024, compiled) measures **37.9%** on this
same H100 (its printed "120%" is on the A100 basis) — the remaining gap is
1–5 points at 3–5× its model size and 2× its sequence length.

Everything else measured innocent or negative: fused AdamW + foreach clipping
+0.1/+0.0% (noise); RoPE cast hygiene −0.9/+0.3% (noise); async H2D copies
−0.6/+0.3% (noise); a run on instantly-generated random tokens +0.3% — the
whole input pipeline, on-the-fly NeoX tokenization included, costs ~nothing at
these sizes. The nanoGPT-style plain-module+autocast path is **slower** than
FSDP2-bf16 at world size 1 (−4.9% h16, −7.2% h20 where micro 16 OOMs), so
FSDP2 is exonerated too. enable_gqa (native-GQA SDPA) is +2.4/+3.2% in eager
but reproducibly −1.6% under compile, so it stayed out of the stack. One bug
found and fixed on the way: block-level compile guards on Residual's per-block
log_name string, hits dynamo's recompile_limit and silently drops half the
blocks to eager — the knob compiles RoPEAttention+GatedFeedForward instead
(`8e2d5f6b`).

### The E7 ordering tripwire, resolved with evidence

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

## Correctness results (the "does everything still work" half)

| check | result |
|---|---|
| Full test suite on entropy's cluster env | **172 passed, 0 failed** (first run since the reset) |
| Fineweb at seq 2048 through the real configs | batches (·, 2048), all ids < 50304 |
| NeoX tokenizer offline on compute nodes | staged into `/storage_nvme_4/nano/hf_cache`, len 50277 |
| First loss of a fresh model | 11.19–11.25 ≈ CPU-fp32 control 11.2235 (≈ln V + 0.4 at seq 2048; GPU is one bf16 ulp away) |
| Token accounting | exact: 30 × 1,048,064 on every run |
| Periodic eval + final_ce at seq 2048 (E5) | fired at steps 10/20 (8.125→7.25); final_ce over **exactly** 8192 holdout samples: 7.107 ([7y2zup9q](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/7y2zup9q)) |
| Resume through a dcp checkpoint on GPU (E6) | **bit-identical**: max loss delta 0.0 over 14 overlapping steps — measured on h12 first (reasoned deviation) and then **re-run on the plan's h16 with the same result** (job 31210) |
| Pattern-mixed stack under a real optimizer | gqa+mqa every_2nd trains; loss 11.25→7.19 |

## Bugs found and fixed (each with a regression test)

1. **Hardcoded rendezvous ports** — 18 tests failed on the shared node, zero
   locally: fixture/subprocess tests pinned MASTER_PORT and leaked it via
   `setdefault`. Now `conftest.free_port()` asks the OS. (`a83d48f8`)
2. **`int(os.environ.get("WORLD_SIZE"))` → `int(None)`** in `AbstractDataset` —
   a TypeError naming neither variable nor fix for any dataloader built outside
   main.py. Defaults to 1/0. (`393d200`)
3. **Same bug one layer up** — `get_dataloader` used `os.environ[...]`, bare
   KeyError. (`855b00c`)
4. **Throughput stub test probed CUDA on GPU nodes** — my own new test; found by
   the first entropy suite run. (`ff673c4e`)
5. **Checkpoint-artifact test assumed the CPU shape** — a CUDA node writes the
   world-size-1 model as a dcp shard (`__0_0.distcp`), which resumes perfectly;
   the test now accepts both forms. (`393d2000`)
6. **First-loss criterion mis-calibrated** — fresh-model CE at seq 2048 is
   ~ln(V)+0.4 on any device (proved by CPU control); band recalibrated with the
   calibration written down. (`ad85c17e`)

## Measurement infrastructure added

`src/core/mfu.py` (measured FLOPs from the live model, bf16 peak table,
None-not-wrong for unknown GPUs), per-step `train/step_time` / `tokens_per_s` /
`mfu` / peak-memory logging (numerics-no-op, gated bit-identical), the first-loss
sanity in the e2e tests, and `scripts/mfu_window.py` (wandb run → this table's row
+ pass/fail verdicts).

## Parked for Maciej

1. **Activation checkpointing** — none exists; h4 at micro-64 already costs 45 GiB
   of activations. An AC knob would multiply feasible micro-batches and likely push
   MFU up at h16+. Biggest single efficiency lever found.
2. **torch.compile** — untried; the other standard lever.
3. **Token-count convention** — `processed_tokens` counts model-consumed tokens
   (512×2047); decide if the spec should count raw 512×2048 instead.
4. **World-size-1 checkpoints are dcp shards** — fine for resume (proven), awkward
   for export; decide whether 1-GPU runs should write a plain file.
5. **wandb project name** `nano-mfu-audit` — rename if wanted.
