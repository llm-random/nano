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
| h16 mha | 358.8M | 2.27 | 16×32 | 123,950 | 29.1% | 8.456s (0.1%) | — | [2f6nfosg](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/2f6nfosg) |
| h16 mqa | 350.9M | 2.17 | 16×32 | 133,322 | 28.8% | 7.863s (1.5%) | — | [2fp8a1w7](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/2fp8a1w7) |
| h16 gqa+mqa every_2nd | — | — | 16×32 | 127,991 | 28.2% | 8.191s (1.6%) | — | [alj69rdy](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/alj69rdy) |

Conventions: FLOPs/token = `6·P + 12·s·datt·L` measured from the live model (PaLM,
no causal discount); peak = 989 TFLOP/s dense bf16; micro-batches from per-model OOM
probes (largest that fits). MFU grows with model size as utilization should —
**no efficiency anomaly found**; the 20%-on-H100 investigation tripwire never fired.
Readings: mqa buys h16 +8% tokens/s; gqa vs mha is wall-clock identical here (kv
width is ~0.3% of FLOPs at this size — gqa's win is kv-cache memory at inference,
not training speed). ~28–31% without torch.compile or activation checkpointing is a
healthy baseline; both are the obvious future levers.

## Correctness results (the "does everything still work" half)

| check | result |
|---|---|
| Full test suite on entropy's cluster env | **172 passed, 0 failed** (first run since the reset) |
| Fineweb at seq 2048 through the real configs | batches (·, 2048), all ids < 50304 |
| NeoX tokenizer offline on compute nodes | staged into `/storage_nvme_4/nano/hf_cache`, len 50277 |
| First loss of a fresh model | 11.19–11.25 ≈ CPU-fp32 control 11.2235 (≈ln V + 0.4 at seq 2048; GPU is one bf16 ulp away) |
| Token accounting | exact: 30 × 1,048,064 on every run |
| Periodic eval + final_ce at seq 2048 (E5) | fired at steps 10/20 (8.125→7.25); final_ce over **exactly** 8192 holdout samples: 7.107 ([7y2zup9q](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/7y2zup9q)) |
| Resume through a dcp checkpoint on GPU (E6) | **bit-identical**: max loss delta 0.0 over 14 overlapping steps |
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
   the first entropy suite run. (`a83d48f8` era)
5. **Checkpoint-artifact test assumed the CPU shape** — a CUDA node writes the
   world-size-1 model as a dcp shard (`__0_0.distcp`), which resumes perfectly;
   the test now accepts both forms. (`393d200`)
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
5. **fp32 comparison runs** were dropped (bf16 via FSDP2 is the only production
   path); say the word if a fp32 baseline matters.
6. **wandb project name** `nano-mfu-audit` — rename if wanted.
