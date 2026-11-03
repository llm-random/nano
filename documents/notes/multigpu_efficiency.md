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

## 4. Baselines — status

Three runs, same model, same workload, same commit; only the GPU count changes.
Configuration: FSDP2 (`trainer.distributed` from `configs/_trainer/llama.yaml`,
sharding embedding / block / head), regional compile (`+compile_blocks=true`),
no autocast knob (FSDP2's `MixedPrecisionPolicy` already computes in bf16 and
reduces in fp32), `trainer.profile_timers=true` and
`trainer.kernel_profile=true` on.

| # | world size | gres | ga | status | step ms | tokens/s | MFU | peak mem |
|---|---:|---|---:|---|---:|---:|---:|---:|
| B1 | 1 | `gpu:h100:1` | 32 | pending | | | | |
| B2 | 2 | `gpu:h100:2` | 16 | pending | | | | |
| B4 | 4 | `gpu:h100:4` | 8 | pending | | | | |

B1 is not redundant with the 45.6% reference: that run had no FSDP2 and used
whole-model compile plus a compiled loss. B1 is the *same-commit, same-node*
FSDP2 + `compile_blocks` anchor, so the 1→2→4 scaling curve measures
communication rather than two weeks of code drift.

## 5. Results

_(filled as runs land)_

## 6. Log

- **2026-11-03** — scope set; worktree `multigpu-mfu` created off `pretraining`.
  MFU accounting fixed (§3) with a world-size-parametrized regression test.
  `GeluFeedForward` cherry-picked from `mfu-abl/nanogpt-arch` (`9f3e7421`) — it
  had never been merged, so the GPT-2 architecture was not buildable on
  `pretraining`. Entropy checked: h100a is one node with 8×H100, our QoS cap is
  8 GPUs, so 2- and 4-GPU jobs are allocatable on a single node (multi-node is
  out of scope for now).
