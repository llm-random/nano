# AME training-efficiency plan — three tiers

**Status: historical — closed, and kept for provenance.** All three tiers are done; the
code is committed on `ame_optimize` and `ame_main`. Written 2026-11-04 against HEAD
`f302ae12`. Archived 2026-11-05.

**Read [ame_optimization_report.md](../ame_optimization_report.md) instead** for what was
tried, what it was worth and what is left. It supersedes the whole of this document's
narrative — the tier plans below describe work that is finished.

**What is still only here: §3, the experiment ledger.** It is the only map from a result
back to the run that produced it — **14 wandb run ids and 17 config paths**, plus peak
memory and step time per job, which the report condenses away. If you need to reproduce
or re-examine a number, start there. §4, the decision log, is the other original: the
report's §4 is distilled from it.

- **Tier 1** — B1/B2 measured, O9 measured and **rejected on this test bed**
  (−2.53% on entropy, +3.33% on lem), O31 attribution produced. The attribution
  gate fired **NEGATIVE**: the bank gradient/optimizer path is ~1.3–5% of step
  time, not a top-two term. Tiers 2 and 3 stopped there.
- **Tiers 2 and 3** — the gate was then explicitly overridden and both were
  built as one module, `src/core/memory_parallel.py`, with `mem_parallel_size`
  as the knob. Every §1.4 correctness gate passes, including bit-identity to the
  ordinary path on the CPU toy model and an O8 divisor test with a negative
  control.

**The headline, and it is not the one the fixed test bed alone would give.**
What these tiers are worth scales with `gradient_accumulation_steps`, because
everything they delete is per-micro-step work:

| | `ga` | measured |
|---|---:|---:|
| 1 GPU, micro-batch 4 | 128 | **+38.5%** — pkmf's 1.43× transfers |
| 4 GPUs, micro-batch 4 | 32 | **+22.6%** |
| 4 GPUs, micro-batch 16 *(the pinned test bed)* | 8 | **−1.0%** |

The pinned bed is the one configuration measured in which they do not pay: the
largest micro-batch that fits, on the fastest interconnect, with FSDP2 already
sharding four ways. See "The ga-scaling curve" below §3.

**Companion:** [ame_optimization.md](ame_optimization.md) — the full 31-item survey of
pkmf's static-memory stack. This document is the subset we committed to, in order, with
the experiments that decide each step.

**Target:** AME **h20** on the **entropy** cluster, **4×H100**. Metric is **MFU per GPU**
(GPU-count invariant); tokens/s is reported alongside it.

---

## Goal

Raise AME **h20** training MFU on **4×H100** by executing Tiers 1→3 in order, measuring
every step on the entropy `h100` partition, and keeping the mathematics unchanged
throughout.

- **Tier 1** establishes a trustworthy 4-GPU baseline (B1/B2), lands the gradient-sync
  fix (O9), and produces the per-submodule attribution table. That table is what
  *authorises* Tiers 2 and 3 — if the bank gradient path is not a top-two term after O9,
  re-rank before building anything.
- **Tier 2** removes the 2.576B bank parameters from FSDP2's dense gradient and
  optimizer path (O1–O5).
- **Tier 3** shards those banks vertically across the four GPUs (O6–O8).

**Done** = each tier's exit criteria met, every run recorded in the §3 ledger, and a
measured MFU/GPU delta against B1 reported per tier. A tier that measures *worse* is a
completed tier with a negative result, not a failure — record it and stop, do not tune
past the gate.

**Not in scope:** anything listed in §0 as cut (O10, O11, O16, O17) or deferred
(O14/O15, O19/O20, O21–O27). Do not reintroduce them to hit a number.

### Goal revision — 2026-11-05, from Maciej

Tiers 1–3 are complete and recorded. The goal is now a **number, not a tier list**:

> **Exceed 40% MFU for AME training, ideally matching the dense ceiling of 44.44%.**

The h20 4-GPU configuration stays the reference baseline because it is the one
every number in this document is comparable against. It is no longer the only
shape under test: **h12 and h16 at 1 GPU are in scope**, and the cut list above
is lifted to the extent that Maciej's directive of 2026-11-05 asks for
"different strategies and improvements" — but O10, O16 and O17 stay
unimplemented until he says otherwise, and §4 records them as the flagged
lever rather than a silently taken one.

**O11 (fused-linear-CE) was lifted from the cut list on 2026-11-05** after the
measurements below made the case: it removes a 6.14 GiB allocation with no
recompute of the model and no change to the mathematics. **O10 stays cut** —
it trades compute for memory and nothing has measured whether the micro-batch
gain beats the recompute cost.

**What "matching dense" actually requires, stated as arithmetic rather than as
an aspiration.** B2 is 44.44% at 443,432 tokens/s and 2.364 s/step, and dense
H-balanced h20 is **3.97 GFLOPs/token** against AME's **3.768**. MFU is
`tokens/s × GFLOPs/token ÷ peak`, so AME needs

    443,432 × 3.97 / 3.768 = 467,200 tokens/s  →  **2.244 s/step**

to show the same MFU. **AME has to be 5% faster than dense in wall-clock to
report the same number**, because it buys its parameters with gathers that cost
time and no FLOPs. This is a structural headwind, not a tuning gap, and any
claim of "dense parity" has to clear it.

| target | step (4 GPU, h20) | from today's 2.902 s |
|---|---:|---:|
| 40% | 2.497 s | −405 ms |
| 44.44% (dense parity) | 2.244 s | −658 ms |

**Where this stands as of 2026-11-05:** AME 4-GPU h20 is **34.42%**, which is
**77.4% of the dense ceiling**, up from B1's 60.4%. The remaining gap is 10.0
MFU points, of which the §3 post-Tier-3 ledger can presently account for
roughly 100–170 ms of candidate savings — **not enough**, which is why the plan
below changes shape rather than continuing down the same list.

---

## 0. Scope: what is in, and what we deliberately left out

Everything here raises **MFU** — compute efficiency. Items whose only effect is peak
memory were cut, by decision on 2026-11-04, even where pkmf ships them:

| Cut | pkmf item | Reason |
|---|---|---|
| Activation checkpointing | O10 | Trades ~30% recompute for memory. Lowers MFU by construction. |
| Liger fused-linear-CE | O11 | Removes the logit tensor; a memory win, not a FLOP win. |
| `ChunkedAdamW` | O16 | Bounds optimizer temporaries. Memory only. |
| CPU-offloaded optimizer moments | O17 | Host↔device transfer is strictly slower than staying resident. Ruled out outright — **nothing gets offloaded to CPU in this plan.** |
| Fused/deduplicated AME gather | O14/O15 | Deferred (former tier 4). Highest ceiling but must be written from scratch — `pkmf.memory.*` is not in the drop. |
| Per-group LR, 4× bank multiplier | O19/O20 | Deferred. Quality, not throughput. |
| Batch ramp, token-space WSD, branch-decay | O21–O27 | Deferred. Experiment throughput, not MFU. |

**Consequence to accept up front:** with activation checkpointing off, the micro-batch is
bounded by what fits. At 1 GPU that was 4 against the dense baseline's 16. At 4 GPUs
FSDP2 shards parameters, gradients and optimizer state four ways, so the bound is much
looser — how much looser is the first thing Tier 1 measures (§2.3).

What remains, in dependency order:

| Tier | Items | Mechanism |
|---|---|---|
| **1** | baseline + **O9** + **O31** | Measure honestly; delete the redundant gradient traffic |
| **2** | **O1–O5** | Take the banks out of FSDP2's gradient/optimizer path |
| **3** | **O6–O8** | Shard the banks vertically across the 4 GPUs |

Tier 2 is a hard prerequisite for Tier 3: vertical sharding needs a hand-managed
`main_grad` to reduce in a subgroup.

---

## 1. Test bed

### 1.1 Cluster

**entropy**, partition `h100` — a single node `h100a`, 8×H100. It is the only node with
`/storage_nvme_4`, which holds FineWeb, the pixi env and `HF_HOME`. Nothing in this plan
runs on `a100` or `common`.

- ssh alias is **`entropy_j`** (the config's `server: entropy` fails auth — override it).
- Only `~/.ssh/id_rsa` authenticates as `j321m`. Rapid retries trip fail2ban (~1 min);
  retry at most once a minute, never in parallel.
- `gres` syntax is `gpu:h100:N`; `configs/_cluster/entropy.yaml` leaves it `???`, so the
  experiment config **must** set it.
- Defaults from `_cluster/entropy.yaml`: `cpus_per_gpu: 14`, `mem_per_gpu: 90G`,
  `partition: h100`, `nodes: 1`.
- The GPU cap is **shared across the whole account** — these jobs compete with other
  people's sessions. Check with
  `sacctmgr -n -P show qos <qos> format=Name,MaxTRESPerUser`.
- GitHub:22 worked from the login node as of 2026-11-03 (verified by an unattended
  submission, job 32306). Re-check before assuming otherwise.

### 1.2 Submission

```bash
./.pixi/envs/default/bin/python run_exp.py \
    --config-path configs/ame --config-name <bare-name>
```

Per-directory `--config-path`, bare `--config-name`, and **never via `pixi run`** — it
rewrites `pixi.lock` v6→v7.

Rules that have cost hours before and apply to every submission in this plan:

1. **`version_code` `git add`s everything, including untracked files**, and pushes it to
   the cemetery branch before restoring your tree. `pkmf/` is already gitignored
   (`.gitignore:203`); check for new scratch before each submit.
2. **Submit from a named branch** — `version_code()` reads `repo.active_branch.name`, so
   a detached worktree dies instantly. Submitting from a throwaway branch in a clean
   worktree also stops other sessions' untracked files being swept into the push.
3. **Never SIGTERM `run_exp.py` mid-flight** — the restore is in a `finally` that a
   `timeout` wrapper bypasses, leaving a stray commit and HEAD on the experiment branch.
   Give it a generous timeout and background it.
4. `WANDB_API_KEY` is forwarded from the local shell. If it is missing locally the job
   runs with no logging and you find out hours later.
5. **Submitting is outward-facing — confirm with Maciej before each launch.**
6. **Verify ~2 minutes after submit:** job running, first steps in the log, first metric
   in wandb. Bad paths and init errors all fail inside the first couple of minutes.
7. Use `sbatch`, never an interactive `srun` over SSH — the job dies with the connection.
   A CPU-only `srun` (no `--gres`) still lands on the GPU node, sees `/storage_nvme_4`
   and the env, and costs no GPU quota: use it for path and version checks.
8. If a job sits `Pending`, triage before resubmitting:
   `sbatch --test-only --wrap="true"` predicts the start time and submits nothing.
   `QOSMaxGRESPerUser` is the account cap; `Resources` while GPUs look free usually means
   **RAM** (4 GPUs × 90G = 360G, which one memory-holding session can block).

Results land in wandb `ideas_cv/llm-random-test`, project `nano-ame`.

### 1.3 Workload, held constant across every run in this plan

Base config is `configs/ame/h20_smoke.yaml`. Every measurement keeps:

| | Value |
|---|---|
| Model | AME h20 — `d_model` 1280, 20 blocks, 20 heads, `d_head` 64, vocab 50304 |
| Parameters | **3.163B** (2.576B banks + 587M backbone) |
| FLOPs/token | **3.77 G** (`src/core/mfu.py`; the banks correctly contribute zero) |
| Sequence length | 2048 |
| **Global** batch | **512 sequences** → 1,048,576 tokens/optimizer step |
| Steps | 30, warmup 10; **eval off, checkpointing off** |
| LR | `2^-12` = 0.000244140625 |
| Distributed | FSDP2 from `_trainer/llama.yaml`, `MixedPrecisionPolicy(bf16, fp32)` |
| Compile | `compile_blocks_full: true`; no `autocast_bf16`, no `compile_loss`, no `compile_ce` |

`common.batch_size` is the **global** batch; `get_dataloader` hands each rank
`512 / world_size`. So the micro-batch is set by gradient accumulation:

```
gradient_accumulation_steps = 512 / (ngpu × microbatch_per_gpu)
```

| ngpu | micro-batch 4 | 8 | 16 |
|---|---|---|---|
| 1 | ga 128 | 64 | 32 |
| 4 | ga 32 | 16 | 8 |

**This is the pkmf invariant (ablation §5 of the survey): hold the global token schedule
fixed and vary only the physical split.** Every cell trains the identical 512×2048-token
step, so tokens/s and MFU are directly comparable across GPU counts and micro-batches.

MFU denominator is H100 bf16 dense peak, **989 TFLOP/s**. At 3.77 GF/token:

```
tokens/s per GPU = MFU × 989e12 / 3.77e9 = MFU × 262,335
```

so 18.1% = 47,474 tok/s/GPU and 44% = 115,400 tok/s/GPU. Report both.

### 1.4 Correctness gate — applies to every tier

Every change in Tiers 2 and 3 alters where gradients are reduced and who steps the
parameters. None of them may alter the mathematics. Before any throughput number is
quoted:

1. **CPU unit test on a toy model** (`d_model` 64, 2 blocks, vocab 256, 3 steps): the new
   path and the baseline path must produce **identical parameter tensors** after N steps
   under a fixed seed. A frozen-parameter or clipping bug reproduces here in seconds and
   costs a multi-hour run otherwise.
2. **Loss-trajectory equivalence on the cluster**: same seed, same config, 30 steps,
   baseline vs changed path. Compare per-step loss. Tier 1 (O9) must be **bit-identical**
   — suppressing sync during accumulation changes nothing mathematically. Tiers 2–3 are
   not bit-identical (different reduction order, separate clip) and need a stated
   tolerance plus a matching trajectory shape.
3. **Grad-norm trace** logged every step. A silent divisor error (§Tier 3, O8) shows up
   here as a constant factor and nowhere else.

---

## 2. Baselines

### 2.1 What exists

| Run | GPUs | ga | micro-batch | tokens/s | MFU | step | source |
|---|---:|---:|---:|---:|---:|---:|---|
| **AME h20**, fused kernel | 1 | 128 | 4 | 47,391 | **18.1%** | 22.1 s | entropy job 32360 |
| dense GPT-2-arch h20, FSDP2+compile | 1 | 32 | 16 | 109,361 | 45.57% | 9.58 s | `mfu_audit/h20_mgpu.yaml` |
| dense GPT-2-arch h20 | 2 | 16 | 16 | 214,499 | 44.69% | 4.89 s | ″ |
| dense GPT-2-arch h20 | 4 | 8 | 16 | **425,054** | **44.28%** | 2.47 s | ″ |

### 2.2 The baseline caveat, stated precisely

The multi-GPU dense table above is the **GPT-2-arch** h20 (4.122 GF/token), produced by
adding `GeluFeedForward`, `n_blocks=25` and `kv_heads=q_heads` overrides. The
**H-balanced** h20 that `configs/ame/h20_smoke.yaml` is built on is 3.97 GF/token and
**has never been measured multi-GPU**. AME itself is MHA-shaped at 3.77 GF/token.

So there are three different denominators in play. MFU percentages are comparable across
them (each uses its own FLOPs/token); raw tokens/s are **not**.

### 2.3 Baselines this plan must produce first

**B1 — AME h20 at 4 GPUs, micro-batch sweep.** The number every later result is read
against, and the test of whether the micro-batch-4 bound was a 1-GPU artefact.
Config: copy `configs/ame/h20_smoke.yaml` → `configs/ame/h20_4gpu.yaml`, set
`gres: gpu:h100:4`, and submit three variants:

| variant | `gradient_accumulation_steps` | micro-batch/GPU |
|---|---:|---:|
| `h20_4gpu_mb4` | 32 | 4 |
| `h20_4gpu_mb8` | 16 | 8 |
| `h20_4gpu_mb16` | 8 | 16 |

Record tokens/s, MFU/GPU, step time, **peak GiB/GPU**. Expect `mb16` to fit: at 4 GPUs
FSDP2 shards the 3.163B parameters, fp32 masters, fp32 grads and both Adam moments four
ways, so the ~57 GiB that forced micro-batch 4 at 1 GPU becomes ~14 GiB/GPU of resident
state. If `mb16` OOMs, record where and fall back; that is a finding, not a failure.

**B2 — dense H-balanced h20 at 4 GPUs**, micro-batch 16, ga 8, as the honest
same-architecture comparator. One run, `configs/mfu_audit/h20_mgpu.yaml` with
`gres: gpu:h100:4` and `gradient_accumulation_steps: 8`, **no** GPT-2-arch overrides.

Everything downstream is reported as a delta against B1, with B2 as the ceiling.

---

## Tier 1 — Measure, and delete the redundant gradient traffic

**Goal:** a trustworthy 4-GPU AME baseline, the per-submodule attribution that tells us
where the remaining time goes, and the one free throughput fix.

### T1.1 — Baselines B1 and B2

As §2.3. No code change. Gate: B1 recorded at all three micro-batches with peak memory,
B2 recorded.

**Configs written and pre-flighted 2026-11-04** (compose + `check_point` + `instantiate`
on the `meta` device, all four clean):

| run | config | ga | mb/GPU | params | wandb name |
|---|---|---:|---:|---:|---|
| B1/mb4 | `configs/ame/h20_4gpu_mb4.yaml` | 32 | 4 | 3.163B | `ame_h20_4gpu_mb4` |
| B1/mb8 | `configs/ame/h20_4gpu_mb8.yaml` | 16 | 8 | 3.163B | `ame_h20_4gpu_mb8` |
| B1/mb16 | `configs/ame/h20_4gpu_mb16.yaml` | 8 | 16 | 3.163B | `ame_h20_4gpu_mb16` |
| B2 | `configs/mfu_audit/h20_hbal_4gpu.yaml` | 8 | 16 | 0.620B | `mfu_h20_hbal_4gpu_mb16` |

All four are `gres: gpu:h100:4`, 30 steps, global batch 512 × 2048, eval and
checkpointing off, `compile_blocks_full: true`, project `nano-ame`.

The submission command needs one CLI override that is deliberately **not** in the
committed config — `infrastructure.server=entropy_j`. `_cluster/entropy.yaml` ships
`server: entropy`, which is the `mstefaniak` alias and fails auth; `entropy_j` is
`j321m`. Personal account aliases stay out of shared config.

```bash
./.pixi/envs/default/bin/python run_exp.py \
    --config-path configs/ame --config-name h20_4gpu_mb4 \
    infrastructure.server=entropy_j
```

**Metrics come from the trainer, not from arithmetic.** `_log_throughput`
(`src/core/trainer.py:716`) already logs `train/step_time`, `train/tokens_per_s` and
`train/mfu`, and the MFU denominator is correctly the **global** peak (per-GPU peak ×
world size), so `train/mfu` is already MFU *per GPU*. `train/tokens_per_s` is global.
`train/max_memory_gib` is logged once at the end of training, from rank 0. The job log
also prints one `MFU accounting:` line naming the measured GFLOPs/token, which is how
§2.2's three denominators get confirmed rather than assumed.

### T1.2 — O9: suppress gradient sync during accumulation

**What.** `src/core/trainer.py:549` `calculate_loss` loops `loss.backward()` over
`gradient_accumulation_steps` chunks with no sync control. Under FSDP2 every one of those
backwards triggers a full reduce-scatter of all 3.163B gradients, in **fp32**
(`reduce_dtype=torch.float32`, `src/core/distributed_training.py:66`). Only the last one
is needed.

**Change.** Disable gradient sync for micro-steps `0 .. GA-2`, enable it for the last —
FSDP2's `set_requires_gradient_sync(bool)` on the root module, guarded so non-FSDP2 and
CPU paths are unaffected. Roughly 10 lines in `calculate_loss`.

**Why it is first.** Traffic estimate at B1's `mb4` shape (ga 32), fp32 reduce-scatter of
3.163B parameters ≈ 12.65 GB moved per micro-step:

| | reduce-scatters/step | fp32 bytes moved/step |
|---|---:|---:|
| today, 4 GPU, ga 32 | 32 | ~405 GB |
| today, 1 GPU, ga 128 | 128 | ~1.6 TB (local copy + cast, no NCCL) |
| after O9 | **1** | ~12.65 GB |

These are estimates from parameter count and dtype, not measurements — T1.3 is what
converts them into attributed time. But the ratio is 32:1 and the fix is ten lines.

**Gate — corrected 2026-11-04.** The plan originally demanded a **bit-identical** loss
trajectory. That gate is unreachable at 4 GPUs and asking for it would have sent us
chasing a non-bug:

```
sync on   ->  rank r ends with   sum_i  shard_r( sum_ranks g_i )
sync off  ->  rank r ends with   shard_r( sum_ranks sum_i g_i )
```

Same terms, different association order. Floating-point addition is not associative, so
the two agree exactly only when there is no cross-rank sum to reassociate — i.e. at
world_size 1. The honest gate is therefore **two-level**:

| world_size | gate |
|---|---|
| 1 | **bit-identical** gradients — no reassociation is possible, so any difference is a bug |
| > 1 | equal to **fp32 tolerance**, with grad-norm traces overlaid |

**Landed 2026-11-04.** Opt-in flag `trainer.skip_grad_sync_during_accumulation`
(default `false`), implemented as `Trainer._accumulation_grad_sync`
(`src/core/trainer.py`), declared in `TrainerSchema` (`src/definitions.py`). Inert unless
the model is an FSDP2 module, the flag is set, `gradient_accumulation_steps > 1`, and the
model is in training mode.

**The memory cost, which the plan did not previously state.** O9 is free in FLOPs and
*not* free in memory. With sync suppressed, FSDP2 holds each group's **unsharded**
gradient between micro-steps and upcasts it to the fp32 reduce dtype
(`to_accumulated_grad_if_needed`, `torch/distributed/fsdp/_fully_shard/_fsdp_param.py`).
Resident gradient per rank:

| | per-rank fp32 gradient at 3.163B params, 4 GPUs |
|---|---:|
| today (sharded) | 3.16 GB |
| with O9 (unsharded) | **12.65 GB** |
| difference | **+9.5 GB/GPU** |

Parameters still reshard after backward; only the gradient stays unsharded. At larger
micro-batches this may be what decides O9, not the saved communication — so **peak
GiB/GPU is reported next to tokens/s on every O9 run**, and `mb16` is the one at risk.

**Tests.** `tests/test_grad_sync_accumulation.py`, 6 tests, over a real FSDP2 CPU/gloo
mesh (the pattern `tests/test_distributed.py` established), at world_size 1 and 2, with
and without the production `MixedPrecisionPolicy(bf16, fp32)`.

A **negative control** was run to confirm the tests are not vacuous, and it found the
sharpest failure mode — which is not numeric drift. While sync is suppressed FSDP2 parks
the running sum in `unsharded_accumulated_grad` and leaves `param.grad` as **None**; only
the reducing micro-step materialises it. A loop that never re-enables sync therefore ends
with every `.grad` None — 7 of 7 parameters on the toy model. `clip_gradient`
(`src/core/trainer.py`) filters on `p.grad is not None`, so that state would clip nothing
and step on nothing while the loss kept being logged. The restore lives in a `finally`
for exactly this reason, and the test asserts gradients exist before comparing values.

**Measurement configs** (identical to their B1 twins except the flag, the job name and
the tags): `configs/ame/h20_4gpu_mb{4,8,16}_o9.yaml`. All pre-flighted.

**Scope note.** This is a framework fix, not an AME fix. It benefits every large model in
nano and should be reviewed as such. It is opt-in rather than default-on **because of the
memory cost** — flipping the default would change peak memory for every existing
multi-GPU config in the repo, and that is a separate decision with its own evidence.

### T1.3 — O31: per-submodule forward/backward attribution

**The original design does not work, and fails silently. Corrected 2026-11-04.**
O31 was specified as CUDA-event pairs around each AME sub-block — bank gather, QK-norm,
the KV-shift mix, `flex_attention`, `v_proj`/`o_proj`. Every one of those sits *inside* a
`TransformerBlock`, and `compile_blocks_full: true` compiles each whole block. Measured
with `torch._dynamo.explain` on torch 2.7.1:

```
plain      graphs=1 breaks=0 ops=2
annotated  graphs=1 breaks=0 ops=2
W... Profiler function <class 'torch.autograd.profiler.record_function'> will be ignored
```

The annotation does **not** break the graph — dynamo *drops* it. `ops` is unchanged, so
the annotation is simply absent from the compiled code. Annotating inside a compiled
block therefore yields no attribution at all, with no error and no missing-data signal:
it would have produced an empty table that looked like a broken profiler. A CUDA-event
pair inside the block would instead force a real graph break, measuring a model that is
not the one we ship.

**What replaces it**, all of it outside the compiled region and so compile-safe:

| instrument | what it gives | why it survives compile |
|---|---|---|
| `trainer.profile_timers` | `data_fetch`, `data_h2d`, **`forward`**, **`backward`**, **`clip`**, **`optim`**, `log`, vs `step_total`, as a `timers_ms:` line per step | `LoopTimers` wraps the *call sites* in `train()` and `calculate_loss`, not the model internals |
| `trainer.kernel_profile` | one `kprof_us:` line per CUDA kernel over steps 12–13 | kernel names are observed, not injected |

The gate's "≥90% of step time" is satisfied **by construction** rather than by hope: the
`LoopTimers` sections are a partition of the step, so coverage is a property of the
instrument.

Per-*path* attribution then comes from the kernel dump, because inductor fuses pointwise
work into opaque `triton_poi_fused_*` kernels but does **not** fuse away the four things
this tier must separate. `scripts/kprof_agg.py` gained two categories for exactly this
(`tests/test_kprof_agg.py` asserts they are purely additive — they only claim kernels that
previously fell through to `other`, so every table in `training_efficiency.md` still
reproduces):

| category | catches | answers |
|---|---|---|
| **`comm(nccl)`** | `ncclDevKernel_ReduceScatter*`, `*AllGather*`, `*AllReduce*` | gradient communication — O9's target, and Tier 3's |
| **`embedding/bank`** | `embedding_dense_backward`, `compute_grad_weight`, `krn_partial_segment_offset`, `sum_and_scatter`, `indexSelect*`, `index_add*` | **the bank gradient path** — what Tier 2 exists to remove |

The ordering of those two inside `CATS` is load-bearing and nearly went wrong: first match
wins, and inductor names fused kernels after the ops it fused, so
`triton_poi_fused_embedding_dense_backward_3` is a real kernel name. Placed ahead of
`compiled(triton)` the embedding pattern steals it and silently moves an existing
category's number. They sit *after* it, and a test pins that.

**Cross-check, because one instrument is not evidence.** The banks are 2.576B of 3.163B
parameters = **81.4%**, and both AdamW and reduce-scatter cost scale with element count.
So the measured `embedding/bank` + bank share of `comm(nccl)`/`optimizer` should land near
that analytic share. If the two readings disagree, neither is trusted until reconciled.

**Reading the results.** `scripts/run_summary.py <slurm log>` reduces a finished run to
the row this ledger needs — the `MFU accounting:` denominator, per-GPU MFU, global
tokens/s, step time, peak GiB, and the `timers_ms` section split averaged over a
steady-state window (rank 0 only, from step 15, which clears both warmup and the two
profiled steps). It prints the section coverage percentage directly, so the gate's
"≥90% of step time" is read off rather than argued. It **exits non-zero on any failure
signature** — OOM, traceback, `srun: error`, a cancellation — because a log that crashed
and a log that finished look identical otherwise. Tested in `tests/test_run_summary.py`.

**Still unavailable:** sub-block attribution *within* the compiled block (gather vs QK-norm
vs the mix vs the kernel). Getting it needs either a compile-off run — which measures a
different machine, compile being worth ~11 MFU points — or inductor provenance tracking.
Neither is needed for the Tier 2/3 decision, so neither is in scope. Recorded so the gap
is known rather than assumed covered.

**Gate.** An attribution table for AME h20 at 4 GPUs that accounts for ≥90% of step time.
**This table decides whether Tiers 2 and 3 are worth their cost** — if the banks'
gradient and optimizer path is not a top-two term after O9, we re-rank before building
anything.

### Tier 1 exit criteria

- B1 (3 micro-batches) and B2 recorded in this document.
- O9 landed, bit-identical, with its measured delta.
- Attribution table covering ≥90% of step time, with the bank grad/optimizer path
  quantified.

---

## Tier 2 — Hand-managed bank gradient and optimizer path (O1–O5)

**Goal:** stop FSDP2 and AdamW touching 2.576B of bank parameters densely on every
micro-step. pkmf measured **1.43×** end-to-end for this mechanism alone on 1 GPU.

### What gets built

A controller modelled on `pkmf/src/poirot/memory_parallel.py:696`
`MemoryParallelController`, owning every `q_bank.weight` / `k_bank.weight`:

| | Mechanism | pkmf reference |
|---|---|---|
| **O1** | For each bank: set `requires_grad=False` so FSDP2 creates no master, no reduce and no step; attach a **persistent fp32 `main_grad`**; keep a **private fp32 master** that a dedicated AdamW steps. The backward scatters into `main_grad` and never zeros it, so it accumulates across micro-steps for free. | `memory_parallel.py:696` |
| **O2** | One `all_reduce(SUM)` of `main_grad` per optimizer step, then `div_(world_size)` to preserve mean semantics. | `memory_parallel.py:1024` |
| **O3** | fp32 master → bf16 model param copy after each step. The forward reads the bf16 param. | ″ |
| **O4** | Separate `clip_grad_norm_` over the masters. | ″ |
| **O5** | Three hook points in the step loop: *step begin* → re-home master and `main_grad` onto the param's device; *after backward, at the accumulation boundary* → all-reduce and step; *log* → surface the bank LR. | `callbacks.py:3347` |

### The AME-specific problem Tier 2 must solve

pkmf's banks are read by a **custom fused kernel** that `atomicAdd`s straight into
`main_grad`. AME's banks are `nn.Embedding` (`src/core/ame.py:269-270`), whose backward is
`embedding_dense_backward` writing to `.grad`. With `requires_grad=False` there is no
`.grad` and **no gradient at all** — autograd simply stops.

Three candidate routes, to be decided by a short spike before the full build:

| Route | Mechanism | Risk |
|---|---|---|
| **A** | Keep `requires_grad=True`, hide the banks from FSDP2 by excluding them from the sharded module set, and install a post-accumulation hook that folds `.grad` into `main_grad` | Smallest change; does not remove the dense `.grad` allocation, so it captures O2–O5 but only part of O1 |
| **B** | Custom `autograd.Function` wrapping the gather, whose `backward` scatters into `main_grad` via `index_add_` and returns no parameter gradient | Removes the dense per-micro-step grad; `index_add_` on a 2.576B fp32 buffer is the thing to benchmark |
| **C** | As B, with the scatter fused into a custom kernel | This is O14; **out of scope**, deferred |

**Route B is the expected choice**, with A as the fallback if the spike shows
`index_add_` is not competitive. Decide on measurement, not on preference.

### Measurement

Re-run the B1 micro-batch sweep at 4 GPUs, plus one 1-GPU run to compare directly against
pkmf's 1.43× claim on a shape where no NCCL traffic confounds it.

### Gates

- CPU toy-model equivalence (§1.4.1) — parameters identical after 3 steps.
- 30-step loss trajectory matching B1 within a stated tolerance, with grad-norm traces
  overlaid.
- Throughput delta reported at 1 and 4 GPUs, micro-batch held.

### Risk register

- **`index_add_` cost** (route B) — 2.576B fp32 target; benchmark standalone before
  committing.
- **Interaction with `compile_blocks_full`** — a custom `autograd.Function` inside a
  compiled region may graph-break. Measure with and without.
- **Interaction with the existing clip** — `src/core/trainer.py:649` computes a correct
  global norm over DTensor shards. Splitting the clip changes the math. Either clip
  jointly (gather both norms) or document the change as deliberate; do not let it happen
  silently.
- **The bank AdamW is a single fused/foreach step over 2.576B params.** pkmf hit an OOM
  here with foreach and solved it by chunking (O16), which we cut. If it OOMs, the
  minimal unblock is `foreach=False, fused=False` on that optimizer only — not a
  reintroduction of chunking.

---

## Tier 3 — Vertical bank sharding across the 4 GPUs (O6–O8)

**Goal:** cut per-rank bank VRAM and bank gradient comm by `mem_parallel_size`.
pkmf measured **~2.2×** for O1+O6+O7 together on their 4-GPU grid (215k vs 97k tok/s),
against ~1.00× for the hand-managed path *without* sharding — i.e. **on multi-GPU this
tier is where the win actually is.**

### What gets built

| | Mechanism | pkmf reference |
|---|---|---|
| **O6** | Two process groups at init. **Memory-parallel group**: contiguous blocks of `mem_parallel_size` ranks; each owns a column slice `[vocab, width/mp]` of every bank. **Memory-DP subgroup**: strided, size `world/mp`; `main_grad` is all-reduced *only* here. At `mp == 1` every helper is identity. | `memory_parallel_groups.py:43` |
| **O7** | Forward: all-gather token ids within the mp-group → local gather on the owned slice for all ranks' tokens → all-to-all so each rank ends with its own tokens at full width. Backward mirrors it. Autograd wrappers: all-gather fwd / reduce-scatter bwd; all-to-all with swapped splits on bwd. | `memory_parallel_groups.py:192, 229` |
| **O8** | **The divisor.** The all-reduce runs in the mem-DP subgroup (size `world/mp`), but the divisor stays the **global** `world_size` — the forward's id all-gather already summed `mp` ranks' tokens into each slice's local gradient, so `mp × (world/mp) = world`. | `memory_parallel.py:1024` docstring |

At `world=4, mp=4`: one mp-group `[0,1,2,3]`, four singleton dp-groups → the bank
all-reduce disappears entirely and each rank holds a quarter of every bank.

### Why the column split fits AME better than it fits pkmf

AME's bank is `[vocab, q_heads × d_head]`. A **column** slice is therefore a **head
subset**, and heads are independent until `o_proj`. Each rank can serve any token id from
its own slice, so no rank ever needs a row it does not own.

This is a different choice from the one `ame.md §7` anticipated ("shard them the way
Megatron shards a vocab-parallel embedding — rows split across ranks"). The row split
would need every rank to fetch arbitrary rows; the column split does not. **Record this
as a deliberate deviation from the design record.**

Open design question for the spike: whether to all-to-all the gathered rows back (pkmf's
scheme, keeps attention unchanged) or to let each rank run attention for its own heads and
all-gather only before `o_proj` (cheaper, but it is tensor parallelism of the attention
layer and interacts with FSDP2 and `flex_attention`). **Default to pkmf's scheme**; the
second is a separate investigation.

### Measurement

Sweep `mem_parallel_size ∈ {1, 2, 4}` at 4 GPUs with micro-batch held at B1's best,
reporting tokens/s, MFU/GPU and peak GiB/GPU for each.

### Gates

- Loss trajectory at `mp ∈ {1,2,4}` must coincide — this is the test that catches an O8
  divisor error, which otherwise shows up as a constant factor on the grad norm and
  nothing else.
- Checkpoint save/load at `mp=4` round-trips to an evaluable model.
- Throughput table across `mp` and micro-batch.

### Risk register

- **O8's divisor** — the single most likely silent bug in this plan. Assert it in a unit
  test comparing `mp=1` and `mp=4` gradient magnitudes on the toy model.
- **`overlap_comm` is a trap with vertical sharding** (O13). pkmf measured overlap
  *halving* VS throughput (215k → 123k). The FSDP2 instinct is to add overlap; measure
  before enabling anything of the sort.
- **Checkpointing.** 3.163B parameters is minutes, not seconds, to gather and write. The
  benchmark configs keep checkpointing off, so this only bites when a real training run
  follows. pkmf's answer is rank-local shards (O30) — out of scope here, flagged as the
  follow-up that real runs will need.
- **FSDP2 interaction.** The banks are excluded from FSDP2 from Tier 2 onward, so Tier 3
  shards them independently. The two sharding schemes must not both claim the same
  parameters; assert it at construction.

---

## 3. Experiment ledger

Every run recorded here as it completes: job id, config, wandb run, tokens/s, MFU/GPU,
peak GiB/GPU, step time. Cancelled runs recorded as "ignore" so later comparisons do not
pick them up.

| # | Tier | Config | GPUs | ga | mb | job | tokens/s | MFU/GPU | peak GiB | notes |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---|
| ~~B1/mb4~~ | 1 | `ame/h20_4gpu_mb4` | 4 | 32 | 4 | 32541 | — | — | — | **ignore** — cancelled 17:05, never ran |
| ~~B1/mb8~~ | 1 | `ame/h20_4gpu_mb8` | 4 | 16 | 8 | 32542 | — | — | — | **ignore** — cancelled 17:05, never ran |
| ~~B1~~ | 1 | `ame/h20_4gpu_mb16` | 4 | 8 | 16 | 32543 | — | — | — | **ignore** — cancelled 19:44, pre-fix snapshot, never ran |
| ~~B2~~ | 1 | `mfu_audit/h20_hbal_4gpu` | 4 | 8 | 16 | 32544 | — | — | — | **ignore** — cancelled 19:44, never ran |
| ~~O9~~ | 1 | `ame/h20_4gpu_mb16_o9` | 4 | 8 | 16 | 32555 | — | — | — | **ignore** — cancelled 19:44, never ran |
| ~~O31~~ | 1 | `ame/h20_4gpu_mb16_attrib` | 4 | 8 | 16 | 32559 | — | — | — | **ignore** — cancelled 19:44, never ran |
| **B1** ✅ | 1 | `ame/h20_4gpu_mb16` | 4 | 8 | 16 | **32576** | **281,748** | **26.84%** | **60.6** | 3.720 s/step, wandb `89nupukg` |
| **B2** ✅ | 1 | `mfu_audit/h20_hbal_4gpu` | 4 | 8 | 16 | **32577** | **443,432** | **44.44%** | **43.4** | 2.364 s/step, dense ceiling, wandb `oh5lepzx` |
| **O9** ✅ | 1 | `ame/h20_4gpu_mb16_o9` | 4 | 8 | 16 | **32578** | **274,627** | **26.16%** | **69.5** | 3.840 s/step, **slower than B1**, wandb `aoa5vzdj` |
| **O31** ✅ | 1 | `ame/h20_4gpu_mb16_attrib` | 4 | 8 | 16 | **32579** | 274,179 | 26.11% | 60.7 | attribution run; profiler costs ~2.7%, **not a baseline**, wandb `fztfn0lu` |
| **O31-dense** ✅ | 1 | `mfu_audit/h20_hbal_4gpu_attrib` | 4 | 8 | 16 | **32601** | — | — | 43.4 | dense attribution twin of 32579; 2360.3 ms/step |
| **trace-AME** ✅ | 1 | `ame/h20_4gpu_mb16_attrib` | 4 | 8 | 16 | **32623** | — | — | — | chrome trace for the fwd/bwd split |
| **trace-dense** ✅ | 1 | `mfu_audit/h20_hbal_4gpu_attrib` | 4 | 8 | 16 | **32624** | — | — | — | ditto, dense |
| ~~B1-lem~~ | 1 | `ame/h20_4gpu_mb16_lem` | 4 | 8 | 16 | 6236712 | — | — | — | **ignore** — died in compile, Triton cache race, 0 steps |
| ~~O9-lem~~ | 1 | `ame/h20_4gpu_mb16_o9_lem` | 4 | 8 | 16 | 6236713 | — | — | — | **ignore** — same, 0 steps |
| ~~B1-lem.2~~ | 1 | `ame/h20_4gpu_mb16_lem` | 4 | 8 | 16 | 6236888 | — | — | — | **ignore** — triton fix held; died on the DTensor diagnostics bug |
| ~~O9-lem.2~~ | 1 | `ame/h20_4gpu_mb16_o9_lem` | 4 | 8 | 16 | 6236892 | — | — | — | **ignore** — same |
| **B1-lem** ✅ | 1 | `ame/h20_4gpu_mb16_lem` | 4 | 8 | 16 | **6237108** | **256,123** | n/a | **60.65** | **lem**, 4.092 s/step, wandb `jyvfozb7` |
| **T23/mp4** ✅ | 2+3 | `ame/h20_4gpu_mb16_mp4` | 4 | 8 | 16 | **32632** | **278,636** | **26.54%** | **61.91** | 3.761 s/step, loss matches B1 for 28/30 steps, wandb `wix4sh1s` |
| **T23/mp4-attrib** ✅ | 2+3 | `ame/h20_4gpu_mb16_mp4_attrib` | 4 | 8 | 16 | **32634** | — | — | 61.92 | attribution twin of 32632; **not a baseline**, wandb `ca52qnp4` |
| **T23/mp2** ❌ | 2+3 | `ame/h20_4gpu_mb16_mp2` | 4 | 8 | 16 | **32638** | — | — | **OOM** | **CUDA OOM at 76.56 GiB in use / 79.18 total**, wandb `h800c0v3` |
| **T2/mp1** ❌ | 2 | `ame/h20_4gpu_mb16_mp1` | 4 | 8 | 16 | **32649** | — | — | **OOM** | **CUDA OOM at 79.16 / 79.18 GiB** in the first optimizer step; fwd+bwd fitted, Adam moments did not |
| **T2/1gpu** ✅ | 2 | `ame/h20_1gpu_mb4_mp1` | 1 | 128 | 4 | **32652** | **65,651** | **25.01%** | **64.31** | **1.385x over job 32360** (47,391 / 18.1% / 22.1 s); 15.966 s/step, loss 11.25→7.28, wandb `euckimjt` |
| **B1/mb4** ✅ | 1 | `ame/h20_4gpu_mb4` | 4 | 32 | 4 | **32656** | **192,092** | **18.30%** | **24.06** | 5.456 s/step, the deferred B1 sweep point revived as the ga-scaling twin, wandb `do325fpp` |
| **T23/mb4-mp4** ✅ | 2+3 | `ame/h20_4gpu_mb4_mp4` | 4 | 32 | 4 | **32659** | **235,421** | **22.42%** | **25.29** | **+22.56% over 32656**; 4.452 s/step, wandb `1j8hwgj9` |

**Everything above this line ran with the UNFUSED bank path and
`@torch.compiler.disable`.** D2 changed the code, not the configs, so re-running
a config from the table above now measures something different. Runs from here
carry `_fused` in their wandb name to keep the two apart.

| # | Tier | Config | GPUs | ga | mb | job | tokens/s | MFU/GPU | peak GiB | notes |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---|
| **D3/ddp-mb4** ✅ | D3 | `ame/h20_4gpu_mb4_ddp_mp4` | 4 | 32 | 4 | **32664** | **241,679** | **23.02%** | **39.75** | **+25.81% over 32656**; 4.337 s/step, carries D2 as well, wandb `fwt5br9d` |
| **D2/mb16-fused** ✅ | D2 | `ame/h20_4gpu_mb16_mp4` + fused | 4 | 8 | 16 | **32666** | **287,729** | **27.41%** | **61.90** | **+2.20% over B1** — the pinned test bed, flipped from -1.03%, wandb `l4tjf6u4` |
| **D3/ddp-mb16** ❌ | D3 | `ame/h20_4gpu_mb16_ddp_mp4` | 4 | 8 | 16 | **32672** | — | — | **OOM** | **CUDA OOM at 73.68 / 79.18 GiB**, needing 6.14 GiB for cross-entropy |
| **D2/mb4-fused** ✅ | D2 | `ame/h20_4gpu_mb4_mp4` + fused | 4 | 32 | 4 | **32674** | **246,636** | **23.49%** | — | **+28.39% over 32656** — the best configuration measured; 4.250 s/step, wandb `ame_h20_4gpu_mb4_mp4_fused` |
| **O9-lem** ✅ | 1 | `ame/h20_4gpu_mb16_o9_lem` | 4 | 8 | 16 | **6237109** | **264,659** | n/a | **69.50** | **lem**, 3.960 s/step, wandb `jroouzw9` |

**Two clusters are in this table and their rows must not be compared across.** lem is
`gpu:hopper:4` — four GPUs per *node* — against entropy's 8×H100 SXM in one node. O9 is a
communication optimisation, so what it is worth is a property of the interconnect.
Compare **B1-lem with O9-lem**, and **B1 with O9**; never a lem row with an entropy row.
Each cluster yields its own self-consistent A/B. B2 (the dense ceiling) and O31 (the
attribution) exist only on entropy and are not duplicated.

All live runs sit at **ga 8 / micro-batch 16**, the shape `h20_mgpu.yaml` measured
the dense model at on 4 GPUs. 32543 vs 32555 is the O9 A/B; 32544 is the
same-architecture ceiling; 32559 carries `profile_timers` + `kernel_profile` and is an
attribution run, so **its throughput is not a baseline** — the profiler costs time. 32541 and 32542 were the deferred micro-batch sweep points,
cancelled before running. **`h20_4gpu_mb4` was revived on 2026-11-04 as job 32656**,
once the 1-GPU result made the micro-batch the interesting variable rather than a
leftover: at ga 32 / micro-batch 4 it measures 192,092 tok/s, 18.30% MFU,
5.456 s/step, 24.06 GiB — so **micro-batch 16 is 1.466x micro-batch 4 on an
identical token schedule**, and that ratio is the per-micro-step overhead Tier 2
exists to delete. `h20_4gpu_mb8` (ga 16) remains unrun.

B1 and B2 were submitted from branch `ame_optimize` at `f302ae12` with the working tree
carrying only configs and documents — **deliberately before the O9 code change**, so the
baseline measures unmodified code. 32555 carries the O9 implementation. wandb project
`nano-ame`; run names match the config names in §T1.1.

**Queue note.** None of the four started on submission. The shared `j321m` account had
six 4-GPU jobs named `brrrrr` submitted at 15:55–15:56 from another session, filling
h100a (8/8 GPUs) and stacking the queue hourly, so the four baselines are scheduled
22:00 / 23:00 / 00:00 / 01:00. No preemption occurred: our QOS preempts only
`lukaskoz_h100`, which by then held a single GPU — not the four a baseline needs.

Reference points to carry into it: AME 1-GPU = 47,391 tok/s / 18.1% / 22.1 s (job 32360);
dense GPT-2-arch 4-GPU = 425,054 tok/s / 44.28% / 2.47 s.

### First result — O9 measured on lem, 4×H100, ga 8 / micro-batch 16

Both runs completed 30 steps on identical data from identical init.

| | step time | tokens/s (global) | tokens/s per GPU | peak GiB/GPU |
|---|---:|---:|---:|---:|
| **B1** (baseline) | 4.092 s | 256,123 | 64,031 | **60.65** |
| **O9** (sync suppressed) | **3.960 s** | **264,659** | **66,165** | **69.50** |
| **delta** | **−3.2%** | **+3.33%** | **+3.33%** | **+8.85** |

Steady-state means over steps 15–29.

**The memory prediction was right to within 1%.** §T1.2 predicted +9.5 GB/GPU from
`P × 4 bytes × (1 − 1/world)` = 3.163e9 × 4 × 0.75 = 9.49 GB = **8.84 GiB**. Measured:
**8.85 GiB**. The mechanism is understood, not merely observed.

**The throughput win is real but small, and that is the more interesting number.** At
ga 8 there are only 8 reduce-scatters per optimizer step and O9 removes 7 of them, yet
the step gets just 3.3% faster. **Gradient communication is therefore not a dominant term
at this shape** — which is a direct, if partial, answer to the question T1.3 exists to
settle, and it lowers the prior on Tiers 2 and 3. Note the plan's §T1.2 traffic estimate
assumed ga 32 (micro-batch 4); at ga 8 the saving was always going to be 4× smaller.

**Correctness gate: PASSED.**

| check | result |
|---|---|
| Step-0 grad norm, B1 vs O9 | 4.00139 vs 4.00137 — **4×10⁻⁶ relative** |
| Divisor error (a constant factor) | **none** — the ratio is 0.999995, not 4 or ¼ |
| Losses bit-identical | **19 of 30 steps** (0–18) |
| max \|Δloss\|/loss over 30 steps | 1.29×10⁻² |
| mean \|Δloss\|/loss | 2.45×10⁻³ |

Step 0 is the decisive one. §1.4.3 says a silent divisor error "shows up here as a
constant factor and nowhere else"; there is no constant factor, and agreement to 4×10⁻⁶
at the first optimizer step is fp32 reassociation and nothing else. Thereafter the
trajectories separate as chaos amplifies through `gradient_clipping=1.0`, which is
nonlinear and active for the first ~16 steps.

**One thing to keep watching:** the loss difference is **one-sided** — O9 is never above
the baseline, equal in 19 steps and below in 11. Pure rounding should be symmetric. The
benign reading is that the logged loss is quantised to bf16 (ulp 0.03125 at ≈7, and every
difference is exactly 1–3 ulps) over a monotonically decreasing series, so a vanishingly
small lead reads as "lower" rather than "sometimes higher". That is consistent with the
evidence, but it is an explanation, not a measurement. **If a longer O9 run ever shows a
persistent one-sided loss gap, revisit this.**

**MFU is not reported for these runs.** lem's GPUs identify as plain `NVIDIA H100`, which
matches no entry in `src/core/mfu.py:_PEAK_BF16` (`H100 80GB HBM3`, `H100 PCIe`,
`H100 NVL`), so `peak_flops_bf16` correctly returned `None` and the trainer logged
tokens/s without MFU — the module's documented behaviour, and the right one: a guessed
peak is worse than no MFU. Entropy's H100s are recognised, so the canonical MFU numbers
come from there. **Do not convert these tokens/s to MFU by assuming 989 TFLOP/s.**

### Tier 1 result — the gate says STOP, and O9 is a negative result on entropy

All four entropy runs completed 2026-11-04 20:05–20:25 on h100a, 4×H100 SXM
(`NVIDIA H100 80GB HBM3`, 989 TFLOP/s/GPU), ga 8 / micro-batch 16, 30 steps.

| run | MFU/GPU | tokens/s (global) | per GPU | step | peak GiB |
|---|---:|---:|---:|---:|---:|
| **B1** AME baseline | **26.84%** | 281,748 | 70,437 | 3.720 s | 60.6 |
| **B2** dense H-balanced | **44.44%** | 443,432 | 110,858 | 2.364 s | 43.4 |
| **O9** sync suppressed | **26.16%** | 274,627 | 68,657 | 3.840 s | 69.5 |

**AME reaches 60.4% of the dense ceiling** — 26.84% against 44.44%, a gap of 17.6 MFU
points. B2 also validates the instrument: 44.44% here against the 44.28% recorded for the
GPT-2-arch h20 at 4 GPUs, two different architectures landing within 0.2 points, each
against its own FLOPs/token denominator (3.964 vs 4.122 G).

#### O9 measures *worse* on entropy, and the opposite of lem

| | entropy (8×H100 SXM, one node) | lem (4 GPUs/node) |
|---|---:|---:|
| B1 tokens/s | 281,748 | 256,123 |
| O9 tokens/s | 274,627 | 264,659 |
| **O9 delta** | **−2.53%** | **+3.33%** |

**Same code, same config, opposite sign.** This is not noise — it is the mechanism, and
it is exactly why §3 forbids comparing rows across clusters.

O9 does not make communication free; it *trades NCCL traffic for HBM traffic*. With sync
suppressed FSDP2 accumulates into an unsharded fp32 buffer on every micro-step
(`unsharded_accumulated_grad += unsharded_param.grad`), which is a read-modify-write of
12.65 GB per micro-step — ~101 GB of extra HBM traffic per optimizer step at ga 8 —
in exchange for removing 7 of 8 reduce-scatters. On entropy's intra-node NVLink the
reduce-scatter was already cheap and well overlapped, so the trade loses. On lem's
topology it wins.

**Verdict: O9 is rejected on the plan's test bed.** Per the Goal, "a tier that measures
worse is a completed tier with a negative result, not a failure — record it and stop, do
not tune past the gate." The flag stays in the tree, default `false`, documented and
tested, because it is a real win on a slower interconnect — it is simply not one here.

#### O31 attribution — where the 3.84 s step actually goes

Section timers (`profile_timers`, rank 0, steps 15–29), which tile the step by
construction:

| section | ms | share |
|---|---:|---:|
| forward | 1515.4 | 39.4% |
| backward | 2308.6 | 60.1% |
| log (host, overlaps) | 217.0 | 5.6% |
| **optim** | **12.3** | **0.3%** |
| clip | 4.1 | 0.1% |
| data_fetch + data_h2d | 0.5 | 0.0% |
| **step_total** | **3842.6** | coverage **105.6%** |

Kernel attribution (`kernel_profile`, steps 12–13, via `scripts/kprof_agg.py`), with the
profiler's op-wrapper rows (`nccl:_*`, `section:*`, `FSDP::*`) excluded as double-counting:

| what | ms/step | note |
|---|---:|---|
| `triton_tem_fused_0` + `_zeros_1` | **1390.7** | flex-attention templates — 160 launches = 20 layers × 8 micro-steps |
| gemm (`nvjet_*`) | **1267.2** | `v_proj`, `o_proj`, FF, head |
| NCCL all-gather | **554.7** | **parameter** traffic — unaffected by O9 |
| NCCL reduce-scatter | **381.7** | **gradient** traffic — what O9 removes 7/8 of |
| optimizer (`multi_tensor_apply`) | ~164 | see the discrepancy below |
| **`embedding/bank`** (incl. fused) | **~50** | **the bank gradient path** |

#### The gate: the bank gradient/optimizer path is NOT a top-two term

Tier 2 (O1–O5) exists to stop FSDP2 and AdamW touching the 2.576B bank parameters
densely. Measured, that target is:

- **bank gradient path ≈ 50 ms/step ≈ 1.3%** of the step
- **optimizer ≈ 12–164 ms ≈ 0.3–4.3%** (instruments disagree, see below)
- **combined: at most ~5% of step time**

against attention templates at ~36%, GEMM at ~33% and NCCL at ~24%. The top two terms are
**attention compute** and **dense GEMM**; communication is third. The banks are **not**
in the top two — they are roughly 25× smaller than the smallest of them.

**So, per the Goal's instruction, Tier 2 does not proceed and Tier 3 (which is gated on
it) does not either.** This is the gate working as designed: it was put there precisely
to stop us building a 2.576B-parameter hand-managed optimizer path for a 1.3% term.

**Why pkmf's 1.43× does not transfer.** pkmf measured that speedup for the hand-managed
bank optimizer on a setup where the optimizer state was *not* sharded. Under FSDP2 each
rank already holds only 790M of the 3.163B parameters, and AdamW is bandwidth-bound, so
the work Tier 2 would remove has already been removed by sharding.

**One honest discrepancy, recorded rather than resolved.** `LoopTimers` puts `optim` at
12.3 ms (and `Optimizer.step#AdamW.step` agrees at 11.4 ms), while the `multi_tensor_apply`
kernels sum to ~164 ms. Those kernels are shared by AdamW, the foreach gradient clip and
FSDP2's own copies, so the 164 ms is an upper bound on the optimizer and the 12.3 ms a
lower one. **The conclusion is insensitive to which is right** — 4.3% is not a top-two
term either — but the gap should be closed before anyone quotes a single optimizer number.

**Known limit of this attribution:** ~52% of kernel time sits in `triton_tem_fused_*`
templates, which are compiled and do not resolve to a submodule. They are almost certainly
flex-attention forward and backward (launch count matches layers × micro-steps exactly),
but that is an inference from naming and arity, not a measurement. Closing it needs
inductor provenance tracking — see §T1.3's "still unavailable".

### Where AME's MFU actually goes — matched kernel comparison against dense

**Jobs 32579 (AME) and 32601 (dense H-balanced), the same run in every respect but
the attention type:** same node, 4×H100 SXM, FSDP2 + `MixedPrecisionPolicy(bf16, fp32)`,
`compile_blocks_full`, global batch 512×2048, ga 8 / micro-batch 16, profiler over steps
12–13.

The kernel tables already in [training_efficiency.md](../training_efficiency.md) §4.2 could
**not** serve as the dense side: they are one GPU, plain module + autocast, **no FSDP2**
("nothing in this table measures the FSDP2 path"), and §8 states every number there is
single-GPU. Subtracting them would have charged AME for the whole distributed path.

| | AME | dense | delta |
|---|---:|---:|---:|
| **ATTENTION** | **1390.7** (flex) | **506.9** (flash) | **+883.8** · **2.74×** |
| NCCL all-gather (params) | 554.7 | 185.4 | +369.3 |
| NCCL reduce-scatter (grads) | 381.7 | 124.3 | +257.3 |
| bank gather + gradient | 147.5 | 0 | +147.5 |
| other triton (fused glue) | 472.7 | 332.9 | +139.8 |
| FSDP2 param packing | 163.4 | 32.4 | +131.0 |
| optimizer (AdamW/foreach) | 163.6 | 36.5 | +127.1 |
| other eager | 193.9 | 79.2 | +114.7 |
| softmax / CE | 57.4 | 57.7 | −0.3 |
| **GEMM** (proj + FF + head) | **1264.4** | **1298.4** | **−34.0 · 0.97×** |
| **step wall** | **3842.6** | **2360.3** | **+1482.3 · 1.63×** |

Section timers agree: forward 1515.4 vs 741.9, backward 2308.6 vs 1610.3, optim 12.3 vs
5.0, clip 4.1 vs 0.8.

#### The headroom, ranked

| rank | item | excess | share of AME step |
|---|---|---:|---:|
| **1** | **attention: flex vs flash** | **+883.8 ms** | **23.0%** |
| **2** | **communication (NCCL)** | **+626.7 ms** | **16.3%** |
| 3 | bank gather + gradient | +147.5 ms | 3.8% |
| 4 | FSDP2 param packing | +131.0 ms | 3.4% |
| 5 | optimizer | +127.1 ms | 3.3% |
| — | GEMM | −34.0 ms | −0.9% (AME is *cheaper*) |

**GEMM is exonerated.** 0.97× — AME's architecture costs nothing extra in matmuls. The
Q/K projections it removes are worth about as much as the shapes it changes. Every
remaining point of the 17.6-MFU gap is attention, communication, or bank bookkeeping.

#### Lever 1 — AME pays 2.74× for attention to keep a capability it is not using

AME uses `flex_attention` rather than flash, and `src/core/ame.py` says exactly why:

> ALiBi slopes. Fixed geometric by default; learnable is an ablation, **and it is the
> reason the flex backend exists** — flash's `alibi_slopes` [takes a non-differentiable
> tensor]

But `learnable_slopes=False` is the **default**, no config overrides it, and every number
in this document was measured with the slopes as a **frozen buffer**. The differentiability
that flex is paid for is not being used. Flash's `alibi_slopes` argument computes the same
`−m_h·(i−j)` bias, and the dense twin shows what that costs: **506.9 ms against 1390.7**.

**Not a drop-in, and the obstacle is specific:** `sink=True` is also default. The learned
sink is kv column 0 with *zero* distance bias and unconditional visibility
(`score_bias`), whereas flash applies its ALiBi slope uniformly across keys. Replacing
flex with flash therefore needs the sink handled — a separate term, a flash build with
attention-sink support, or dropping the sink and measuring the quality cost. That is a
real engineering question, but it is bounded, and the prize is **23% of the step**.

#### Lever 2 — the banks are 81% of parameters and FSDP2 all-gathers all of them

NCCL is 3.02× the dense model's. The mechanism: FSDP2 all-gathers each sharded block's
parameters once per micro-step, and AME's blocks carry two `[50304 × 1280]` banks — 257 MB
each in bf16 — of which a micro-batch reads a few thousand distinct rows. That is
**41 GB of all-gather traffic per step** for the banks alone.

Note **all-gather (554.7) exceeds reduce-scatter (381.7)**: the larger half is *parameter*
traffic, which neither O9 nor Tier 2 touches. The lever is keeping the banks sharded and
gathering rows on demand — the mechanism of pkmf's O6/O7, i.e. Tier 3's machinery, but
aimed at the parameter all-gather rather than at the gradient reduce it was scoped for.

**A prediction I recorded beforehand and got wrong, which is informative.** I predicted
dense NCCL would be AME's divided by the 5.10× parameter ratio = 183.6 ms. Measured:
**309.7 ms**, a ratio of 3.02× not 5.10×. Collectives have a **latency floor** — both runs
issue the same 352 all-gathers and 176 reduce-scatters, only the payload differs, so the
dense model's smaller messages are latency-bound rather than bandwidth-bound. Consequence:
shrinking bank traffic will recover **less** than proportionally, and ~310 ms of the 936 ms
is a floor that no amount of sharding removes.

### Forward / backward split, every component (jobs 32623 AME, 32624 dense)

Produced by `scripts/kprof_phase.py` from the chrome traces, attributing each GPU kernel
to the CPU operator that launched it and asking whether that operator sits under an
autograd node. **Validated against independent ground truth:** dense flash is
name-identifiable, and the method returns 139.2 fwd / 366.0 bwd against the kernel names'
141.5 / 365.4 — within 1.6%.

| AME (step 3842.6 ms) | forward | backward | total |
|---|---:|---:|---:|
| **ATTENTION (flex)** | **730.4** | **661.0** | **1391.4** |
| GEMM | 359.2 | 910.1 | 1269.3 |
| NCCL all-gather | 269.3 | 276.6 | 545.8 |
| triton glue | 178.6 | 294.0 | 472.6 |
| NCCL reduce-scatter | 5.1 | 379.8 | 384.8 |
| optimizer | 65.8 | 98.6 | 164.4 |
| FSDP2 packing | 39.6 | 123.9 | 163.5 |
| bank / embedding | 30.5 | 121.9 | 152.5 |
| other eager | 5.2 | 102.8 | 108.0 |
| copy / memcpy | 4.1 | 80.8 | 84.9 |
| softmax / CE | 23.6 | 33.8 | 57.4 |
| **TOTAL** | **1711.4** | **3083.2** | **4794.7** |

| dense (step 2360.3 ms) | forward | backward | total |
|---|---:|---:|---:|
| GEMM | 403.5 | 887.9 | 1291.4 |
| **ATTENTION (flash)** | **139.2** | **366.0** | **505.2** |
| triton glue | 127.6 | 204.6 | 332.1 |
| NCCL all-gather | 58.8 | 94.8 | 153.6 |
| other eager | 5.3 | 60.4 | 65.7 |
| NCCL reduce-scatter | 5.2 | 57.6 | 62.9 |
| softmax / CE | 23.9 | 33.8 | 57.7 |
| optimizer | 19.0 | 17.8 | 36.7 |
| FSDP2 packing | 9.2 | 23.2 | 32.4 |
| copy / memcpy | 4.6 | 6.1 | 10.7 |
| bank / embedding | 3.1 | 1.4 | 4.5 |
| **TOTAL** | **799.4** | **1753.7** | **2553.1** |

#### The finding: it is flex's FORWARD that is anomalous

| | AME flex | dense flash | ratio |
|---|---:|---:|---:|
| attention **forward** | 730.4 | 139.2 | **5.25×** |
| attention **backward** | 661.0 | 366.0 | 1.81× |
| **bwd / fwd** | **0.91×** | **2.63×** | |

A normal attention backward costs ~2.5× its forward — dense's 2.63× is textbook. AME's is
**0.91×**, and not because its backward is cheap: its backward is 1.81× dense's. The
forward is the outlier at **5.25×**, and it alone is **+591 ms, 15.4% of the AME step**.
That is the single largest addressable item found anywhere in this work, and it is a
*forward-pass* cost — so activation checkpointing (cut in §0) would have made it worse,
not better.

Hypothesis for the spike, **not yet tested**: `flex_attention`'s forward evaluates
`score_mod` (the ALiBi term) per score block and consults the block mask, where flash
folds its causal masking into the kernel schedule and gets ALiBi for free from
`alibi_slopes`. `scripts/bench_attention.py` is written and would settle it by timing
`_attend_flex` against the SDPA flash floor at this exact shape.

#### Two other things the split shows

- **FSDP2 all-gathers the parameters twice.** AME's all-gather is 269.3 forward **plus
  276.6 backward** — the backward re-gathers what the forward already fetched and
  resharded. For a model whose parameters are 81% bank, that doubles the bank traffic, and
  it is why the all-gather (545.8) exceeds the reduce-scatter (384.8). A row-on-demand
  bank gather would save on both passes, not one.
- **GEMM is confirmed innocent in both passes.** AME 359.2/910.1 against dense
  403.5/887.9 — AME is *cheaper* in the forward and within 2.5% in the backward.

#### A method note worth keeping

Two timestamp-based attributions were tried before this one and both produced confident,
wrong tables. Keyed by thread id, backward came out as **0.0 ms everywhere** and 65% of
the step landed in "unattributed" — autograd runs on its own thread. Switched to global
time intervals, the dense attention split came out **327.6 fwd / 177.6 bwd**, inverted,
because the autograd thread lags into the next micro-step's forward window. Only operator
identity — is the launching op under an autograd node — agrees with the kernel names.
**The lesson is that the totals agreed in all three versions.** Only the split moved, so
nothing but an independent ground truth could have caught it.

### Tier 2 cannot land alone: it is memory-infeasible at h20 on 4×H100

Found 2026-11-04 while starting the Tier 2 build, **before writing any code**. Recorded
here because it changes the build order and nothing in §Tier 2 or §Tier 3 anticipates it.

**The enabler exists.** `torch.distributed.fsdp.fully_shard` in torch 2.7.1 takes
`ignored_params: Optional[set[nn.Parameter]]`, so the banks can be kept out of FSDP2's
gradient and optimizer path cleanly — no `requires_grad=False` trick, no custom
`autograd.Function` needed for O1. Route A of §Tier 2 becomes straightforward.

**But excluding them from FSDP2 also un-shards them**, and the arithmetic does not fit.
Bank parameters are 2 banks × 50304 × 1280 × 20 layers = **2.576B**. Per-rank bank state:

| | sharded by FSDP2 (today) | excluded from FSDP2 (Tier 2 alone) |
|---|---:|---:|
| bf16 parameter | 1.29 GB | 5.15 GB |
| fp32 master | 2.58 GB | 10.30 GB |
| fp32 gradient | 2.58 GB | 10.30 GB |
| Adam moments (2 × fp32) | 5.15 GB | 20.61 GB |
| **total per rank** | **11.6 GB** | **46.4 GB** |

B1 measured a 60.6 GiB peak with everything sharded. Replacing 11.6 GB of sharded bank
state with 46.4 GB of replicated state adds ~34.8 GB → **~95 GB per rank against an 80 GB
H100. It will OOM**, and not marginally.

**Consequence for the build order.** §0 states "Tier 2 is a hard prerequisite for Tier 3",
and that is true of the *code* — the controller must exist before anything can shard what
it owns. But the reverse is true of *runnability*: **Tier 2 is not executable at h20
without Tier 3's sharding**. The two must land together and be measured together. Tier 2
alone is only testable on a toy model, which is exactly what the §1.4.1 gate uses, so the
correctness work is unaffected — but no throughput number for "Tier 2 alone" is obtainable
at this model size.

**Suggested shape for the combined build**, as one module with `mem_parallel_size` as the
knob — `mp=1` is Tier 2 (toy-model testable, OOM at h20), `mp=4` is Tier 2+3:

1. `fully_shard(..., ignored_params=bank_params)` so FSDP2 never touches them (O1).
2. Column-shard each bank `[vocab, H·D]` across the memory-parallel group — a column slice
   is a **head subset**, and heads are independent until `o_proj` (§Tier 3).
3. Forward: all-gather token ids within the mp-group → local gather on owned columns →
   all-to-all so each rank ends with its own tokens at full width (O7).
4. Persistent fp32 `main_grad`, fp32 master, bf16 compute parameter (O1/O3).
5. One `all_reduce` in the **memory-DP subgroup** per optimizer step, divisor = **global
   `world_size`** (O2/O8) — because the forward's id all-gather already summed `mp` ranks'
   tokens into each slice, so `mp × (world/mp) = world`.
6. Separate clip over the masters, three hooks in the step loop (O4/O5).

At `world=4, mp=4` the mem-DP groups are singletons, so the bank all-reduce disappears
entirely and each rank holds a quarter of every bank — 11.6 GB again, the same as today,
but without FSDP2 all-gathering whole banks twice per micro-step (which the fwd/bwd split
above measured at 269.3 + 276.6 ms).

### Tiers 2 and 3 as built — one module, `mem_parallel_size` as the knob

Built 2026-11-04 on `ame_optimize`, after the gate was explicitly overridden. The
two tiers are one file, `src/core/memory_parallel.py`, because the memory finding
above makes them inseparable in practice: `mem_parallel_size=1` is Tier 2 alone
(correct, toy-testable, and it does not fit at h20) and `mem_parallel_size=4` is
Tier 2+3.

| Item | Where it lives | Note |
|---|---|---|
| O1 | `fully_shard(..., ignored_params=...)` in `src/core/distributed_training.py`; `ShardedBank.main_grad` | Persistent fp32 accumulator, never zeroed between micro-steps. No dense gradient is ever allocated. |
| O2 | `MemoryParallelController.reduce` | One `all_reduce(SUM)` per optimizer step, in the mem-DP subgroup. |
| O3 | `MemoryParallelController.step` | fp32 master → bf16 compute parameter after each step. |
| O4 | `Trainer.clip_gradient` + `controller.local_grad_norm_sq` / `clip_` | **Joint**, not separate — see the decision log. |
| O5 | `Trainer.train` | `materialize()` at step begin, `reduce()` after the accumulation loop and before the clip, `step()` beside the optimizer; `train/bank_lr` logged. |
| O6 | `MemoryParallelController._build_groups` | Contiguous mp-groups, strided dp-groups. Both loops are walked by every rank. |
| O7 | `_BankGather.forward` / `.backward` | id all-gather → local gather on owned columns → all-to-all; backward mirrors it. |
| O8 | `MemoryParallelController.reduce` | `div_(world_size)`, **not** `dp_size`. |

**Route B, taken without the standalone spike.** The plan asked for a spike to
choose between routes A, B and C before the full build. Route B was taken
directly, because the risk the register names — "`index_add_` on a 2.576B fp32
target" — is not the shape that occurs. The scatter target is one bank's slice,
`[50304, 320]` fp32 = 64 MB at `mem_parallel_size=4`, and there are 40 of them;
it is the same total work `embedding_dense_backward` already does, written to a
persistent buffer instead of a fresh one. Route A was never viable: keeping
`.grad` is exactly the allocation the memory arithmetic above rules out at
`mp=1`, and at `mp=4` it would duplicate `main_grad`. The in-situ measurement
(`embedding/bank` in the attribution twin) is a better number than a synthetic
benchmark would have been.

**The banks keep `requires_grad=True`.** O1 as written says to set it False so
FSDP2 builds no master, no reduce and no step. That cannot work here: autograd
only calls a `Function`'s backward if one of its inputs requires grad, and the
bank gradient has to *arrive* before it can be scattered. So the parameter stays
`requires_grad=True` and `_BankGather.backward` returns `None` for it. The effect
O1 wanted is reached by a different route — `.grad` is never materialised, and
`Trainer.clip_gradient` filters on `p.grad is not None`, so the banks drop out of
the backbone's clip without being named. FSDP2 and the backbone AdamW are kept
off them explicitly instead: `ignored_params` and
`parameter_groups(..., exclude=...)`.

#### The correctness gate (§1.4), and what it actually proved

`tests/test_memory_parallel.py`, 22 tests, all on CPU, no cluster needed. The toy
model is the one §1.4.1 specifies — `d_model` 64, 2 blocks, vocab 256, 3 steps —
built from the real `AMEAttention`, and run in **float64** so that a structural
error is orders of magnitude clear of the floating-point floor.

| Gate | Result |
|---|---|
| §1.4.1, clipping off | **Bit-identical.** Every parameter, every step, exactly equal to the ordinary FSDP2/AdamW path. |
| §1.4.1, clipping on | Agrees to 1e-7. The difference is **exactly one fp32 ulp** in the norm scalar — see below. |
| `main_grad` vs `embedding_dense_backward` | Bit-identical (atol 1e-12 asserted; measured 0). |
| `weight.grad` | None after a full run, i.e. no dense gradient was allocated. |
| MFU denominator | Unchanged by the swap. |
| O8 divisor, `mp=1` vs `mp=2` and `mp=4` | Gradients agree to 1e-14; losses to 1e-12; masters to 1e-12 after 3 steps. |
| O8 negative control | Dividing by `dp_size` shows up as exactly 4.0x, so the gate has teeth. |
| Column split (O7) | The sharded gather is `torch.equal` to the unsharded one. |
| Separate-clip negative control | Moves the model by >1e-4, i.e. far outside the gate above. |
| Checkpoint round-trip at `mp=1` and `mp=4` | Restored model bit-identical; skipping the sidecar is >1e-3 different. |

**The one-ulp finding, recorded because it looks like a bug and is not.**
`Trainer.clip_gradient` computes the global norm and then casts it with
`.float()`. The baseline sums one list of per-parameter norms; the joint clip
sums the backbone's list and adds the banks' separately. In float64 the two sums
agree to ~1e-16, but the fp32 cast quantises the result and they land on
neighbouring fp32 values: 1.3832852840 against 1.3832854032 at step 1, identical
at every later step. The parameters then carry that one-ulp difference forward at
~1e-7. This is float associativity in a scalar, not a difference in the function
being computed — which the clipping-off arm settles by being exact.

#### End-to-end through `main.py`, before any cluster time

`tiny_local` + `common.attention.type=ame`, 6 steps, ga 2, with and without
`trainer.mem_parallel_size=1`:

- `train/loss` **bit-identical at all 6 steps** (11.344449996948242, 11.08735466,
  10.554288864, 11.291622162, 10.914241791, 11.235111237).
- `train/grad_norm` identical at steps 1-4; steps 0 and 5 differ in the 7th
  significant figure (6.742344856 vs 6.742345333) — the same one-ulp effect.
- `train/bank_lr` tracks the cosine-with-warmup schedule the backbone is on.

That exercises the real wiring: the controller built before compile and before
`fully_shard`, the banks excluded from `parameter_groups`, the three step-loop
hooks, and the joint clip.

### Tier 2+3 measured: mathematically right, throughput-neutral, slightly better gradients

Job **32632**, `ame/h20_4gpu_mb16_mp4`, against **B1** (job 32576). Same global
token schedule, same seed, same everything but `trainer.mem_parallel_size`.

| | B1 (32576) | T2+3 mp=4 (32632) | delta |
|---|---:|---:|---:|
| MFU / GPU | 26.82% | **26.54%** | **-1.03%** |
| tokens/s (global) | 281,541 | 278,636 | -1.03% |
| step time | 3.723 s | 3.761 s | +38 ms |
| peak GiB / GPU | 60.6 | **61.91** | +1.3 |
| FLOPs/token | 3.768 G | 3.768 G | unchanged |

**The headline is that it is not a win.** -1.03% MFU, within a percent of the
baseline, after removing the bank parameter all-gather entirely (measured at
269.3 ms forward + 276.6 ms backward in Tier 1) and deleting the bank gradient
reduce-scatter outright (`dp_size=1`, so `reduce()` does no collective at all).
Something costs what those saved. The attribution twin (job 32634) is what says
what; the candidates, in the order the Tier 1 evidence ranks them:

1. **The all-to-all is activation-sized, and collectives have a latency floor.**
   Tier 1's own lesson, from predicting dense NCCL at 183.6 ms and measuring
   309.7 ms: the number of collectives matters as much as their payload.

   Collectives per optimizer step, counted exactly:

   | | calls/step |
   |---|---:|
   | FSDP2 parameter all-gather (what this replaced) | 22 modules x 8 micro x 2 = **352** |
   | id all-gather (forward only; backward reuses the saved ids) | 40 banks x 8 = 320 |
   | all-to-all | 40 x 8 x 2 = 640 |
   | **Tier 2+3 total** | **960** |

   **2.7x more collective calls**, each smaller: the all-to-all moves
   3/4 x 32768 x 1280 bf16 = 62.9 MB against the all-gather's
   3/4 x 50304 x 1280 bf16 = 96.6 MB. Payload down 0.65x, call count up 2.7x --
   and Tier 1 already measured that call count is what dominates (dense NCCL
   predicted at 183.6 ms from payload, measured at 309.7 ms).

   **Two of those calls are avoidable and the fix is small.** The token ids are
   *identical* for all 40 banks within a micro-step, so the id all-gather can be
   hoisted out of `ShardedBank` and done once per micro-step: 320 -> 8. And
   `q_bank` and `k_bank` of one layer are gathered back to back on the same ids,
   so their all-to-alls can be one call on the concatenation: 640 -> 320. Total
   **960 -> 328**, below the 352 the scheme replaced. Neither changes the
   mathematics, so the gates above cover them unchanged.
2. **The graph break.** `ShardedBank.forward` is `@torch.compiler.disable`d, so
   each of the 20 compiled blocks is now three graphs instead of one.
3. **`index_add_` against `embedding_dense_backward`** -- same work, different
   kernel, and the new one writes fp32 where the old one wrote bf16.

**Memory came out where the arithmetic said.** 61.91 GiB against 60.6: the
+1.3 GiB is the persistent bf16 compute shard (1.29 GB). FSDP2 keeps an fp32
sharded parameter and materialises bf16 transiently; this keeps both. Per-rank
bank state is 0.644B parameters -- exactly 2.576B / 4, as logged by the
controller on every rank.

#### Correctness on the cluster (§1.4.2 and §1.4.3)

| step range | `train/loss` | `train/grad_norm` |
|---|---|---|
| 0-27 | **identical**, every step | relative 1e-7 growing smoothly to 3e-3 |
| 28-29 | differ by 0.0625 and 0.03125 | — |
| 17 | — | 2.623 vs 4.476, a shared spike reached at different heights |
| 18-29 | — | 10-25% relative, decaying together |

The logged loss is bf16, whose ulp on [4, 8) is `2^2 x 2^-7 = 0.03125`. So the
two runs are **two representable values apart at step 28 and one at step 29**,
after twenty-eight steps of landing on exactly the same value.
The grad norm is the far more sensitive quantity and it diverges the way a
chaotic system does — smoothly to 0.3% through step 16, then a shared gradient
spike at step 17 (both runs spike; B1 to 2.62, mp4 to 4.48, against neighbours
near 1.0) amplifies it. Both are clipped at 1.0 there, so what reaches the
parameters differs in direction, not scale.

The float64 toy-model gate settles what this divergence is *not*: with clipping
off the two paths are bit-identical, so the function being computed is the same.

**O5's third hook checks out on real hardware too.** The banks are stepped by
their own AdamW, so "the learning rate" stops being one number by construction.
Over job 32632's 30 steps `train/bank_lr` equals `train/lr` **exactly, at every
step**, through the 10-step warmup and the cosine decay that follows
(2.44140625e-05 at step 0, 2.44140625e-04 at step 10, 2.59124545e-05 at step 29).
Copying the LR off `optimizer.param_groups[0]` before `scheduler.step()` is what
makes that hold; reading it from the scheduler instead would have put the banks
one step ahead of the backbone for the whole run.

#### The gradients are not just different, they are better

Mechanism, and it is measurable without a GPU
(`test_an_fp32_accumulator_beats_a_bf16_one_on_duplicate_token_ids`):

FSDP2 holds the unsharded bank parameter in bf16, so autograd hands
`embedding_dense_backward` a **bf16** buffer to scatter into. A micro-batch of
32768 tokens over a 50304-row vocabulary hits the commonest rows hundreds or
thousands of times, and every one of those adds rounds to bf16's 8 significant
bits. Tier 2's `main_grad` is fp32 from the first add.

On a Zipf-ish draw with a maximum duplication of 2778 hits, relative error on the
hot rows against an exact fp64 sum:

| accumulator | relative error |
|---|---:|
| bf16 (what FSDP2's path does) | 1.99e-3 |
| fp32 (`main_grad`) | 6.35e-8 |

A factor of **31,000**. And 2e-3 is the same order as the 3e-3 relative grad-norm
divergence measured at step 16 — consistent with this being the dominant cause,
though a single 30-step pair cannot prove it on its own. The direction is the
reassuring one: the hand-managed path rounds *less* than the path it replaces.

### The attribution: Tier 2+3 is a large communication win the workload cannot cash

Job **32634** (`mp4_attrib`) against job **32579** (the Tier 1 attribution twin).
Identical in every respect but `mem_parallel_size`, both with `profile_timers`
and `kernel_profile`, both read with `scripts/kprof_agg.py`.

| kernel category | Tier 1 (32579) | Tier 2+3 (32634) | delta |
|---|---:|---:|---:|
| **comm(nccl)** | **2128.3 ms** / 1766 launches | **914.1 ms** / 3686 launches | **-1214.2 ms (-57%)**, +109% launches |
| compiled(triton) | 2011.6 / 8000 | 1820.2 / 7200 | -191.4 |
| gemm | 1269.3 / 2744 | 1152.3 / 2744 | -117.0 |
| optimizer | 176.2 / 1467 | 82.4 / 953 | -93.8 |
| **eager elementwise** | 179.3 / 3189 | **282.0** / 4235 | **+102.7** |
| **embedding/bank** | 10.2 / 55 | **96.0** / 375 | **+85.8** |
| other | 1712.6 / 2700 | 1708.9 / 3062 | -3.7 |
| **total busy** | **7552.8** / 20566 | **6121.4** / 23019 | **-1431.4 ms (-19%)** |

Section timers, rank 0, steady state:

| section | Tier 1 (32579) | Tier 2+3 (32634) |
|---|---:|---:|
| forward | 1515.4 | 1512.1 |
| backward | 2308.6 | 2224.3 |
| clip | 4.1 | 4.7 |
| optim | 12.3 | 5.2 |
| **bank_reduce** | — | **1.8** |
| **bank_step** | — | **20.7** |

**Nineteen percent of the GPU's kernel time disappeared and the step got
longer.** That is the whole result, and it has one explanation: **the
communication Tier 3 deleted was never on the critical path.** NCCL's 2128 ms in
Tier 1 sits *inside* forward + backward's 3824 ms, because FSDP2 prefetches its
all-gathers and overlaps them behind compute. Deleting overlapped work frees the
copy engines, not the clock.

What *did* land on the clock is the work that runs on the compute stream and
overlaps with nothing:

- **eager elementwise +102.7 ms** — the permute/reshape/contiguous shuffling
  around the all-to-all, 1046 more launches per step;
- **embedding/bank +85.8 ms** — `index_add_` into fp32 `main_grad`, 320 more
  launches, writing 4 bytes per element where `embedding_dense_backward` wrote 2.

That is +188.5 ms of unhidden work against a measured +38 ms step. The rest is
paid for by the -93.8 ms the optimizer gave back and by backward shortening by
84 ms.

**And the hand-managed optimizer path did exactly what Tier 2 promised.**
`bank_reduce` is **1.8 ms** — at `mem_parallel_size=4` the memory-DP subgroups
are singletons, so the bank gradient all-reduce does not happen at all.
`bank_step` is **20.7 ms** for AdamW over 644M parameters per rank. Against Tier
1's bank gradient + optimizer term of 50-311 ms (the instruments disagreed, see
the Tier 1 record), that is the win the tier was designed for — and it is 0.5%
of the step, which is what the Tier 1 gate said when it fired NEGATIVE.

#### What this means, stated plainly

**AME at h20 on 4xH100 SXM is compute-bound, and NVLink is fast enough to hide
every byte of bank traffic behind flex attention.** Tier 2+3 removes 57% of the
communication and 19% of all kernel time, and buys **-1.03%**. The machinery is
correct, it is memory-neutral (+1.3 GiB), and on this hardware it has nothing to
sell.

**It rhymes exactly with Tier 1's O9 result**, which was -2.53% on entropy and
**+3.33% on lem** — the same optimisation, opposite signs, decided by the
interconnect. The prediction this makes is specific and testable: **Tier 2+3
should be a win on lem**, where four GPUs sit one per node behind a slower
fabric and communication is *not* hidden. That is the measurement this result
asks for next, and it is outside this goal's fixed entropy test bed.

**Run-to-run noise is not the explanation.** Step-time standard deviation over
the last ten steps is 0.0059 s for B1 and 0.0042 s for mp4, against a difference
of 0.039 s — about seven times the noise. (The profiled runs are a different
matter: 32579's sd is 0.319 s, so profiled step totals must not be compared
against each other, and the -1.03% above is taken from the unprofiled pair.)

### The `mem_parallel_size` sweep: only `mp = world_size` fits at h20

Tier 3's Measurement asks for `mem_parallel_size` in {1, 2, 4} at 4 GPUs. Two of
the three points do not run, and that is the result.

| mp | bank state / rank | predicted peak | measured | job |
|---:|---:|---:|---|---|
| 1 | 46.4 GB | ~95 GiB | **OOM at 79.16 / 79.18 GiB**, after one step's forward+backward | 32649 |
| 2 | 23.2 GB | ~72 GiB | **OOM at 76.56 / 79.18 GiB** | 32638 |
| 4 | 11.6 GB | ~61 GiB | **61.91 GiB** ✅ | 32632 |

At `mp=4` the prediction was right to 1.3 GiB. At `mp=2` it was 4.5 GiB
optimistic, and the allocator says why in its own error message: **4.52 GiB
reserved by PyTorch but unallocated** — fragmentation, not resident state. The
run died trying to allocate 3.07 GiB with 2.61 GiB free.

`mp=1` got **further than predicted**, and the reason is worth knowing: AdamW
allocates `exp_avg` and `exp_avg_sq` lazily, on the first step. So the run
completed a whole forward and backward at 2.576B unsharded bank parameters per
rank — the controller logged exactly that on every rank — and then died when the
bank optimizer reached for its 20.6 GB of moments, with 8 MiB free and 4.68 GiB
lost to fragmentation. All four ranks died at the same allocation. **This is the
"Tier 2 cannot land alone" arithmetic confirmed by measurement**, and it confirms
it in the sharpest possible way: the forward and backward fit, and the optimizer
state is what does not.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is the obvious unblock and it
was **not** tried, deliberately: it changes the allocator under a measurement
whose entire purpose is to be compared against B1 and `mp=4`, which ran without
it. A throughput number obtained that way would not belong in the same table.
If the `mp=2` point is wanted later, the honest way to get it is to re-run B1
and `mp=4` with the same allocator setting.

**The practical consequence is simple: at h20 on 80 GB cards, the only usable
setting is `mem_parallel_size == world_size`.** Anything less keeps more than one
copy of a 2.576B-parameter bank's optimizer state per rank, and there is no room
for it. That also means the `mp=1` vs `mp=4` comparison, which is the one that
would separate "sharding helped" from "deleting the all-reduce helped", is only
available on the toy model — where
`test_o8_divisor_is_the_global_world_size` already runs it at `mp` in {1, 2, 4}
and confirms the mathematics is identical across all three.

### The 1-GPU point, and a prediction stated before the measurement

Tier 2's Measurement asks for "one 1-GPU run to compare directly against pkmf's
1.43x claim on a shape where no NCCL traffic confounds it"
(`configs/ame/h20_1gpu_mb4_mp1.yaml`, against `h20_smoke.yaml` = job 32360:
47,391 tokens/s, 18.1% MFU, 22.1 s/step at ga 128 / micro-batch 4).

At world_size 1 there is nothing left but the mechanism: `_build_groups` returns
`(None, None)`, `reduce()` does no collective and divides by 1, and the bank
forward is a plain `F.embedding`. What Tier 2 removes there is one thing — the
**dense gradient materialised once per micro-step**.

The arithmetic, written down before the run so the result can be read against it:

| | per optimizer step |
|---|---:|
| baseline: `embedding_dense_backward` produces a dense bf16 `[50304, 1280]` (128.8 MB) per bank, then `grad += ` reads and writes it again | 40 banks x 128 micro x 386 MB = **1.98 TB** |
| Tier 2: `index_add_` touches at most 8192 rows x 1280 fp32, read-modify-write | 40 x 128 x 83.9 MB = **430 GB** |
| at H100 HBM3 ~3.35 TB/s | **~590 ms -> ~128 ms** |

**Predicted: about 460 ms off a 22.1 s step, i.e. ~2%.** ...

#### The measurement, job 32652: **1.385x**. The prediction was wrong by 20x.

| | baseline (32360) | Tier 2, `mp=1` (32652) | ratio |
|---|---:|---:|---:|
| tokens/s | 47,391 | **65,651** | **1.385x** |
| step time | 22.1 s | **15.966 s** (sd 0.152) | **1.384x** |
| MFU / GPU | 18.1% | **25.01%** | **1.382x** |
| peak GiB | ~57 | 64.31 | +7 |

Three independent quantities agree at 1.38x, and the loss trajectory confirms
the runs are comparable: 32360 is recorded as **11.25 -> 7.28** over 30 steps
and 32652 measured **11.25 -> 7.28**. **pkmf's 1.43x transfers to AME almost
exactly.**

**Why the prediction was 20x too small, and it is the most useful thing this
section produced.** The estimate counted only the dense gradient's HBM traffic.
What it left out is everything else FSDP2 does **per micro-step**, which at
`ga=128` happens 128 times per optimizer step:

- the dtype-converting "all-gather" of 2.576B bank parameters from the fp32
  shard to a bf16 working copy — read 4 B, write 2 B per element — **once in the
  forward and again in the backward**;
- the reduce-scatter of the bank gradients back to fp32;
- `embedding_dense_backward` writing a full `[50304, 1280]` tensor per bank, for
  a micro-batch that touches at most 8192 rows of it, then `grad +=` reading and
  writing that tensor again.

Roughly 18 bytes per bank parameter per micro-step for the parameter traffic
alone: 46 GB x 128 = **5.9 TB per optimizer step**, about 2 s at H100 bandwidth,
before the gradient pipeline and its 5,120 kernel launches. Tier 2 deletes all of
it: the banks are outside FSDP2, the compute copy is persistent, and the gradient
is a sparse `index_add_` into a buffer that is never zeroed.

#### This explains the 4-GPU result, and makes Tier 2's value a function of `ga`

Everything Tier 2 removes is **per-micro-step** work, so what it is worth scales
with `gradient_accumulation_steps`:

| | 1 GPU | 4 GPUs |
|---|---:|---:|
| `gradient_accumulation_steps` | 128 | 8 |
| per-micro-step bank work, relative | 1x | **1/4** (FSDP2 shards it 4 ways) |
| micro-steps per optimizer step | 128 | 8 |
| **product** | **128** | **2** |
| measured | **+38.5%** | **-1.0%** |

**A 64-fold difference in how much there is to delete.** That, not the
interconnect, is the first-order explanation of the two results — and it sits
*underneath* the communication finding rather than replacing it: at `ga=8` what
little remains to save is also the part NVLink was already hiding.

**The testable consequence:** at 4 GPUs, Tier 2+3's delta should improve
monotonically as the micro-batch falls and `ga` rises. `configs/ame/h20_4gpu_mb4`
(ga 32) and `h20_4gpu_mb8` (ga 16) already exist from the deferred B1 sweep, so
the curve is four runs away.

**Memory went the predicted direction but further**: 64.31 GiB against the
baseline's ~57. The bank's resident state rises from 41.2 GB (fp32 param + fp32
grad + Adam) to 46.4 GB (bf16 compute + fp32 master + fp32 `main_grad` + Adam),
and FSDP2's transient bf16 copies stop being allocated — but the transients were
freed between micro-steps anyway, so the resident increase is what shows.

### The ga-scaling curve: Tier 2+3's value is a function of the accumulation count

The 1-GPU result predicted this and jobs **32656** (baseline) and **32659**
(Tier 2+3) tested it, submitted back to back on the same node within one hour.

| run | GPUs | ga | micro-batch | MFU/GPU | tokens/s | step | delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 | 4 | 8 | 16 | 26.82% | 281,541 | 3.723 s | — |
| T2+3 `mp=4` | 4 | 8 | 16 | 26.54% | 278,636 | 3.761 s | **-1.0%** |
| B1/mb4 | 4 | 32 | 4 | 18.30% | 192,092 | 5.456 s | — |
| **T2+3 `mp=4`/mb4** | 4 | **32** | 4 | **22.42%** | **235,421** | **4.452 s** | **+22.6%** |
| baseline 1 GPU | 1 | 128 | 4 | 18.1% | 47,391 | 22.1 s | — |
| **T2 `mp=1`, 1 GPU** | 1 | **128** | 4 | **25.01%** | **65,651** | **15.966 s** | **+38.5%** |

**Monotone in `ga`, and the mechanism predicts the ordering exactly.** Everything
Tier 2 deletes is per-micro-step work, scaled down by however many ways FSDP2 has
already sharded it:

| | `ga` | FSDP2 shard | product | measured |
|---|---:|---:|---:|---:|
| 4 GPU, mb 16 | 8 | 1/4 | **2** | **-1.0%** |
| 4 GPU, mb 4 | 32 | 1/4 | **8** | **+22.6%** |
| 1 GPU, mb 4 | 128 | 1/1 | **128** | **+38.5%** |

**What this changes about how the whole exercise reads.** The goal's fixed test
bed — 4 GPUs at ga 8 / micro-batch 16 — is the **only** configuration measured in
which Tier 2+3 does not help, and it is the one with the least per-micro-step
work in existence: the largest micro-batch that fits, on the fastest
interconnect, with FSDP2 already sharding four ways. Everywhere else the same
code is worth between +22.6% and +38.5%.

**And it makes small micro-batches affordable, which is the practical point.**
At micro-batch 4 the baseline pays 18.30% against micro-batch 16's 26.82% — a
1.466x penalty for the smaller split. Tier 2+3 recovers most of it, taking
micro-batch 4 to 22.42%. That matters because small micro-batches are not a
choice at larger models; they are what memory forces. Peak was 25.29 GiB against
the baseline's 24.06 — the same +1.2 GiB the micro-batch-16 pair showed.

**Correctness held at this shape too:** the loss is identical at 19 of 30 steps
and the largest divergence is 0.0938, three bf16 ulps at a loss near 7.5 — the
same picture as the micro-batch-16 pair, one representable value at a time.

### Tier 2 and Tier 3 exit criteria

**Tier 2 (O1-O5)**

| gate | status |
|---|---|
| CPU toy-model equivalence (§1.4.1), parameters identical after 3 steps | ✅ **bit-identical** with clipping off; one fp32 ulp with it on, explained |
| 30-step loss trajectory against B1, stated tolerance, grad-norm traces | ✅ job 32632: loss identical for 28/30 steps, then 2 and 1 bf16 ulp; grad-norm traces overlaid above |
| Throughput delta at 4 GPUs, micro-batch held | ✅ **-1.03%** at mb 16 / ga 8 (26.54% vs 26.82%); **+22.56%** at mb 4 / ga 32 (22.42% vs 18.30%) |
| Throughput delta at 1 GPU | ✅ **+38.5% (1.385x)**, job 32652 — pkmf's 1.43x transfers |

**Tier 3 (O6-O8)**

| gate | status |
|---|---|
| Loss trajectory coincides at `mp` in {1, 2, 4} | ✅ on the toy model at all three (1e-12); on the cluster `mp` 1 and 2 do not fit — recorded as the result |
| **Scope of these gates after D2/D3** | ✅ all four are **correctness** gates and all four run on CPU, so they are independent of which shape fits on an H100. They were re-run on the fused path and on DDP, and `test_toy_model_parameter_equivalence_*` now **asserts** it is exercising the fused path rather than assuming the default. The DDP micro-batch-16 OOM (job 32672) is a **capacity** result, not a gate failure: the same code passes every gate above on a four-rank gloo mesh at that `mem_parallel_size`, and `test_ddp_backbone_with_memory_parallel_banks` covers the DDP path specifically. |
| **O8 divisor asserted by unit test, `mp=1` against `mp=4`** | ✅ `test_o8_divisor_is_the_global_world_size`, gradients agree to 1e-14, with a negative control that catches the wrong divisor as exactly 4.0x |
| Checkpoint save/load at `mp=4` round-trips to an evaluable model | ✅ `test_checkpoint_round_trips_at_every_mem_parallel_size`, restored model bit-identical, with a negative control |
| Throughput table across `mp` and micro-batch | ✅ the sweep table above; micro-batch held at 16 per the goal's fixed test bed |

**The honest summary.** Both tiers are built and both pass every correctness gate
the plan asks for. What they are worth is a function of how much per-micro-step
work there is to delete, and it is monotone:

| | `ga` | measured |
|---|---:|---:|
| 1 GPU, micro-batch 4 | 128 | **+38.5%** (1.385x, matching pkmf's 1.43x) |
| 4 GPUs, micro-batch 4 | 32 | **+22.6%** |
| 4 GPUs, micro-batch 16 | 8 | **-1.0%** |

The goal's fixed test bed is the last row, and it is the **only** configuration
measured in which these tiers do not pay: the largest micro-batch that fits, on
the fastest interconnect, with FSDP2 already sharding four ways. Read on its own
it says "Tier 2+3 is worth -1%". Read against the other two rows it says
something quite different and more useful — **the mechanism works, and what it
is worth is set by the accumulation count**. Since small micro-batches are what
memory forces at larger models, that is the regime this machinery was built
for.

### What is left on the table, with the arithmetic

Not done, because none of it is O1-O8 and the goal's scope is those items. Listed
with enough detail to be picked up without re-deriving anything.

**1. Merge `q_bank` and `k_bank` into one collective per layer.** They are gathered
back to back on the *same* ids and their results are concatenated into the same
`[B, S, 2 x H x D]` shape of work. One all-to-all on the concatenation halves
both the NCCL launches (640 -> 320) and, more importantly, the eager
permute/reshape/contiguous shuffling around them — and that shuffling is the
`+102.7 ms` of `eager elementwise` which the attribution shows IS on the critical
path. Expected: most of that 102.7 ms, against a 38 ms deficit.

**2. Hoist the id all-gather out of `ShardedBank`.** The token ids are identical
for all 40 banks within a micro-step; gathering them 40 times is 320 NCCL calls
where 8 suffice. It also saves memory that is easy to miss: `_BankGather` saves
the gathered ids for its backward, so 320 copies of a `[mp x B x S]` int64
tensor — **about 320 MB at h20** — are held live through the backward where one
shared copy would do.

Together these take the collective count from 960/step to 328, below the 352 the
scheme replaced. Neither changes the mathematics, so
`tests/test_memory_parallel.py` covers them unchanged — in particular
`test_column_shard_reassembles_the_full_width_gather`, which is exactly the test
a wrong concatenation order would fail.

**3. The ceiling on all of it is low on entropy, and that is the point.** Even a
perfect collective merge leaves `index_add_`'s `+85.8 ms` (fp32 writes where the
baseline wrote bf16 — the price of the better gradient), and the communication it
would save is already hidden behind flex attention. Break-even is the realistic
target here. **The measurement that would actually settle Tier 2+3's value is the
lem twin**, by exact analogy with O9's -2.53% on entropy and +3.33% on lem.

**4. `scripts/bench_attention.py` is still unrun** (written in Tier 1). Flex's
5.25x forward spike is 15.4% of the step and remains the largest single lever
this work has found — larger than everything in Tiers 2 and 3 put together.

### Raw material added by Tiers 2 and 3

On the cluster, re-readable without re-running anything
(`python3 scripts/kprof_agg.py <log>`, `python3 scripts/run_summary.py <log>`):

- Tier 2+3 attribution: `/home/j321m/llmrandom_cemetery/ame_h20_4gpu_mb16_mp4_attrib_2026-11-04_22-46-38/` — slurm log plus `kprof_trace_rank0.json`
- Tier 2+3 headline: `.../ame_h20_4gpu_mb16_mp4_2026-11-04_22-40-27/`
- the two OOMs: `.../ame_h20_4gpu_mb16_mp2_*` and `.../ame_h20_4gpu_mb16_mp1_*`
- 1 GPU: `.../ame_h20_1gpu_mb4_mp1_*`

Reach the cluster as **`entropy_j`**, not `entropy` — see the decision log.

### Where the raw material lives, for whoever picks this up

Chrome traces, re-analysable without re-running anything (`scripts/kprof_phase.py <path>`):

- AME: `/home/j321m/llmrandom_cemetery/ame_h20_4gpu_mb16_attrib_2026-11-04_21-42-58/kprof_trace_rank0.json` (93.6 MB)
- dense: `/home/j321m/llmrandom_cemetery/mfu_h20_hbal_4gpu_mb16_attrib_2026-11-04_21-43-21/kprof_trace_rank0.json` (74.8 MB)

Analyse them **on the cluster** — they are 170 MB together and the script only needs
stdlib, so the login node's `python3` is enough.

Unrun and ready: `scripts/bench_attention.py` times AME's real `_attend_flex` forward and
backward against the SDPA flash floor at the h20 shape. It is the direct test of whether
flex's 5.25× forward spike is `score_mod` + block-mask overhead — the largest single lever
this work found.

---

## 3b. Directive — 2026-11-04, from Maciej, after the Tier 2+3 results

Added to the active goal. Two observations and two things to build:

1. **"Memory parallelism should give the biggest speedup of the methods we
   tested, because it divides the backward calculation of the memory bank
   between GPUs."** The measured -1.0% at the pinned test bed contradicts that.
   **Investigate and fix.**
2. **"If it does not help, try running the core model in DDP and parallelise just
   the memory banks, the way the pkmf framework trained their memory models. It
   should be the fastest option."**

### Why observation 1 and the measurement can both be right — the baseline is the issue

Stated before investigating, so the investigation can confirm or refute it.

**pkmf's baseline is DDP. Ours is FSDP2.** Against a DDP baseline, memory
parallelism divides three things by `mem_parallel_size`:

| | DDP baseline | + memory parallelism |
|---|---|---|
| bank gradient buffer | 2.576B per rank | 644M per rank |
| bank optimizer step | 2.576B per rank | 644M per rank |
| bank gradient all-reduce | 2.576B, every step | **none** (`dp_size=1`) |

That is pkmf's ~2.2x, and it is real. But **FSDP2 already does all three**: it
shards gradients and optimizer state `world_size` ways and reduce-scatters
instead of all-reducing. So measured against FSDP2 the only thing left for Tier 3
to remove is the **parameter all-gather** — and the attribution showed that was
already overlapped behind flex attention.

**The per-rank backward compute is not divided by either scheme, and this is the
part of observation 1 that needs care.** Under FSDP2 each rank scatters its own
`n` tokens across the full width: `n x H x D` element-updates. Under memory
parallelism each rank scatters `mp x n` tokens across `H x D / mp`:

    FSDP2            n x (H·D)        = 32768 x 1280 = 41.9M updates/rank
    mem-parallel  (mp·n) x (H·D/mp)   = 131072 x 320 = 41.9M updates/rank

Identical. `mp` times as many rows, `1/mp` of the columns. What memory
parallelism divides is the **dense gradient buffer** (50304 x 1280 -> 50304 x 320)
and the **optimizer state**, not the scatter itself. The scatter is redistributed,
not reduced — and redistributing it costs an all-to-all.

**So the prediction is that observation 2 is the one that pays**, and for the
reason observation 1 gives: the division is real, but it is only visible against a
baseline that is not already doing it. That is DDP.

### What gets built and measured

- **D1 — decompose the 914 ms of NCCL in job 32634** into backbone all-gather,
  bank all-to-all and id all-gather, from the per-kernel dump. This says how much
  of the current cost is FSDP2's backbone traffic, which DDP would delete.
- **D2 — cut the unhidden cost the attribution found**: +102.7 ms eager
  elementwise (the permute/reshape around the all-to-all) and +85.8 ms
  `index_add_`. The q||k merge and the hoisted id all-gather are already
  specified in "What is left on the table".
- **D3 — DDP backbone + memory-parallel banks.** A `ddp` branch in
  `setup_distributed_training`, the banks staying with the controller. Memory is
  the question: the 587M backbone replicated at fp32 costs ~9.4 GB/rank against
  FSDP2's ~2.35 GB, so ~+7 GB on top of B1's 60.6 GiB at micro-batch 16. It may
  only fit at a smaller micro-batch — which the ga-scaling curve says is the
  regime where this machinery pays anyway.

### D1 result — the defect is the graph break, not memory parallelism

Investigated 2026-11-04 on the directive above, from the per-kernel dumps of jobs
32579 (Tier 1) and 32634 (Tier 2+3). **An instrumentation bug had to be fixed
first:** `index_add_` on CUDA dispatches to `indexFuncLargeIndex`, which matches
neither `index_add` nor `embedding`, so 93.8 ms/step of the Tier 2 scatter was
being filed under `other`. `scripts/kprof_agg.py` now matches `indexFunc`, and
the corrected numbers are below.

**NCCL, with the profiler's op-wrapper rows removed:**

| | Tier 1 (32579) | Tier 2+3 (32634) |
|---|---:|---:|
| parameter all-gather | 545.8 ms / 352 | — |
| backbone all-gather + **id all-gather** | — | 196.3 ms / **672** |
| bank all-to-all (`SendRecv`) | — | 164.0 ms / 640 |
| reduce-scatter | 384.8 ms / 176 | 58.7 ms / 176 |
| **real total** | **930.6 ms** | **419.0 ms** |

So Tier 3 did cut communication by **55%**. Of what is left, ~95 ms is the **id
all-gather's pure latency** — 320 launches moving 262 KB each, which is the fix
already specified in "What is left on the table".

**And here is the actual regression.** The bank path, correctly categorised:

| | Tier 1 | Tier 2+3 | delta |
|---|---:|---:|---:|
| `embedding/bank` | **10.2 ms / 55 launches** | **189.7 ms / 695 launches** | **+179.5 ms** |
| of which forward gather (`indexSelectLargeIndex`) | 3.2 / 8 | 90.8 / 328 | +87.6 |
| of which backward scatter (`indexFuncLargeIndex`) | — | 93.8 / 320 | +93.8 |
| `compiled(triton)` | 2011.6 | 1820.2 | **-191.4** |
| `eager elementwise` | 179.3 | 282.0 | **+102.7** |

**In Tier 1 the bank gather and its backward were inside the compiled region**,
fused by inductor into `triton_poi_fused_embedding_*` — which is why
`compiled(triton)` falls by 191.4 ms at exactly the moment `embedding/bank` rises
by 179.5 ms. The work did not get cheaper or more expensive in principle; it
moved out of Triton and into eager kernels, and eager it costs ~5.6x its
bandwidth bound (90.8 ms for 168 MB of traffic that should take 0.05 ms/launch).
The `+102.7 ms` of eager elementwise is the permute/reshape around the all-to-all,
unfused for the same reason.

**The cause is one line of mine:** `@torch.compiler.disable` on
`ShardedBank.forward`. Measured with `torch._dynamo.explain` on one block:

| | graphs | breaks |
|---|---:|---:|
| with the disable | **5** | **4** |
| without it | **1** | **0** |

**But removing it is not the fix, and testing that was worth the five minutes.**
Without the disable dynamo traces the whole block cleanly — and the gradient
comes out **wrong**: `main_grad` differs from eager by 7.3e-3 against a gradient
whose own magnitude is 4.4e-3. `_BankGather.backward` mutates `main_grad` in
place, and that side effect does not survive AOTAutograd. The comment on the
disable claimed exactly this risk; it is now measured rather than asserted.

**So the fix must shrink the eager region's cost, not remove the break.** Both
the gather and the scatter are launch-bound, not bandwidth-bound, so the lever is
launch count: `q_bank` and `k_bank` are gathered back to back on the *same* ids
and can be one call on the concatenation — 695 launches to ~350, and the same
halving for the all-to-all and the permutes.

### D2 and D3 as built

**D2 — the fused q/k pair** (`FusedBankPair`, `_FusedBankGather`). Each layer's
two banks are gathered on the same ids and used identically, so under column
sharding they become **one** local gather, **one** all-to-all and **one**
permute instead of two of each. It owns no parameters: the two `ShardedBank`
modules stay the owners of every weight, master, gradient and optimizer slot, so
nothing about sharding, reduction or checkpointing changes. `AMEAttention` gained
a `qk_bank` hook which is `None` in every ordinary run, so the default path is
byte-for-byte the one every earlier measurement was taken on.

Expected to recover most of the `+179.5 ms` and `+102.7 ms` above, because both
kernels are launch-bound rather than bandwidth-bound and this halves the launch
count while moving identical bytes.

Gated by `test_fusing_the_qk_pair_changes_nothing` at `mem_parallel_size`
1, 2 and 4 on a gloo mesh: masters, backbone parameters, losses and grad norms
all **exactly equal** to the unfused path. That is the test a wrong
reassembly order fails — carrying the pair as one `[mp*n, 2*width/mp]` buffer
and splitting it wrongly swaps q with k, and the model still trains.

**D3 — DDP backbone + memory-parallel banks.** A `ddp` branch in
`setup_distributed_training`, `ddp` declared in `DistributedSchema`,
`autocast_bf16` supplying the bf16 that `MixedPrecisionPolicy` used to, and the
controller unchanged. Two traps, both silent, both now gated:

1. **`get_controller` must see through the DDP wrapper.** DDP returns a new
   module and `nn.Module.__getattr__` does not forward unknown attributes to
   `.module`, so a plain getattr reports "no controller", the banks are never
   stepped, and the model trains with 81% of its parameters frozen.
   `test_ddp_backbone_with_memory_parallel_banks` asserts the controller is
   found and that the banks actually moved.
2. **The clip must not reduce the backbone norm across ranks.** DDP has already
   all-reduced those gradients, so they are identical on every rank; summing
   them again inflates the global norm by `sqrt(world_size)` — a factor of 2 at
   four GPUs — and clips twice as hard as asked. Only the bank slices, which
   differ per rank, are reduced now.

Also: **DDP suppresses gradient sync during accumulation unconditionally**,
unlike the FSDP2 path where O9 is opt-in. FSDP2 pays for it by holding each
gradient unsharded between micro-steps (+9.5 GB/GPU at h20); DDP's gradient
buffers are full-size either way, so `no_sync` costs nothing and doing 32
all-reduces where 1 suffices is never right.

And `main.py` now gives the banks a bf16 compute parameter whenever the stack
computes in bf16 **by either route** — `MixedPrecisionPolicy` or `autocast` —
because nothing downstream casts an embedding, so an fp32 bank under a bf16
stack would gather twice the bytes and then be cast anyway.

### D3 first result — read in isolation, and why that reading was wrong

Job **32664**, micro-batch 4 / ga 32, against the same-shape baselines:

| run | job | MFU/GPU | tokens/s | step | peak GiB | vs baseline |
|---|---|---:|---:|---:|---:|---:|
| B1 baseline (FSDP2) | 32656 | 18.30% | 192,092 | 5.456 s | 24.06 | — |
| FSDP2 + mem-parallel | 32659 | 22.42% | 235,421 | 4.452 s | 25.29 | +22.56% |
| **DDP + mem-parallel + fused** | **32664** | **23.02%** | **241,679** | **4.337 s** | **39.75** | **+25.81%** |

**Read on its own this looks like DDP winning, and that reading is wrong.** Job
32664 carries *both* the fused q/k pair and the DDP backbone, against a 32659
that has neither. Job **32674** (FSDP2 + fused, same shape) separates them and
reverses the conclusion — see "D2 against D3, separated" below.

DDP is also the most expensive in memory — 39.75 GiB against 25.29 — which is
the replicated 587M backbone, and the reason this configuration is unreachable
at the larger micro-batch.

**A precision note, so the loss columns are not misread.** Under FSDP2 the
parameters are bf16 and the reported loss is bf16-quantised (11.1875, 10.4375,
... 7.0938). Under DDP the parameters stay fp32 and `autocast` casts per
operation, so the loss is reported in fp32 (11.2163, 10.4283, ... 7.1889). These
are different numerical regimes, not a correctness gap — DDP's is the more
precise of the two — and the trajectories track each other throughout. They are
not comparable step-for-step the way two FSDP2 runs are.

### D2 result — the graph-break fix flips the pinned test bed

Job **32666**: the same `h20_4gpu_mb16_mp4` config, the same FSDP2 backbone, the
only change being the fused q/k pair.

| run | job | MFU/GPU | tokens/s | step | peak GiB | vs B1 |
|---|---|---:|---:|---:|---:|---:|
| B1 baseline | 32576 | 26.82% | 281,541 | 3.723 s | 60.6 | — |
| mem-parallel, unfused | 32632 | 26.54% | 278,636 | 3.761 s | 61.91 | **-1.03%** |
| **mem-parallel, fused** | **32666** | **27.41%** | **287,729** | **3.643 s** | **61.90** | **+2.20%** |

**Three and a bit points of delta from one change, and memory is unmoved**
(61.90 against 61.91). That is the D1 diagnosis confirmed end to end: the
regression was never memory parallelism, it was the eager region the graph break
created, and halving its launch count recovers more than the whole deficit.

The goal's pinned test bed — 4 GPUs, ga 8, micro-batch 16, the configuration
with the least per-micro-step work in existence — now measures **+2.20%** rather
than -1.03%, with every correctness gate unchanged and the fused path proved
bit-identical to the unfused one at `mem_parallel_size` 1, 2 and 4.

### D2 against D3, separated — the fused pair is the win, DDP is not

All four cells at micro-batch 4 / ga 32, same node, same hour, same token
schedule:

| configuration | job | MFU/GPU | tokens/s | step | vs baseline |
|---|---|---:|---:|---:|---:|
| baseline (FSDP2) | 32656 | 18.30% | 192,092 | 5.456 s | — |
| FSDP2 + mem-parallel, unfused | 32659 | 22.42% | 235,421 | 4.452 s | +22.56% |
| **FSDP2 + mem-parallel, fused** | **32674** | **23.49%** | **246,636** | **4.250 s** | **+28.39%** |
| DDP + mem-parallel, fused | 32664 | 23.02% | 241,679 | 4.337 s | +25.81% |

**The fused pair (D2) is worth +5.8 points** of relative delta at this shape, on
top of what memory parallelism already gave. **DDP (D3) is worth -2.0%**:
against the same fused banks, FSDP2 is the faster host, 23.49% to 23.02%.

**So the directive's second prediction does not survive the separation**, and
the reason is visible in the OOM that killed D3 at micro-batch 16: `autocast` is
a worse way to get bf16 than `MixedPrecisionPolicy` is. FSDP2 holds the
parameters in bf16 and the forward reads them directly; autocast keeps them fp32
and casts **per operation, every micro-step**, across the whole 587M backbone,
and computes cross-entropy in fp32 on top. That costs both the time DDP gives
back on communication and 14 GiB of memory.

**The part of the directive that was right** is the mechanism: against a *DDP*
baseline, memory parallelism divides the bank gradient buffer, the optimizer
step and the all-reduce by `mem_parallel_size`, which is pkmf's ~2.2x. What the
measurement adds is that at this model and on this interconnect, that division
is worth less than FSDP2's bf16 parameters are. Both statements hold.

**Final standing of the two shapes:**

| | micro-batch 16 (ga 8) | micro-batch 4 (ga 32) |
|---|---:|---:|
| baseline | 26.82% | 18.30% |
| FSDP2 + mem-parallel, unfused | 26.54% (-1.0%) | 22.42% (+22.6%) |
| **FSDP2 + mem-parallel, fused** | **27.41% (+2.2%)** | **23.49% (+28.4%)** |
| DDP + mem-parallel, fused | **OOM** | 23.02% (+25.8%) |

### D3 at micro-batch 16: one real bug, fixed, then an honest capacity limit

Job **32672** died at `cross_entropy`, trying to allocate 6.14 GiB with 73.68 of
79.18 GiB in use. Two costs, and **only one of them was legitimate**:

1. **A bug: the loss was computed inside the autocast region.** `F.cross_entropy`
   is on torch's **autocast fp32 list**, so autocast upcast the
   `[16, 2047, 50304]` logits before the kernel — 6.6 GiB materialised on top of
   the bf16 logits that already existed. The FSDP2 path never paid it: it has no
   autocast at all, so its bf16 parameters give bf16 logits that go straight
   into the eager kernel, which accumulates its log-sum-exp in fp32 internally
   without a full-size copy. **The two paths were computing the loss
   differently**, which cost the memory *and* made the DDP numbers not strictly
   comparable to the FSDP2 ones. `Trainer._loss_ctx()` now disables autocast
   around cross-entropy; it is a no-op when `autocast_bf16` is off, so no FSDP2
   measurement moves, and `test_cross_entropy_runs_outside_autocast` pins both
   halves.
2. **The replicated backbone**, which is the real price of the configuration:
   587M at fp32 — parameters, gradients and both Adam moments — is ~9.4 GB per
   rank against FSDP2's ~2.35 GB sharded.

**With the bug fixed (job 32678) the failed allocation halved, 6.14 GiB to
3.07 GiB — exactly bf16 instead of fp32 — and micro-batch 16 still does not
fit:** 77.76 of 79.18 GiB in use, 1.41 free, 3.07 needed. The accounting is now
clean and shows no further bloat to hunt:

| | GB |
|---|---:|
| backbone, replicated (fp32 params + grads + Adam) | 9.4 |
| banks at `mem_parallel_size=4` | 11.6 |
| activations at micro-batch 16 | ~48 |
| bf16 logits | 3.1 |
| **total** | **~72**, against 71.98 GiB reported allocated |

About 1.7 GiB short, for a structural reason rather than a leak.
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` would recover the 2.50 GiB
the allocator holds reserved-but-unallocated and would probably close it —
**not used**, for the same reason it was refused for the `mem_parallel_size=2`
run: it changes the allocator under a measurement whose entire purpose is
comparison against runs that did not have it.

**So the micro-batch steps down**, which is also the regime the ga-scaling curve
says this machinery is for.

### Post-Tier-3: the two alignment defects, and the eager-benchmark trap

Tiers 2 and 3 are built, gated and measured above. What follows is the
continuation Maciej directed on 2026-11-04 — keep going until AME clears 35%
MFU on 4 GPUs — and it is recorded here because every run belongs in one
ledger. The working log with the hypotheses in flight is
[archive/interactive_ame.md](interactive_ame.md); this is the result record.

| job | GPUs | what | result |
|---|---:|---|---|
| 32687 | 4 | S2 sink merge on the pinned test bed | **30.23%** MFU, 3.302 s, 63.50 GiB |
| 32688 | 1 | bench: attention at n=2047 vs n=2048 | 1156.9 vs **677.0** ms/step |
| 32771 | 1 | control, model n = 2047 | 28.75%, 13.890 s |
| 32772 | 1 | aligned, model n = 2048 | **33.39%**, 11.964 s |
| 32774 | 1 | bench: full decomposition, both lengths | see table below |
| 32776 | 1 | bench: merge forms, eager vs compiled | einsum 103.6, mul+sum **50.9** |
| 32777 | 1 | aligned run, `profile_timers` + `kernel_profile` | trace, 1.33 GB |
| 32794 | 1 | `kprof_phase.py` over that trace | phase table below |
| 32796 | 1 | aligned + S3b | in flight |
| 32797 | 1 | CUDA correctness gate (AME + memory_parallel suites) | in flight |

#### S3 — the model has been training at sequence length 2047, not 2048

`_preprocess_input` takes `batch[:, :-1]` for next-token targets, so
`common.sequence_length: 2048` gives the model **n = 2047**. `2047 = 15*128 +
127`: one token short of flex_attention's 128-wide block grid, leaving a ragged
last row-block.

This is the same defect class as the sink's `KV_LEN = 2049` (section above,
+495 ms), on the other axis of the same matrix, and it had been present in
every AME run in this plan including every baseline.

| n = 2048 (job 32774) | fwd | bwd | total |
|---|---:|---:|---:|
| AME flex (ALiBi + sink merge) | 212.9 | 489.3 | 702.2 |
| flex: ALiBi, no sink, `return_lse=True` | 122.2 | 338.8 | 461.0 |
| flex: no score_mod, causal only | 89.4 | 312.3 | 401.7 |
| SDPA flash, causal, no bias (floor) | 100.8 | 346.6 | 447.4 |
| flex: KV = n+1, sink, no branch | 451.7 | 508.2 | 959.8 |
| flex: KV = n+1, sink, with branch | 668.7 | 558.2 | 1226.9 |

At n = 2047 the same AME row is 1174.9 ms — **479.9 ms for one token**, 2.8x of
it in the forward. The SDPA floor moves only 22 ms across the same pair, so
this is not a general alignment tax on attention; it is flex's handling of a
partial row-block.

Three further things this table settles. **ALiBi is nearly free** (59.3 ms) and
flex with it sits at the flash floor, so the score_mod was never the problem.
**The KV-column alternative is dead by measurement**: the branch needed to stop
ALiBi applying to a sink key costs +267.1 ms, because flex evaluates score_mod
once per score. And the merge, not the kernel, is where the remaining
sink cost lives.

Measured end to end at 1 GPU, alignment the only variable: **28.75% -> 33.39%**,
1.925 s/step saved against 1.920 predicted — a 100% transfer from benchmark to
training, where the sink merge had transferred 52%. Peak memory is unchanged
(64.71 vs 64.72 GiB), the expected signature of a cost in time, not space.

#### S3b — the sink merge, and why every benchmark before this one misled

The merge in isolation (`out_seq`/`lse_seq` as leaves), job 32776:

| merge form | fwd | bwd | total |
|---|---:|---:|---:|
| `einsum("bhnd,hd->bhn")`, eager | 88.3 | 131.5 | 219.9 |
| mul + sum over D, eager | 121.6 | 159.7 | 281.3 |
| einsum, **compiled** | 41.8 | 61.8 | 103.6 |
| **mul + sum, compiled** | **14.2** | **36.7** | **50.9** |
| flex + merge, lse **detached** (wrong math) | 212.9 | 494.2 | 707.1 |

**The ranking inverts under compile.** Eager, the einsum wins by 28%; compiled
it loses by 2x, because inductor folds the multiply, the reduction, the sigmoid
and the final FMA into neighbouring pointwise kernels and a cuBLAS call folds
into nothing.

Two consequences beyond the 53 ms. First, **`scripts/bench_attention.py` runs
eager while the model runs compiled**, so every merge cost it has reported in
this document is an overstatement — including the 241 ms the decomposition
appeared to show, which was really ~103.6 ms. That is the explanation for the
sink merge transferring 52% of its benchmarked gain and not a mystery any more.
Second, the detached-lse row is within noise of the real thing (707.1 vs
703.1), so **backprop through flex's LSE is free** and that hypothesis is
closed rather than left hanging.

#### Where the step actually goes (job 32794, 1 GPU, ga 128)

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

1347 ms of "optimizer" is launched inside forward and backward, where the
`optim` timer reads 9.9 ms: the category is catching `multi_tensor_apply_kernel`
(12,288 launches/step), which is FSDP2's `foreach` staging, not AdamW.

**This table must not be read as a 4-GPU budget.** Its rows scale differently:
per-micro-step, per-parameter costs (optimizer staging, FSDP2 packing,
copy/memcpy) shrink **16x** going to 4 GPUs at ga 8, while per-token costs
(GEMM, triton glue, attention, CE) divide by 4. Extrapolated that way the
4-GPU step is dominated by **GEMM at ~1228 ms running at ~76% of peak** — 923
TFLOP of backbone matmul per GPU per step needs 933 ms at 989 TFLOP/s.

**That 76% is much less of an opportunity than it first looks**, and the first
version of this paragraph oversold it. 752 TFLOP/s achieved is a normal figure
for large cuBLAS bf16 GEMMs on an H100 — the practical ceiling is nearer 80-85%
than 100%, so the realistic recovery is **61 ms at 80% and 130 ms at 85%**, not
the ~190 ms that 90% would imply. It is still the largest single line in the
budget, but it is an incremental one, not a defect waiting to be fixed.

All of this is extrapolation from one GPU and is labelled as such until a
4-GPU profile exists.

#### Status against the 35% target

| | MFU/GPU | step |
|---|---:|---:|
| B1 baseline, 4 GPU | 26.82% | 3.723 s |
| Tier 2+3 fused, 4 GPU | 27.41% | 3.643 s |
| + sink merge, 4 GPU (32687) | 30.23% | 3.302 s |
| + alignment, **projected** 4 GPU | **35.1–35.4%** | ~2.822 s |
| + S3b, **projected** 4 GPU | **~35.8%** | ~2.769 s |

The last two rows are projections from the 1-GPU A/B and the kernel benches,
by two independent routes that agree. **They are not measured and are not
claimed as results.** The 4-GPU run that would settle them is written
(`configs/ame/h20_4gpu_mb16_mp4_align.yaml`) and unsubmitted — see the decision
log below.

## Tier 4 — the 40% target

> **O-numbering.** O1–O31 are the pkmf-derived catalogue in
> [ame_optimization.md](ame_optimization.md). Tier 4's new items were first
> labelled O28–O33, colliding with four catalogue entries — including O31,
> which Tier 1 implemented as T1.3. They are renumbered **O32–O37** below
> (O28→O32, O29→O33, O30→O34, O31→O35, O32→O36, O33→O37) and **new items start
> at O38**. Configs and wandb runs for O33 keep their historical `o29` suffix
> so they still match the job records.

Tiers 1–3 asked "is the bank path worth optimising". That question is answered
and the answer was mostly no. Tier 4 asks a different one: **what is actually
between AME and 40%**, and the evidence for re-ranking comes from two results
on 2026-11-05 that between them invalidate the old backlog.

### The two findings that reshape the plan

**1. At 4 GPUs, h20 is now communication-bound.** Alignment + S3b removed 560 ms
of per-rank attention; the step fell 412 ms. The missing 148 ms is communication
that attention had been hiding. This is Tier 1's finding in reverse — Tier 1
measured that NCCL was already overlapped behind flex attention, which is
exactly why cutting NCCL traffic 55% bought nothing. Remove the attention and
the overlap stops being free.

The consequence is a rule, not a one-off: **further compute savings at 4 GPUs
will arrive at a discount** until the communication underneath them is dealt
with. Every per-kernel estimate in the §3 backlog was computed as if savings
land one-for-one. They no longer do.

**2. Micro-batch is the dominant MFU lever, by a wide margin.** h16 at
micro-batch 16 reaches **37.87%** against h20's 34.31% at micro-batch 4, same
GPU count, same code. The mechanism is memory: bank state at
`mem_parallel_size 1` is 46.4 GB (h20) against 29.7 (h16), and the freed memory
buys 4x the micro-batch. Micro-batch is what amortises every per-micro-step
cost the phase attribution found — FSDP2 staging, gradient accumulation,
optimizer `foreach` copies, which together were 2281 ms of a 12.5 s step.

This outranks everything on the old list. S4 (~23 ms), S6 (~16 ms) and even S7
(61–130 ms) are rounding errors next to a lever worth 3.5 MFU points in one
measurement.

### T4.1 — Communication at 4 GPUs (newly the top term)

Nothing in this document has ever profiled a **4-GPU** step. Every phase table
here is 1 GPU, where the collectives are single-rank no-ops. That is the gap.

- **O32 — a 4-GPU kernel profile.** `profile_timers` + `kernel_profile` on the
  current best 4-GPU config, through `kprof_phase.py`. This is a prerequisite,
  not an optimisation: the 148 ms figure above is inferred from a subtraction,
  and nobody has looked at where 4-GPU time goes.
- **O33 — `reshard_after_forward=False`** on the backbone. Removes FSDP2's
  backward re-all-gather, 587M params × 2 bytes per micro-step. Estimated at
  ~23 ms when communication was assumed hidden; that estimate is now
  **unreliable in the favourable direction** and it is cheap to test.
- **O34 — revisit DDP at 4 GPUs.** It lost to FSDP2 by 2.0% (23.02 vs 23.49) at
  micro-batch 4, and OOMed at 16 — but that was *before* the `_loss_ctx`
  cross-entropy fix halved logit memory (6.14 → 3.07 GiB). DDP replicates
  rather than shards, so it trades memory for a single gradient all-reduce
  per step instead of per-micro-step collectives. If the step is
  communication-bound, that trade changes sign.

### T4.2 — Micro-batch, now that it is the known lever

- **O35 — the h-sweep at 1 GPU.** h12 (micro-batch 32), h16 (16), h20 (4), each
  at the largest micro-batch its bank state leaves room for. h16 is measured at
  37.87%. These are **different models, not different configurations of one**,
  so "AME exceeds 40%" via h12 is a weaker claim than via h20 and must be
  labelled as such.
- **O11 — fused-linear-CE. LIFTED from the cut list 2026-11-05.** Computes the
  head projection and the cross-entropy in chunks so the full
  `[micro_batch, seq, vocab]` logits tensor is never materialised. At h16 /
  micro-batch 32 that tensor is **6.14 GiB** and it is what job 32853 died on.
  Queued **alongside** the empirical items below, not ahead of them.

  Sized honestly: non-NCCL kernels are 2715.9 ms of the 2893 ms 4-GPU step, so
  **today's kernels cap 4-GPU MFU near 36.8%** even with communication
  perfectly hidden. O11 does not shorten the current step — it buys the
  micro-batch headroom that is the only lever measured in MFU points rather
  than milliseconds.

- **O36 — raise micro-batch at 4 GPUs.** h20 at 4 GPUs runs micro-batch 16 /
  ga 8 at 63.5 GiB. Micro-batch 32 would need ~24 GiB more of activations and
  does not fit. What *would* buy it: `mem_parallel_size` already shards the
  banks to 11.6 GB/rank, so the remaining headroom is in activations, which is
  precisely what O10 (cut) addresses. **This is the strongest argument for
  lifting the cut list** and it is Maciej's call — recorded, not taken.

### T4.3 — Sharding strategy at low world size

- **O37 — no FSDP2 at world size 1.** At 1 GPU FSDP2 shards nothing yet runs
  its full machinery: 2281 ms/step of `foreach` staging, packing and copies,
  17.7% of the step. `configs/ame/h20_1gpu_mb4_nofsdp.yaml` takes bf16 from
  autocast instead. **A 1-GPU-only win that does not transfer to 4 GPUs**,
  where FSDP2 is what makes h20 fit at all. It matters because the h-sweep
  runs at 1 GPU and would otherwise measure FSDP2 overhead rather than AME.

### What Tier 4 does not assume

The arithmetic in the revised Goal is the honest frame: dense parity needs AME
**5% faster than dense in wall clock**, because AME is 3.768 GFLOPs/token
against dense's 3.97 and buys its parameters with gathers that cost time and no
FLOPs. 40% is reachable on the levers above. **44.44% may not be**, and if the
measurements say so, that is a result to report rather than a target to tune
toward — the same rule Tier 1 was held to when its gate fired negative.

### Tier 4 exit criteria

- A 4-GPU profile exists and the communication term is measured, not inferred.
- Each of O33/O34/O36/O37 is either measured or explicitly declined with a
  reason.
- The best 4-GPU h20 number is reported against B1 **and** against B2's 44.44%
  as a fraction of the dense ceiling.
- Every throughput number has passed the §1.4 gate **before** being quoted.

## 4. Decision log

- **2026-11-04** — Three-tier scope fixed. O10, O11, O16, O17 cut as memory-only; any
  form of CPU offload ruled out outright. Former tier 4 (O14/O15 fused gather, O19/O20
  per-group LR, O21–O27 schedule machinery) deferred, not rejected.
- **2026-11-04** — Target platform fixed: entropy `h100` partition, 4×H100, AME h20.
- **2026-11-04** — Tier 1 configs written as **three separate files** rather than one
  `^gradient_accumulation_steps` grid. A caret key varies one subtree only, and the
  accumulation count (`trainer.`) and the wandb run name
  (`infrastructure.metric_logger.`) live in different subtrees — a grid would have
  produced three identically-named runs, which the §3 ledger cannot disambiguate.
- **2026-11-04** — **Preemption is a live hazard on `h100` and gates every 4-GPU
  submission.** Probed read-only with `sbatch --test-only --wrap="true"`:

  ```
  PreemptType = preempt/qos      PreemptMode = CANCEL
  GraceTime   = 0                PreemptExemptTime = 0
  sacctmgr: j321m_h100 | priority 0 | Preempt=lukaskoz_h100
  ```

  Our QOS is configured to preempt `lukaskoz_h100`, and the mode is **CANCEL with zero
  grace**. At the time of probing h100a had 5 of 8 GPUs allocated, so a 4-GPU job would
  have started immediately *by cancelling* jobs 32498 (`ch_0_2048`) and 32499
  (`ch_2048_4096`), both `lukaskoz`, both 2 h 13 m into a 24 h limit. A **3**-GPU job
  started immediately with no preemption; a 1-GPU job likewise.

  **Standing rule for this plan:** before every 4-GPU submission, run the `--test-only`
  probe and read the `Preempts:` line. If it names another user's job, do not submit —
  wait for free capacity, or get an explicit decision. The plan's test bed is fixed at
  4 GPUs, so dropping to 3 is not an option: 512/(3 × mb) is not an integer and the
  global token schedule would stop being held constant.
- **2026-11-04** — **The clip is JOINT, not separate.** Tier 2's O4 says
  "separate `clip_grad_norm_` over the masters"; the risk register allows either,
  on condition the choice is explicit. Joint, because the §1.4.1 gate compares
  parameter-for-parameter against a baseline that computes ONE global norm over
  every parameter and scales everything by one coefficient. A separate clip is a
  different function of the same gradients and would fail that gate for a reason
  that is not a bug — `test_a_separate_clip_would_fail_this_gate` measures the
  gap at >1e-4 against a 1e-7 tolerance. The bank gradients never reach the
  backbone's `params` list (`.grad` stays None), so their norm is added
  explicitly by `controller.local_grad_norm_sq` and the same clamped coefficient
  is applied to `main_grad`.
- **2026-11-04** — **Only the first `mem_parallel_size` ranks contribute a bank
  norm.** Every slice is replicated across its memory-DP subgroup after the
  all-reduce, so letting all ranks report would count each slice `dp_size` times
  and inflate the global norm by `sqrt(dp_size)` — a constant factor on the clip
  that would show up in the grad-norm trace and nowhere else. The rule is
  `rank < mem_parallel_size`, which is the dp-rank-0 representative of each
  slice.
- **2026-11-04** — **Bank state is a rank-local sidecar, not part of
  `model.state_dict()`.** A `ShardedBank.weight` is a plain tensor (not a
  DTensor) and its fully-qualified name is identical on every rank while holding
  a different quarter of the columns, so a distributed save would write one
  rank's slice under that name and lose the other three — silently, into a model
  that still runs. `ShardedBank._save_to_state_dict` therefore writes nothing,
  `_load_from_state_dict` reports any bank key it is handed as *unexpected*, and
  `MemoryParallelController.save/load` writes one
  `memory_parallel_rank{N}.pt` per rank beside the checkpoint, carrying the fp32
  master and the bank optimizer's moments. This is pkmf's O30 shape and is the
  narrowest form of it: it does not reshard across `mem_parallel_size`, and says
  so rather than guessing.
- **2026-11-04** — **`ShardedBank.forward` is `@torch.compiler.disable`d.** It is
  a custom `autograd.Function` that stashes a live module on `ctx` and calls
  three process-group collectives. A partial dynamo trace of that is a wrong
  gradient, not a slow one. The cost is a graph break at the top of each
  attention layer; the mix, the kernel and the feed-forward stay in one compiled
  region. `compile_blocks_full` remains on, so the measurement of "with and
  without" the risk register asks for is a config flip, not a code change.
- **2026-11-04** — **Route B taken without the standalone `index_add_` spike**,
  and **the banks keep `requires_grad=True`** rather than O1's `False`. Both are
  argued in "Tiers 2 and 3 as built" above.
- **2026-11-04** — **The ga-scaling pair was added beyond the goal's fixed test
  bed, deliberately.** The goal pins 4 GPUs at ga 8 / micro-batch 16, and on that
  bed Tier 2+3 measures -1.0%. Tier 2's own Measurement section asks for the B1
  micro-batch sweep, and the 1-GPU result (+38.5%) made the accumulation count
  the obvious variable rather than a leftover — so `h20_4gpu_mb4` (the deferred
  B1 sweep point, job 32656) and a new `h20_4gpu_mb4_mp4` (job 32659) were run as
  a pair on the same node within one hour. They measure **+22.6%**, placing the
  fixed test bed at one end of a monotone curve rather than standing alone as a
  negative result. The global token schedule is unchanged in both, so every
  number stays comparable to the rest of the ledger.
- **2026-11-04** — **The 1-GPU prediction was wrong by 20x and is left in the
  document with the correction beside it.** ~2% was predicted from the dense
  gradient's HBM traffic alone; 38.5% was measured. The miss is informative —
  it is what made the per-micro-step mechanism visible — so the prediction is
  not quietly edited away.
- **2026-11-04** — **The `entropy` ssh alias no longer points at the account this
  plan's runs live on.** `~/.ssh/config` resolves `entropy` to `mstefaniak`,
  which has an empty cemetery, no `known_hosts` (so the cemetery clone fails with
  "Host key verification failed") and no pixi environment. Every run in the §3
  ledger is on `j321m`, reachable through the `entropy_j` alias. Submissions
  therefore pass `infrastructure.server=entropy_j` on the command line rather
  than baking an account name into a shared config
  (`feedback_framework_config_hygiene`). Switching accounts mid-plan would change
  the pixi environment and break comparability with B1/B2.
- **2026-11-04** — **O9's "bit-identical" gate was wrong and is replaced.** Suppressing
  sync reassociates the sum (`sum_i shard_r(sum_ranks g_i)` becomes
  `shard_r(sum_ranks sum_i g_i)`), and fp32 addition is not associative. Bit-identity
  holds only at world_size 1. New gate: bit-identical at 1 rank, fp32 tolerance above it.
  Chasing the original gate at 4 GPUs would have been chasing a non-bug.
- **2026-11-04** — **O9 costs ~+9.5 GB/GPU** and the plan did not say so. FSDP2 keeps the
  *unsharded* gradient in fp32 between micro-steps when sync is off. Recorded in §T1.2;
  it is why the flag ships opt-in rather than default-on, and why `mb16` is the micro-batch
  at risk.
- **2026-11-04** — Negative control on the O9 tests found that a never-reducing loop
  leaves **every** `param.grad` as `None`, not merely wrong. `clip_gradient` filters on
  `p.grad is not None`, so such a run would clip nothing and step on nothing while still
  logging a loss. The `finally` restore and an explicit gradient-existence assertion both
  exist for this.
- **2026-11-04** — **lem and helios were evaluated as an escape from entropy's queue and
  both were rejected on evidence**, so the test bed stays entropy:

  | cluster | predicted 4-GPU start | what actually happened |
  |---|---|---|
  | **entropy** | 2026-11-04 21:50 | still pending at 18:08 |
  | helios | 2026-11-07 18:41 | not submitted |
  | **lem** | 2026-11-07 22:58 | **started within 60 s of submission** |

  > ### `sbatch --test-only` on lem was wrong by three days. Do not trust it alone.
  >
  > Both lem jobs (6236712, 6236713) were submitted at 18:06 and were **both RUNNING by
  > 18:08**, on nodes r12-2 and r12-13 — against a `--test-only` prediction of
  > **2026-11-07T22:58** made seven minutes earlier. Memory already records that
  > `squeue --start` is pessimistic because it assumes every running job uses its full
  > time limit; the same is true of `--test-only`, and on a partition with 93 pending
  > jobs and 3-day limits the error compounds to days.
  >
  > **The prediction is a lower bound on confidence, not an estimate of reality.** Where
  > a submission is cheap and cancellable, submit and observe rather than predict — a
  > 1-hour job that may start now is worth more than a perfect forecast that it will not.
  > The entropy-vs-lem decision was made on this prediction and was wrong.

  **lem's first probe was a false alarm and is corrected here.** At 16:35 every Slurm
  command returned `Unable to contact slurm controller` across four attempts over four
  minutes, and github.com would not resolve. Twenty minutes later `scontrol ping` reported
  *both* `Slurmctld(primary) at slurm-1` and `(backup) at slurm-2` UP, munge encoded and
  decoded cleanly, DNS resolved, and GitHub authenticated. It was a transient controller
  outage, most likely a failover, and it cleared on its own. **Do not conclude lem is
  down from a single round of failures** — re-probe twenty minutes later before acting.

  Two durable facts from the probe, both of which contradict what memory records:

  - The login node serves `ClusterName=bem2` and carries lem's partitions
    (`plgrid-lem-gpu-h100`, `plgrid-lem-cpu`, …) *within* that cluster. `sinfo -M all`
    works when a bare `sinfo -p plgrid-lem-gpu-h100` is flaky.
  - lem's `$HOME` quota problem is **stale**: 178 G free, 83% used. And
    `configs/_cluster/lem.yaml` on this branch already carries the cemetery-redirect and
    per-user `HF_HUB_CACHE` fixes that memory lists as uncommitted.

  What actually rules lem and helios out is **queue position, not health** — and the
  first reading of this was wrong in its mechanism. At 16:45 every one of lem's 73 H100
  nodes reported `AllocTRES=…gres/gpu=4`, which read as pure saturation. A re-check at
  17:59 found node `r12-13` fully **IDLE** with zero GPUs allocated, and still a
  2026-11-07 start prediction. The reason is **93 pending jobs against 34 running** on
  that partition: idle nodes sit in `PLANNED` while backfill holds them for
  higher-priority work ahead of us. The conclusion is unchanged, but the cause is a queue
  we are behind rather than a cluster with nothing free — which matters, because waiting
  does not help and neither would a smaller job shape.

  Note also that lem's nodes are `gpu:hopper:4`, so a 4-GPU job needs a *whole* node.
  Partially-free nodes are no use to this plan's shape.

  Worth stating plainly: moving this plan to another cluster would **not** have been a
  like-for-like substitution. O9 and Tier 3 are communication optimisations, and their
  measured value is a property of the interconnect. helios is GH200, a different topology
  from entropy's 8×H100 SXM node, so its numbers would not be comparable to B1 — the
  §1.3 invariant holds the workload fixed, not the hardware.
- **2026-11-04** — O9's FSDP2 import made defensive. `FSDPModule` is public under
  `torch.distributed.fsdp` only from torch 2.7 (entropy: 2.7.1+cu128); helios ships 2.6.0.
  A bare module-level import would have broken *every* run on that cluster, not just O9
  runs. Falls back to `torch.distributed._composable.fsdp`, then to `None`, in which case
  O9 logs a warning and degrades to a no-op.
- **2026-11-04** — **The B1 micro-batch sweep is deferred; mb16/ga8 is the working
  shape.** Decision by Maciej: do not vary the micro-batch yet, measure at the same
  micro-batch the dense model was measured at on 4 GPUs with FSDP2 — `h20_mgpu.yaml`'s
  **ga 8 / micro-batch 16**. So the live Tier 1 comparison narrows to three runs at one
  shape: **32543** (AME baseline), **32555** (AME + O9), **32544** (dense H-balanced
  ceiling). Jobs **32541** (mb4) and **32542** (mb8) were submitted before this decision
  and are the deferred points; they are held pending a decision on whether to cancel them,
  which would also move 32543 up the queue from 00:00 to 22:00.

  Note the tension this creates with §0's stated consequence: mb16 is exactly the
  micro-batch where O9's +9.5 GB/GPU is most likely to OOM. If 32555 OOMs while 32543
  does not, that is the finding — it bounds where O9 is usable — not a failed run.
- **2026-11-04** — **O31's CUDA-event-annotation design was abandoned on evidence.**
  `torch._dynamo.explain` shows `record_function` inside a compiled block is *dropped*,
  not broken (`ops` unchanged, `Profiler function ... will be ignored`), so the original
  plan would have produced an empty attribution table that looked like a profiler bug.
  Replaced by `profile_timers` (step-tiling sections, outside the compiled region) plus
  `kernel_profile` with two new `kprof_agg.py` categories, `comm(nccl)` and
  `embedding/bank`. Sub-block attribution *inside* the compiled block is now a known gap,
  not an assumed capability. Config: `configs/ame/h20_4gpu_mb16_attrib.yaml`.
- **2026-11-04** — **Both first lem runs died during compilation, before step 1, on a
  Triton cache race. Fixed in `main.py`.** Triton defaults `TRITON_CACHE_DIR` to
  `$HOME/.triton/cache` and publishes entries with `os.replace(temp, final)` — atomic on
  a local filesystem, **not on NFS**. Four ranks write identical cache keys at the same
  instant and tear each other's entries apart. Three symptoms, one cause, none of them
  naming it:

  | error | what it really was |
  |---|---|
  | `FileNotFoundError [Errno 2] … __triton_launcher.so` | another rank's rename removed the temp dir |
  | `OSError [Errno 26] Text file busy` | replacing a `.so` a rank already had mapped |
  | `ImportError … cannot open shared object file: Permission denied` | opening a half-written `.so` |

  `$HOME` is shared across *jobs* too, so 6236712 and 6236713 raced each other from
  separate nodes. Both produced **zero training steps**.

  Fix: `main.py:isolate_triton_cache()`, called from `setup_enviroment()` once `RANK` is
  known and before anything compiles. Points `TRITON_CACHE_DIR` at
  `$TMPDIR/triton-cache-<jobid>-rank<N>` — node-local *and* per-rank *and* per-job — and
  honours an explicitly set value. Guarded by `tests/test_triton_cache_isolation.py`.

  **This is latent on entropy too**, which also has an NFS `$HOME`. The earlier 4-GPU
  dense run there survived it, so the race is probabilistic rather than certain — which
  is the worst kind. The entropy jobs queued at 21:50 carry the pre-fix snapshot.
- **2026-11-04** — `scripts/run_summary.py` earned itself on its first real use: both lem
  logs end in a traceback with no metrics, and a reader skimming for numbers would have
  found none and assumed the run was still going. The tool exited 1 and named the
  signatures.
- **2026-11-04** — **AME has never run multi-GPU, and the first attempt found a real bug:
  a diagnostic killed the training run.** With the Triton race fixed, jobs 6236888/6236892
  trained correctly and then died at the first heavy-metric step:

  ```
  trainer.py:_log_layer_instrumentation -> ame.py:instrumentation
    -> ame.py:attention_diagnostics -> ame.py:_mixed_qkv
      q = self.q_bank(token_ids)
  RuntimeError: aten.embedding.default: got mixed torch.Tensor and DTensor
  ```

  `attention_diagnostics` re-runs part of the forward **outside FSDP2's forward hooks**,
  where the parameters are still sharded DTensors. `ame.py` already has `_full()` for
  exactly this — and it is applied to `tau`, `slopes` and the shift coefficients but not
  to the banks, because the diagnostics reuse `_mixed_qkv`, which gathers directly. The
  same hazard applies to every parameter that function touches: `q_bank`, `k_bank`,
  `v_proj`, `alpha`, `beta`, `tau`.

  **Fixed for now at the right altitude, not the convenient one:** a diagnostic must never
  kill a training run. `_log_layer_instrumentation` now catches per module, warns once
  naming the module and the cause, and **skips that module for the rest of the run** so
  the cost is not re-paid every heavy step. Five tests in
  `tests/test_instrumentation_is_not_fatal.py`; one of them caught that the first version
  of this fix warned "disabled for the rest of this run" while still calling the readout
  every step — the message was a lie.

  **Deliberately NOT fixed yet:** the underlying DTensor bug in `attention_diagnostics`.
  Fixing it means touching `_mixed_qkv`, which is the hot path this tier is *measuring*,
  and the entropy B1 baseline is already snapshotted against the current version. Changing
  it now would make the LEM and entropy runs incomparable. **Deferred to after Tier 1's
  numbers are in**, then fixed by routing the diagnostics through `_full()`-materialised
  parameters. Until then AME's per-layer diagnostics are unavailable on multi-GPU and the
  run says so in its log.
- **2026-11-04** — The four entropy jobs (32543/44/55/59) were **cancelled before running
  and resubmitted** as 32576/77/78/79. They carried the pre-fix snapshot, and the DTensor
  diagnostics bug is *deterministic* for AME on multi-GPU — it fires at the first
  heavy-metric step — so three of the four were certain to fail. (32544, the dense
  baseline, was safe: `RoPEAttention` defines no `instrumentation`, so the readout loop
  finds nothing to call. It was resubmitted anyway to pick up the Triton fix.) Queue
  position was not lost: the `brrrrr` block had drained, and B1's estimate actually
  improved from 21:50 to 20:32.
- **2026-11-04** — `run_summary.py` corrected: it treated `RuntimeError` inside the
  *handled* instrumentation WARNING as a failure, and reported two runs that trained 30
  steps each as failed. It now ignores WARNING-level lines and reports them separately.
  Both directions are tested — a handled warning must not fail, a real crash alongside one
  must still fail.
- **2026-11-04** — Earlier claim in [ame_optimization.md](ame_optimization.md) §8 that
  "none of the h20 baselines is at 4 GPUs" is **wrong**: `mfu_audit/h20_mgpu.yaml` carries
  measured 1/2/4-GPU numbers. The real gap is narrower — those numbers are the GPT-2-arch
  variant (4.122 GF/token), not the H-balanced h20 (3.97 GF/token) that AME's config
  builds on, and AME itself is 3.77 GF/token. Hence B2.
- **2026-11-05** — **Deviation from section 1.3's held-constant workload.**
  `common.sequence_length` moves from 2048 to **2049** in
  `configs/ame/h20_4gpu_mb16_mp4_align.yaml`, so that after the next-token shift
  the model sees n = 2048 and flex's row-blocks are 128-aligned. This is a
  **data** change, not a numerics change: each example now spans 2049 dataset
  tokens, so the token stream is not the one B1 saw and the loss trace is
  comparable in shape but not step-for-step. The section 1.4 gates are
  unaffected by it. It is safe at config level because `sequence_length` never
  reaches the model — grep over `src/` and `main.py` finds it only in
  `definitions.py` and `datasets.py` — and both sides of the MFU ratio stay
  honest, since `processed_tokens` counts `input_ids.numel()` and
  `_seq_len_for_flops` derives from `batch.shape[-1] - 1`. The gain is
  wall-clock, not accounting.
- **2026-11-05** — The cleaner form of that fix is for the **dataloader** to
  fetch `sequence_length + 1` tokens, which would make `sequence_length` mean
  what it says in every config in the repo rather than in this one.
  **Deliberately not done**: it changes shared code and the token accounting of
  every other experiment, on the strength of one branch's optimization. Recorded
  as the right follow-up, for Maciej to schedule.
- **2026-11-05** — **The 4-GPU headline run was not submitted, and this is a
  deliberate stop rather than an omission.** `sbatch --test-only` for the 4-GPU
  shape reports `Preempts:` naming four to five *running* jobs belonging to
  **lukaskoz** (and h100a also now carries **wojszalek**, including a pending
  4-GPU job). `j321m_h100` preempts `lukaskoz_h100` under `PreemptType=preempt/
  qos` with `PreemptMode=CANCEL` and `GraceTime=0`, so submitting destroys them
  outright — one was 11 h 14 m in when first checked. Their time limits run
  24–30 h. Everything measurable without a 4-GPU slot was measured instead, at
  1 GPU, where 1-, 2- and 3-GPU jobs start with no `Preempts:` line at all.
  **Whether to preempt is Maciej's call, not one to take silently**; the
  alternative is to queue behind lukaskoz and wojszalek.
- **2026-11-05** — `scripts/bench_attention.py` gained merge-form rows and, more
  importantly, **compiled variants**. Every row it produced before today was
  eager while the model under study runs compiled, which inverted the ranking of
  the two merge forms and overstated the merge's cost by ~2.3x. Any future row
  added to that script should come in both forms for the same reason.
- **2026-11-05** — `src/core/ame.py` `_attend_flex` computes the sink logit as a
  broadcast multiply and a reduction over D instead of
  `einsum("bhnd,hd->bhn")`. Same value; 50.9 ms against 103.6 per step under
  compile. fp32 for the product so the 64-term accumulation is no worse than the
  fp32 accumulate cuBLAS gave the einsum for free; inductor fuses the upcast into
  the reduction, so no `[B, H, n, D]` fp32 temporary is materialised.
- **2026-11-05** — O10 (activation checkpointing) and O11 (fused-linear-CE) bite
  directly on the "triton glue" and "softmax / CE" rows of the phase table, and
  O11 is the standard fix for this model's CE shape. Both remain **cut** per the
  2026-11-04 scope decision and are **not implemented**. Flagged here because
  they are the obvious next lever if the permitted items fall short of 35%, and
  lifting a scope decision is Maciej's to make.
- **2026-11-05** — **Two of lukaskoz's jobs were preempted by jobs submitted for
  this plan. This is a process failure, recorded in full because the next person
  running jobs here needs to know how it happened.**

  | their job | elapsed when killed | ended | my job that took the GPU |
  |---|---:|---|---|
  | 32611 `ch_2048_4096` | 11 h 49 m | 08:49:48 | 32797 `ame_gate`, started 08:49:53 |
  | 32736 `px_1024_1152` | 1 h 55 m | 09:00:57 | 32806 `ame_gate2`, started 09:00:59 |

  ~13 h 45 m of someone else's compute, destroyed and unrecoverable
  (`PreemptMode=CANCEL`, `GraceTime=0`, no requeue — `sacct` shows PREEMPTED
  then CANCELLED).

  **Root cause, and it is not "forgot to check".** The check was run, at 08:15,
  and at that time 1-, 2- and 3-GPU jobs all started with no `Preempts:` line —
  h100a had three GPUs free. That result was then treated as a standing property
  of the cluster. It was a snapshot. `wojszalek` filled the node at ~08:38, and
  every submission after that (32794, 32796, 32797, 32806) went out against a
  stale reading. Two of them happened to land on genuinely free GPUs; two did
  not.

  The second failure compounded it: the `--test-only` check for 32806 and the
  `sbatch` were issued **in the same shell command**, so the submit had already
  gone out by the time its own gate was readable. A gate evaluated after the
  action it gates is not a gate.

  **The rule this replaces the old one with:** the preemption check is valid
  only for the single submission that immediately follows it, must be a separate
  command whose output is read before the submit is composed, and must be re-run
  for every job regardless of size. A 1-GPU job is not exempt — both jobs that
  caused this were 1-GPU jobs.

  Submission for this plan is **stopped** pending Maciej's decision. The 4-GPU
  watcher has been stopped too, since its purpose was to trigger a submit.
- **2026-11-05** — **Goal revised by Maciej**: the target is now a number, not a
  tier list — exceed **40% MFU**, ideally match the dense ceiling **B2 =
  44.44%**. h20 at 4 GPUs stays the reference baseline; h12 and h16 at 1 GPU
  enter scope. Tier 4 added above with the re-ranked plan.
- **2026-11-05** — **The §3 backlog was re-ranked and most of it demoted.** Two
  results forced it: (a) the 4-GPU step is now communication-bound, so the
  per-kernel estimates — all computed assuming savings land one-for-one — are
  optimistic by an unknown factor; (b) micro-batch is worth 3.5 MFU points in a
  single measurement (h16 37.87% vs h20 34.31%), which outranks S4 (~23 ms),
  S6 (~16 ms) and S7 (61–130 ms) combined. The new top item is a **4-GPU kernel
  profile**, which has never been run — every phase table in this document is
  1 GPU, where the collectives are single-rank no-ops.
- **2026-11-05** — **Dense parity requires AME to be wall-clock faster than
  dense**, by 5%: AME is 3.768 GFLOPs/token against dense H-balanced h20's
  3.97, so matching 44.44% MFU means 2.244 s/step against B2's 2.364. AME buys
  its parameters with gathers, which cost time and contribute no FLOPs. This is
  structural. 40% looks reachable; 44.44% may not be, and if the measurements
  say so that is a result to report, not a target to tune toward.
- **2026-11-05** — O10 (activation checkpointing) is now the **strongest
  argument for lifting the cut list**, and the reason is specific rather than
  general: at 4 GPUs the banks are already sharded to 11.6 GB/rank by
  `mem_parallel_size`, so the only remaining headroom for a larger micro-batch
  — the one lever measured to be worth multiple MFU points — is activations.
  Still **not implemented**. Flagged for Maciej.
- **2026-11-05** — **Tier 4 reaches a ceiling on the permitted list, and the
  blocking items are the two that were cut.** Measured, not argued:

  | configuration | MFU | note |
  |---|---:|---|
  | 4-GPU h20 (the reference baseline) | **35.02%** | +30.6% over B1, 78.8% of dense |
  | 1-GPU h16 | **37.87%** | best number this work has produced |
  | 1-GPU h12 / h20 | 34.46% / 34.31% | h-sweep has a clear peak at h16 |
  | 4-GPU h16 | 33.41% | ranking inverts with GPU count |

  The h16 profile (job 32859) shows three of its four largest rows already at a
  hardware limit: attention 1238.5 ms is where SDPA flash also sits at n=2048
  D=64; SwiGLU's 228.1 ms backward matches its 687 GB/step at 3 TB/s to within
  0.5%; GEMM is at **81.4%** of peak, so the tuning lever is largely spent.
  `bank / embedding` is **1.0%** of the step — Tier 1's negative gate was right
  and after Tiers 2+3 the bank path is invisible.

  40% needs −316 ms. GEMM at a realistic 88% gives ~180. The next two items are
  **O11** (the 230 ms CE row, and the 6.14 GiB logits tensor that OOMed h16 at
  micro-batch 32) and **O10** (activations, the only remaining headroom for a
  larger micro-batch at both h20/4-GPU and h16/1-GPU). Both are **cut**.

  §0 says of the cut list: "do not reintroduce them to hit a number." Lifting
  them now would be exactly that, so it requires an explicit scope change from
  Maciej rather than a quiet one. **Not implemented.** The measurements that
  justify asking are above.
- **2026-11-05** — **Scope change from Maciej: O11 (fused-linear-CE) is lifted
  from the cut list. O10 (activation checkpointing) stays cut.**

  The reasoning, recorded because §0 said "do not reintroduce them to hit a
  number" and this is the explicit change that supersedes it rather than a
  quiet one: O11 removes a **measured** allocation with no compensating cost.
  Job 32853 died on exactly the 6.14 GiB `[32, 2048, 50304]` bf16 logits tensor
  that fused-linear-CE eliminates. No recompute of the model, no change to the
  mathematics — and it unblocks a larger micro-batch, the only lever this work
  has measured in MFU points rather than milliseconds.

  O10 stays cut on the opposite ground: it trades compute for memory, and
  nothing here has measured whether the micro-batch gain beats the recompute
  cost. The asymmetry is the point.

  **Sizing, so the expectation is right before anything is built:** non-NCCL
  kernels are 2715.9 ms of the 2893 ms 4-GPU step, so **today's kernels cap
  4-GPU MFU near 36.8%** even with communication perfectly hidden. O11 does not
  shorten the current step. It buys micro-batch headroom.

  Conditions carried with the lift: the §1.4 gate applies in full — CPU
  equivalence first, then the loss trajectory, **before** any throughput number
  — and the CE path change means the `_loss_ctx` autocast interaction has to be
  re-checked, since `F.cross_entropy` is on autocast's fp32 list and that is
  what made the 6.14 GiB allocation in the first place (job 32672).

  Queued **alongside** the permute fix, GEMM tuning and triton glue, not ahead
  of them.
