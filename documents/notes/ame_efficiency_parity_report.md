# AME efficiency-parity audit: the record

**Status (2026-11-06, 21:30): Phase 0 done except the checkpoint copy; Phase 2's first
gate and the G-REF probes are running.**

- The plan: [ame_efficiency_parity_plan.md](ame_efficiency_parity_plan.md). This file is
  its record: every check, job, result and decision, with its reason.
- The comparison it serves: [ame_comparison_h12.md](ame_comparison_h12.md).
- The optimizations it audits: [ame_optimization_report.md](ame_optimization_report.md)
  (`§` numbers below are that report's).
- Code, configs and the harness: branch `parity_audit`, worktree `~/code/nano-parity`,
  forked from `f9680709` (the code both runs trained on).

## Verdicts

One row per item of the plan's inventory (section 4). Filled in as the evidence lands.

| # | Item | Verdict | Evidence |
|---|---|---|---|
| A1 | Tier 2 bank path (O1-O5) | pending | |
| A2 | O3, the bf16 bank compute copy: the prep in bf16 | pending | |
| A3 | D2, the fused Q/K pair, traced by compile | pending | |
| A4 | flex_attention, compiled, ALiBi in `score_mod` | pending | |
| A5 | S2, the sink as an LSE merge | pending | source read: the LSE gradient is wired in torch 2.7.1 (Phase 0.2, item 3) |
| A6 | S3b, the sink logit as an fp32 mul+sum | pending | |
| A7 | tau folded into q before the bf16 cast | pending | |
| A8 | the compiled AME block | pending | |
| A9 | the bank sidecar checkpoint (O30) | pending | |
| X1-X8 | shared efficiency choices | pending | |

## Budget

25 GPU-h in total (plan section 8). Running total in the jobs table.

## Jobs

Every job runs a committed state of `parity_audit`; the cemetery branch named in the
clone path records it.

| Job | What | `parity_audit` commit | GPUs | Started | Wall | GPU-h | Result |
|---|---|---|---|---|---|---|---|
| (srun) | FineWeb identity check, CPU only | — | 0 | 21:15 | 1 min | 0 | identical to LEM (Phase 0.2, item 6) |
| 36309 | G0a, S2-PS production, 300 steps | `7bce462e` | 1 | 21:20 | 20:42 | 0.35 | `final_ce` 4.87109375, 3.49 s/step, peak 70.40 GiB (LEM run: 70.40); W&B `a6ndymxf` |
| 36310 | G-REF probe, micro-batch 16, 20 steps | `7bce462e` | 1 | 21:21 | ~3 min | 0.05 | out of memory at step 1 (75.1 GiB allocated, 3.07 GiB requested) |
| 36311 | G-REF probe, micro-batch 8, 20 steps | `7bce462e` | 1 | 21:22 | ~7 min | 0.12 | fits: peak 50.70 GiB, **9.43 s/step** (forward 3.64 s, backward 5.76 s); startup 2.5 min, step 0 24 s, `final_ce` 1 min |
| 36312 | data dump: train step 0-1 (1,024 sequences) and the held-out set (8,192), CPU only | `30ff2c06` | 0 | 21:34 | 3 min | 0 | `~/parity/data/parity_batches.pt` on entropy |
| 36313 | G0b, S2-PS production, 300 steps | `30ff2c06` | 1 | 21:35 | ~21 min | 0.35 | `final_ce` 4.81055 |
| 36314 | G0c, S2-PS at micro-batch 16 × 32, 300 steps | `30ff2c06` | 1 | 21:36 | ~22 min | 0.37 | `final_ce` 4.83594 |
| 36315 | G1, `mem_parallel_size: null`, 300 steps | `30ff2c06` | 1 | 21:37 | ~22 min | 0.37 | `final_ce` **5.07813**; 3.45 s/step |
| 36316 | GD0a, S1-D production, 300 steps | `30ff2c06` | 1 | 21:39 | ~19 min | 0.32 | `final_ce` 5.17188 |
| 36317 | GD0b, S1-D production, 300 steps | `30ff2c06` | 1 | 21:40 | ~19 min | 0.32 | `final_ce` 5.19531 |
| 36319 | G-REF, micro-batch 8 × 64, 220 steps | `dcfce193` | 1 | 21:42 | | | running |
| 36320 | numerics: `budget --states init` | `dcfce193` | 1 | 21:43 | 3 min | 0.05 | dense done; AME stopped out of memory at `ame_prod_eager`, micro-batch 32 |
| 36321 | numerics: `final_ce --arms dense`, `budget --arms dense --states final` | `9387a44d` | 1 | 21:46 | 8.5 min | 0.14 | dense `final_ce` reproduced exactly |
| 36323 | numerics: `budget --arms ame --states init` (eager at micro-batch 16) | `eb21bc17` | 1 | 21:48 | 6 min | 0.10 | done |
| 36324 | numerics: `optim --states init`, then `machinery` | `e27d36b2` | 1 | 21:56 | 2 min | 0.03 | 1.5 passes; `machinery` stopped (it read the half-copied sidecar), to be rerun |
| 36325 | numerics: X3, the fp32-CE variants at init | `e27d36b2` | 1 | 21:56 | 4 min | 0.07 | done |
| 36326 | numerics: `layers --states init` (1.2, 1.4; 12 layers) | `f096ed77` | 1 | 21:58 | ~3 min | 0.05 | done |
| 36327, 36329, 36330 | G0a-220, G0b-220, G0c-220 (micro-batch 16) | `f096ed77` | 3 | 21:58-22:00 | | | running |
| 36331 | G1r, G1 again | `9f4d6384` | 1 | 22:06 | | | running |
| 36333 | G1a, Tier 2 with an fp32 compute copy (A2 off) | `9f4d6384` | 1 | 22:07 | | | running |
| 36336 | G1b, Tier 2 with the fused gather eager (A3 off) | `9f4d6384` | 1 | 22:08 | | | running |

`src/` is identical at `7bce462e`, `30ff2c06`, `dcfce193` and `65702c61`, which add only
configs and the harness. So every gate trains the code the runs used.

## Phase 0: setup

### 0.1 The work branch

`parity_audit` created at `f9680709` in the clean worktree `~/code/nano-parity`.

### 0.2 The inventory, re-derived

**Code identity.** `git diff f9680709 40155323 -- src/ main.py` and the same against
`8a240816` are both empty. S2-PS, S1-D and the merge-base with `ame_optimize` therefore
run the same `src/` and `main.py`, and none of the §7 items is present.

**The logged configs** (W&B `doi36p8k` and `x0xlqroj`) differ only in:

- the attention type (`ame` with `shift: true`, against `gqa`);
- `trainer.mem_parallel_size` (1 against null);
- seven keys that dense's logged config lacks: `compile_ce`, `compile_loss`,
  `fused_linear_ce`, `fused_ce_chunk_size`, `kernel_profile`, `profile_timers` and
  `skip_grad_sync_during_accumulation`. Its grid config has no `/base_schema`, so the
  schema defaults never reached its log. The Trainer's own defaults are False and None,
  the values AME logged, so X8 holds for both runs.

Everything the recipe needs is equal in both: LR 2^-8, wd 0.1, clip 1.0, warmup 38,
`final_lr_fraction` 0.1, ga 16 on 512 × 2049, autocast bf16, `compile_blocks_full`,
`distributed: null`, data seed 123 for train and eval, eval interval 100 × 10 batches.

**Findings from reading the code at `f9680709`:**

1. **The two AdamWs match in their hyperparameters.** The backbone uses
   `AdamW(parameter_groups(...), lr, fused=True)` and the banks use
   `AdamW(masters, lr, weight_decay=0.0)`. Both take the default betas (0.9, 0.999) and
   eps 1e-8, and the banks sit at wd 0 in either path. The only difference is that the
   bank optimizer is not fused: it takes torch's default for CUDA tensors, foreach. Check
   1.5 measures it.
2. **The banks follow the backbone's LR schedule exactly.** `Trainer.train` passes
   `optimizer.param_groups[0]["lr"]` to `banks.step` before `scheduler.step()`. Both
   parameter groups share one LR, which the cosine scheduler asserts.
3. **A5's named risk is ruled out at the source in torch 2.7.1.** The plan's risk was
   that flex's LSE output carries no gradient. It does, by both backward paths:
   - inductor's flex-backward lowering
     (`torch/_inductor/kernel/flex_attention.py:2517-2520`) folds `grad_logsumexp` into
     `delta`, as `delta = sum(out * dout) - grad_lse / ln 2` on the kernel's log2-scaled
     LSE;
   - the math backward (`torch/_higher_order_ops/flex_attention.py:825-847`) adds it to
     `grad_score_mod`.

   The Python wrapper returns `lse * ln 2` (`torch/nn/attention/flex_attention.py:1331`),
   and the two scale factors cancel consistently. This settles only that the term is
   present. Check 1.2 still has to show it is numerically right at the production shape.
4. **The per-token bank gradients reach `main_grad` as bf16.** `ShardedBank.accumulate`
   does `main_grad.index_add_(0, ids, grad.to(fp32))` on gradients of the bf16 compute
   copy. So every token's contribution is rounded to bf16 once before the fp32 sum. In
   the ordinary path the same rows come back fp32. This is part of A2, as the plan
   already says.
5. **A7 adds no rounding that a bf16 kernel input does not already have**, except
   through A2. Both backends fold tau into q before the cast to bf16. That one rounding
   of q·tau has the same relative size (2^-9) as rounding q alone and scaling the fp32
   scores, which is what a plain implementation would do; dense's 1/8 scale is a power
   of two and exact.

   The extra error is that tau itself is rounded to bf16 when q is bf16
   (`self.tau.to(q.dtype)`). That happens only on the Tier 2 path, where q comes from
   the bf16 compute copy: a fixed per-head scale error of up to 0.37% at tau ≈ 8.4.
   Check 1.4 measures it, and it is booked under A2.
6. **The data on entropy are LEM's data.** The FineWeb copies are identical. Both
   directories hold 12,992 entries with byte-identical `state.json` and
   `dataset_info.json` (HF fingerprint `b7337f843adb67c2`, 12,990 shards). The sha256 of
   shards 0, 6000, 12987, 12988 and 12989 matches; the last three hold the 262,144-row
   holdout. Entropy's dataset log reports the same 1,279,107,432 rows and the same cut as
   both LEM runs. `final_ce` can therefore be reproduced on entropy (1.1's precondition).
7. **`final_ce` is quantized to 2^-10.** Each eval batch's `reported_loss` is a bf16 sum
   of 16 bf16 chunk means, with a bf16 ulp of 2^-6 ≈ 0.0156 near 3. `final_ce` is the
   float64 mean of 16 such values: 3.125 = 3200 × 2^-10, and 2.982421875 = 3054 × 2^-10.
   Rounding alone gives each run's `final_ce` a noise of about ±0.001 (std). The 0.143
   gap is known to about ±0.0015. The fp32 common evaluator (1.1) gives the unrounded
   values.
8. **Neither LEM run fell back from compile.** Both `slurm-*.out` files hold stderr and
   have no recompile-limit or graph-break warning (check 1.7, third part).

**Inventory completeness against §1.0 of the optimization report.** Every §1.0 row was
checked for whether it is active in these runs' code and config:

- **Active, and in the plan's inventory:**
  - O1-O5 and O30 (A1, A2, A9);
  - D2 (A3);
  - S2, S3b (A5, A6);
  - `_loss_ctx`, S3, O37 (X3, X4, X2);
  - `compile_blocks_full` (X1, A8);
  - the Triton cache isolation and the non-fatal instrumentation, which do not touch
    numerics.
- **Not active:**
  - O6-O8 (`mem_parallel_size` 1, so no mp group and no collective);
  - O9 (no FSDP2), O11 and O45 (`fused_linear_ce` false), O33 and O40 (no FSDP2);
  - D3;
  - O38-O47 (not in the code);
  - O13, O18 (not changes).
- **No new item was found.** Two details belong to existing items:
  - the bank AdamW is foreach rather than fused (A1);
  - the per-token bank gradients are bf16 (A2).

### 0.3 The CPU baseline at `f9680709`

`pytest -m "not gpu and not slow"` in the worktree, on this laptop's pixi env:
341 selected, **323 passed, 18 skipped, 0 failed**.

### 0.4 Configs

`configs/parity/` on `parity_audit` (`7bce462e`):

- the two bases, `s2_ps_gate_base.yaml` and `s1_d_gate_base.yaml`. Each is the run's recipe
  with the 300-step schedule (warmup 38, cosine to 10% at 300), entropy, no checkpoint,
  and W&B project `nano-ame-parity`;
- one file per gate: `g0a`, `g0b`, `g0c`, `g1`, `gd0a`, `gd0b`;
- the probes `g_ref_probe_mb16` and `g_ref_probe_mb8`.

Each composes locally (`main.py --cfg job`).

**Decision: a W&B project of its own.** `nano-ame-parity` rather than `nano-ame` keeps
300-step runs out of the Stage dashboards and queries that another session reads.

### 0.5 The two checkpoints

Copying LEM → this laptop's stream → entropy `~/parity_ckpt/` (`/home` is NFS, so every
node sees it). The sha256 is recorded at the LEM file, in the stream and at the entropy
file.

| File | Bytes | sha256 (LEM = stream = entropy) | Time |
|---|---|---|---|
| dense `__model_checkpoint_filename.pt` | 2,201,610,095 | `d8b58309…` (all three match) | 914 s |
| dense `__training_state_filename.pt` | 1,497 | `025c6451…` (all three match) | 7 s |
| AME `__training_state_filename.pt` | 1,497 | `240dbb47…` (all three match) | 7 s |
| AME `__model_checkpoint_filename.pt` | 2,116,952,487 | in progress | |
| AME `memory_parallel_rank0.pt` | 11,126,470,095 | queued | |

The stream runs at about 2.4 MB/s, which is this laptop's uplink, so the AME files take
about 1.5 h. Until they land, the checks that need the trained state wait. The init-state
checks and the gates do not need them.

### Phase 1 data

`parity_numerics.py dump_data` (job 36312) wrote the first two training batches of seed
123 (1,024 sequences of 2,049 tokens) and the whole held-out set (8,192 in 16 batches),
exactly as the Trainer's dataloaders yield them, to `~/parity/data/parity_batches.pt` on
entropy. Check 1.1 also verifies that the held-out tensor equals what `Trainer.final_eval`
reads.

## Phase 1: numerics

The harness is `documents/scripts/parity_numerics.py` on `parity_audit`. Its tables come
from `documents/scripts/parity_tables.py`, and jobs go through
`documents/scripts/parity_submit.sh`, a git archive of a commit, so the job log names the
commit. Every path is a *variant*: the run's config plus Hydra overrides, built by
`main.get_model_optimizer_scheduler`, with Residual's metric hook set (as in the runs,
where it broke the graph):

| Variant | What it is | Micro-batch |
|---|---|---|
| `ame_prod` | S2-PS as run: Tier 2 (bf16 compute copy), compiled blocks, flex | 32 |
| `ame_prod_eager` | the same, `compile_blocks_full: false` (A8); flex stays compiled | 16 (32 does not fit eager) |
| `ame_std` | standard mixed-precision AME: banks as fp32 `nn.Embedding`, compiled, flex (A1-A3 off) | 32 |
| `ame_ref16` | plain bf16 AME: reference attention under autocast, eager | 8 |
| `ame_ref32` | the oracle: fp32, reference attention, eager, TF32 off | 4 |
| `ame_prod_ce32`, `dense_prod_ce32` | production with the CE on fp32 logits (X3 isolated) | 16 |
| `dense_prod` | S1-D as run: compiled, SDPA flash | 32 |
| `dense_prod_eager` | the same, eager | 32 |
| `dense_ref32` | the oracle: fp32, SDPA math, eager | 8 |

Gradients are summed exactly as the Trainer sums them: each micro-batch's mean CE is
scaled by its share. They are compared per parameter group by relative error, cosine, and
the **signed projection** ⟨Δ, g⟩/‖g‖², which measures a systematic shrink or stretch
along the true gradient. Every comparison runs on 256 sequences: the first training batch
of seed 123 ("train") and the start of the held-out set ("heldout").

### 1.1 `final_ce` reproduced; the common evaluator

| | LEM `final_ce` | entropy, `Trainer.final_eval` on the checkpoint | production forward, fp32 CE, fp64 sum | fp32 reference path (the common evaluator) |
|---|---|---|---|---|
| dense | 2.982421875 | **2.982421875** (exact) | 2.982760 | 2.982607 |
| AME | 3.125 | **3.125** (exact; sidecar loaded through `load_checkpoint_from_file`) | 3.125377 | 3.125227 |
| **AME − dense** | 0.142578 | 0.142578 | **0.142617** | **0.142620** |

- **Both checkpoints reproduce their `final_ce` exactly on entropy**, which validates the
  harness, the copy and the sidecar load (jobs 36321, 36361). The dumped held-out tensor
  equals what `final_eval` reads (`torch.equal`).
- **The precision tax on the loss is identical in the two arms.** Each production path
  scores 0.00015 above its own fp32 oracle on the full held-out set: dense +0.000153, AME
  +0.000150.
- **The gap is 0.1426 under every evaluator.** The bf16 forward (and with it A2-A8 in the
  forward) costs AME nothing that it does not cost dense.

### 1.6 The whole-model error budget (with 1.3's structure checks)

**At init** (seed-123 init; jobs 36320 and 36323). Relative error / signed projection
against the fp32 oracle, on the training batch:

| group | dense_prod | ame_prod | ame_prod_eager | ame_std | ame_ref16 |
|---|---|---|---|---|---|
| q_proj / bank_q | 7.12e-3 / −3.9e-5 | 6.28e-3 / +3.2e-4 | 6.37e-3 / +3.4e-4 | 5.94e-3 / +3.5e-4 | 6.00e-3 / −6.0e-5 |
| k_proj / bank_k | 6.94e-3 / +1.0e-4 | 6.61e-3 / +3.1e-4 | 6.70e-3 / +3.1e-4 | 6.31e-3 / +2.6e-4 | 6.41e-3 / −5.7e-5 |
| tau | — | 7.45e-3 / **+1.7e-3** | 7.16e-3 / **+1.7e-3** | 7.80e-3 / **+1.6e-3** | 7.18e-3 / **+1.0e-3** |
| alpha | — | 8.67e-3 / **+2.4e-3** | 8.65e-3 / **+2.3e-3** | 9.18e-3 / **+2.0e-3** | 8.38e-3 / **+1.5e-3** |
| beta | — | 5.48e-3 / +6.5e-4 | 5.61e-3 / +5.7e-4 | 5.44e-3 / +6.9e-4 | 5.51e-3 / +1.9e-5 |
| k_sink / v_sink | — | 5.92e-3 / +2.1e-4; 4.99e-3 / +8.8e-4 | similar | similar | similar |
| v_proj | 5.18e-3 / −5.7e-4 | 6.02e-3 / **+1.1e-3** | 6.10e-3 / **+1.2e-3** | 6.02e-3 / **+1.1e-3** | 5.96e-3 / +6.4e-4 |
| o_proj | 5.21e-3 / −5.9e-4 | 6.06e-3 / **+1.2e-3** | 6.17e-3 / **+1.3e-3** | 6.06e-3 / **+1.2e-3** | 6.02e-3 / +7.5e-4 |
| ff | 6.04e-3 / −1.6e-4 | 6.16e-3 / +3.1e-4 | 6.18e-3 / +2.8e-4 | 6.14e-3 / +3.0e-4 | 6.13e-3 / +1.0e-4 |
| norm | 4.35e-3 / −3.1e-4 | 4.93e-3 / +5.8e-4 | 4.99e-3 / +5.9e-4 | 4.94e-3 / +5.7e-4 | 4.91e-3 / +3.1e-4 |
| embedding | 5.66e-3 / −4.5e-5 | 6.00e-3 / +5.4e-4 | 6.00e-3 / +5.7e-4 | 5.98e-3 / +5.1e-4 | 5.94e-3 / +2.8e-4 |
| head | 4.42e-3 / −1.6e-4 | 5.28e-3 / +6.8e-4 | 5.31e-3 / +6.6e-4 | 5.27e-3 / +6.6e-4 | 5.23e-3 / +3.8e-4 |

What the init table says:

- **No group's relative error comes near the flag line** (twice dense's largest, 1.42e-2).
  AME's largest is alpha at 8.7e-3, close to dense's 7.1e-3.
- **The positive projections over 1e-3 (tau, alpha, v_proj, o_proj) are not the
  efficiency work's.** `ame_std`, with Tier 2 and the fused pair off, has them at the same
  size. The plain bf16 reference `ame_ref16` has them for tau and alpha too.
- **The efficiency path differs little from standard mixed precision.** `ame_prod` against
  `ame_std` (A1-A3) is at most 0.42% relative (alpha) and 3.8e-4 in projection.
  `ame_prod` against `ame_prod_eager` (A8) is at most 0.33% and 1.5e-4.
- **Every bf16 AME path stretches the gradient slightly** (projections positive in every
  group), while dense's production path shrinks it slightly. At init every logit is near 0,
  so every log-probability is near −10.826 and rounds to bf16's −10.8125. That inflates
  every p by 1.4% in the bf16-logit CE (X3), a candidate for the common component.
  `ame_prod_ce32` and `dense_prod_ce32` (job 36325) test it.

**At the final checkpoint, dense** (job 36321): relative error 1.1-4.0% per group (head
the largest: 2.81e-2 held-out, 3.99e-2 train), against 0.4-0.7% at init. Trained
networks are more sensitive to rounding. Dense's own production path has systematic
projections past 1e-3: head −6.3e-3 (held-out), norm +4.2e-3 and v_proj −2.2e-3
(train). Compile is not the cause: `dense_prod_eager` sits as far from the oracle, and
1-2% from `dense_prod`.

**At the final checkpoint, AME** (jobs 36362 and 36363). Relative error / signed
projection against the fp32 oracle; "R" means over twice dense's largest group (5.6e-2
held-out, 8.0e-2 train):

| group | ame_prod (train) | ame_std (train) | ame_ref16 (train) | ame_prod (held-out) | ame_ref16 (held-out) |
|---|---|---|---|---|---|
| bank_q | 2.50e-2 / −2.6e-3 | 2.9e-2 / −2.7e-3 | 4.02e-2 / +4.0e-4 | 2.09e-2 / +6.9e-4 | 3.42e-2 / −7.5e-3 |
| bank_k | 1.94e-2 / −5.9e-3 | 2.2e-2 / −8.3e-3 | 2.36e-2 / −6.8e-3 | 2.04e-2 / +2.3e-3 | 2.65e-2 / +1.4e-3 |
| tau | **1.06e-1 R** / −1.2e-2 | **1.1e-1 R** / −1.6e-2 | **9.91e-2 R** / −1.5e-2 | **8.55e-2 R** / +1.9e-2 | **7.72e-2 R** / +1.4e-2 |
| alpha | **1.61e-1 R** / −1.3e-2 | **1.7e-1 R** / −1.5e-2 | **1.60e-1 R** / −1.1e-2 | **1.16e-1 R** / +3.5e-2 | **1.08e-1 R** / +2.4e-2 |
| beta | **3.09e-1 R** / −1.1e-1 | **3.1e-1 R** / −1.1e-1 | **2.80e-1 R** / −1.0e-1 | **1.64e-1 R** / −2.7e-2 | **1.47e-1 R** / −2.4e-2 |
| k_sink | 3.83e-2 / +6.3e-3 | 4.4e-2 / +7.9e-3 | 4.58e-2 / +8.2e-3 | 3.41e-2 / +2.8e-3 | 3.71e-2 / −6.2e-4 |
| v_sink | 2.51e-2 / −1.6e-3 | 2.6e-2 / −1.8e-3 | 2.50e-2 / −6.0e-4 | 1.71e-2 / −3.6e-3 | 1.70e-2 / −3.5e-3 |
| v_proj, o_proj | 1.5-1.7e-2 / ≤ 2.8e-4 | 1.5-1.7e-2 | 1.6-1.7e-2 | 1.3-1.4e-2 | 1.4-1.5e-2 |
| ff | 1.51e-2 / −5.6e-4 | 1.5e-2 / −6.4e-4 | 1.59e-2 / −5.6e-4 | 1.35e-2 / −3.5e-4 | 1.42e-2 / −2.9e-4 |
| norm | 5.74e-2 / −5.1e-4 | 5.7e-2 / −1.3e-3 | 5.01e-2 / −3.5e-3 | 3.37e-2 / +8.4e-3 | 2.89e-2 / +6.4e-3 |
| embedding | 1.64e-2 / −6.7e-4 | 1.7e-2 / −1.0e-3 | 1.74e-2 / −7.7e-4 | 1.45e-2 / −1.9e-4 | 1.56e-2 / +4.5e-6 |
| head | 2.33e-2 / +9.0e-4 | 2.4e-2 / +9.3e-4 | 2.60e-2 / +1.3e-3 | 1.67e-2 / +6.3e-3 | 1.88e-2 / +7.8e-3 |

Precise losses on the 256 sequences:

| Batch | oracle | prod | std | ref16 | prod_ce32 | prod_eager |
|---|---|---|---|---|---|---|
| train | 3.277061 | 3.277227 | 3.277171 | 3.277206 | 3.277229 | 3.277210 |
| held-out | 3.125641 | 3.125737 | 3.125766 | 3.125795 | 3.125739 | 3.125767 |

What the final-state table says:

- **The efficiency path is no less accurate than plain bf16 AME, and on the banks it is
  more accurate.** On every group, `ame_prod`'s error is within the range of `ame_std`
  (A1-A3 off) and `ame_ref16` (A1-A7 off, the plain bf16 implementation). On the banks it
  is better than the plain one: 2.1-2.5% against 3.4-4.0% for bank_q. Flex keeps its
  scores in fp32, while the plain implementation rounds them to bf16 before the softmax.
- **The A-items, one by one, against production:**
  - std (A2's effect) 0.55-2.3%;
  - eager (A8) 0.6-1.7%;
  - CE32 (X3) 0.2-0.9%.

  All are rounding-sized, the same scale as two bf16 paths of dense (dense's eager against
  its compiled: 0.7-2.4%).
- **One AME-specific numerical finding, not caused by the efficiency work: the per-head
  scalars' gradients are poor in bf16 late in training.**
  - alpha, beta and tau carry 8.6-31% error against fp32 at the final checkpoint, against
    0.5-0.9% at init. That is 2-8× dense's worst group (head, 2.8-4.0%).
  - beta's projection on the training batch is −0.11.
  - **Every bf16 path has it, the plain reference included.** It is not tied to the
    prep's precision either: `ame_std`'s prep is fp32.
  - The absolute error grows with training while the true gradient does not shrink
    (alpha: |g| 3.3e-3 at init and at the end, error ×19; tau: |g| ×0.64, error ×9). The
    bf16 attention backward's dq/dk errors (3.5-5.4e-3) are amplified by cancellation in
    the per-head reductions as attention sharpens.
  - **It belongs to X2 (autocast bf16, shared by both arms) interacting with AME's
    design**: dense has no per-head scalars to suffer it. Its training effect is not
    measured here (see the verdicts).

**1.3's exact items at init: all pass.** For all 24 banks, on both batches:

- no absent id has a nonzero row;
- no present id has a zero row;
- no id that occurs only as a target has a gradient, so the banks see `input_ids`, not
  the targets;
- every absent row that follows a present row is exactly 0. The inductor CPU bug (§1.18)
  would put row id's gradient into row id + 1, so this is the decisive test, and it is
  clean.

The error of `main_grad` on present rows is ≤ 8.1e-3 relative. Its correlation with the
*previous* row's reference gradient is ≤ 0.020. That is far below the bug's signature;
control correlations (the next row, the row's own gradient) were added in `48f9da34` to
place it.

**1.3 at the final checkpoint (jobs 36362 and 36363): all exact items pass again**, on
both batches, for all 24 banks.

- `main_grad`'s error on present rows is ≤ 3.6% against the oracle.
- Its correlation with the previous row's gradient (≤ 0.07 train, 0.13 held-out) is
  matched by the *next* row's control (0.06, 0.17). The bug would make the first far
  exceed the second.
- The error's correlation with the row's *own* gradient (0.25-0.56) shows that its large
  part is a per-row rescale: the bf16 path's small systematic shrink, bank_k −5.9e-3 on
  the training batch. `ame_std` has it too (−8.3e-3).

**1.7 at the final checkpoint (job 36361): passes.**

- The cached BlockMask equals a freshly built one tensor for tensor (`kv_num_blocks`,
  `kv_indices`, `full_*`, `q_*`) and is exactly causal at block granularity. The run used
  one shape, 2048, for train, eval and `final_ce`.
- The instrumentation changes no parameter, buffer, bank master, `main_grad`, RNG stream
  (CPU or CUDA) or BlockMask cache entry.
- Neither LEM run fell back from compile (Phase 0.2, item 8).
- (At init, a harness bug, a cache lookup by the wrong device key, first reported the mask
  as different; fixed in `f71aca47`.)

**More at init** (jobs 36324-36326):

- **1.5, one optimizer step, passes (exact item).** Tier 2 and the ordinary path take one
  step from the same state with the same gradients:
  - the total norm is identical, 2.2621946 on both, so the clip coefficient is identical;
  - every parameter, bank masters included, agrees to 8.5e-8 relative;
  - `exp_avg` agrees to 9.9e-8 and `exp_avg_sq` to 1.3e-7. That is fp32 rounding,
    foreach against fused AdamW;
  - the compute copy equals bf16(master) bit for bit after the step.
- **X3 (the CE on bf16 logits) is not the source of AME's stretch.** Production with the
  CE on fp32 logits (`*_prod_ce32`) moves the gradient by 0.02-0.18% in both arms, and
  every projection stays where it was. Logits at init spread about N(0, 1), so their bf16
  rounding is unbiased; the hypothesis in 1.6 above is refuted. Precise losses: AME
  production 11.230237 against the oracle's 11.230283, and dense 11.199529 against
  11.199493.
- **1.2, one layer at a time, at init, all 12 layers** (`layers --states init`). The
  kernel's inputs are the production prep's own bf16 q, k, v; the upstream is the layer's
  true gradient.
  - **flex + the LSE merge against the exact fp32 function of the same inputs:** out
    2.4-2.5e-3, dq 3.3-3.6e-3, dk 3.2-3.4e-3, dv 2.3-2.7e-3, d`k_sink` 0.8-1.2e-3,
    d`v_sink` 2e-7.
  - **A plain bf16 reference on the same inputs:** out 2.0-2.4e-3, dq 3.0-3.1e-3, dk
    3.2-3.3e-3.
  - **SDPA flash on the same inputs** (dense's kernel, causal, no bias or sink): dq
    2.1-2.4e-3, dk 2.2-2.3e-3.
  - So flex's backward error is 1.5× flash's and 1.1× the plain bf16 reference's:
    rounding-sized, inside 2× dense's kernel budget.
  - Signed projections are ≤ 1.4e-4.
- **A5's named risk is ruled out numerically.** A loss that depends on flex's LSE alone
  gives dq with the reference's norm (233.777 against 233.778), with 2.1e-3 relative
  error, and the LSE itself matches to 6e-8. The gradient through the LSE is there, and it
  is right.
- **The sink's weight at init** ranges over heads from 0.26 (the steepest) to 0.003 (the
  flattest), with a mean of 0.06 in every layer. The large-sink case is tested at the
  final checkpoint.
- **1.4, the bf16 prep, at init:**
  - production's q and k reach the kernel 2.3e-3 from fp32, against 1.66e-3 for an fp32
    prep cast once at the end;
  - 25-34% of the elements differ by about an ulp;
  - the prep's own gradients differ from fp32 by alpha 1.4-2.5e-3, tau 1.1-2.5e-3 and bank
    rows 4.5e-4;
  - tau is exactly 8.0 at init, so its bf16 rounding there is zero.

**1.2 and 1.4 at the final checkpoint** (`layers --states final`, job 36364, all 12 layers,
32 × 2048, the trained weights; each layer's real input and upstream gradient):

| | out | dq | dk | dv | d`k_sink` | d`v_sink` |
|---|---|---|---|---|---|---|
| flex + LSE merge against the exact fp32 function of the same bf16 inputs | 2.2-2.6e-3 | 3.5-5.4e-3 | 2.8-4.1e-3 | 2.4-2.8e-3 | 1.8-4.0e-3 | 0.6-1.5e-3 |
| plain bf16 reference on the same inputs | 2.5-4.9e-3 | 3.6-12e-3 | — | — | — | — |
| SDPA flash on the same inputs (causal, no bias or sink) | — | 2.1-5.2e-3 | — | — | — | — |

- **The kernel is at or below the plain implementation's rounding.** In trained layers
  with sharp attention, the plain bf16 reference (scores rounded to bf16 before the
  softmax) reaches dq errors of 1.2e-2. Flex keeps its scores in fp32 and stays at
  3.5-5.4e-3, within 1.0-1.5× of SDPA flash.
- Its signed projections are ≤ 6e-4 for dq and dk, and ≤ 1.2e-3 for d`k_sink`.
- **The LSE gradient is right in every layer** (LSE-only loss: dq 1.9-2.1e-3) at the
  sink's trained weight. That weight is now 0.14-0.24 of each layer's attention mass,
  against 0.06 at init, so the large-sink case the plan asked for is covered.
- **The whole layer** (prep + kernel + merge + projections), production against standard
  mixed precision, both against fp32:
  - out 3.4-4.9e-3 against 3.4-4.8e-3;
  - dx 4.2-5.4e-3 against 4.2-5.3e-3;
  - bank-row gradients 4.6-8.9e-3 against 4.6-8.8e-3.

  The efficiency path adds nothing measurable at the layer's boundary.
- **The prep (1.4):**
  - production's q and k reach the kernel 2.33e-3 from fp32, against 1.66e-3 for an fp32
    prep cast once, in every layer: A2 adds about 40% to that rounding;
  - **the trained tau (8.0-8.9 per layer) is off by up to 0.32-0.37% in its bf16 forward
    copy**, half the 0.0625 grid, as predicted;
  - the prep's own gradients differ from fp32 by alpha 1.1-2.0e-3, tau 1.3-2.2e-3 and
    bank rows 4.5e-4.
- **Per-layer tau and alpha gradients carry larger projections on the production path**:
  layer 0, dtau +5.8e-3 against +2.8e-3, dalpha +3.7e-3 against 0. A layer's tau has 12
  elements, so its projection noise is about 4e-3; the 144-element tau group in the
  final-state budget is the measure that counts.

## Phase 2: training gates

### The 300-step set (jobs 36309, 36313-36317)

`documents/scripts/parity_gates.py --set 300`. The loss gap is paired by step against
G0a, on the same batches.

| Gate | W&B | `final_ce` | last-50 gap vs G0a | same sign | all-steps gap | s/step |
|---|---|---|---|---|---|---|
| G0a, production | `a6ndymxf` | 4.87109 | — | — | — | 3.38 |
| G0b, production | `g0l9pgwu` | 4.81055 | −0.0613 | 50/50 | −0.0644 | 3.37 |
| G0c, micro-batch 16 | `g6vo1tr2` | 4.83594 | −0.0244 | 38/50 (12 equal) | −0.0292 | |
| **G1, `mem_parallel_size: null`** | `2r6ntg3r` | **5.07813** | **+0.2113** | **50/50** | +0.1359 | 3.45 |
| GD0a, dense | `714ema2v` | 5.17188 | | | | 2.97 |
| GD0b, dense | `xvtbv0aj` | 5.19531 | +0.0250 vs GD0a | 37/50 (13 equal) | +0.0508 | |

- **The run-to-run floor is large.** Two identical production runs end 0.061 apart, with
  one sign over all 50 last steps. They diverge chaotically in steps 25-50 (G0b − G0a is
  −0.125 in that window, dense's twins +0.150) and keep the offset.
  - So "one sign in 45 of 50 steps" carries no information on its own here; the size
    against the floor does.
  - F = max(0.061, 0.024) = 0.061, and the harmful line is 3F = 0.184.
- **G1 crosses that line, in the direction nobody expected.** With A1-A3 off (the banks as
  ordinary fp32 parameters in the fused backbone AdamW), AME trains **0.21 nats worse**
  over the last 50 steps.
  - Its `final_ce` is 5.078, against 4.811-4.871 for the three production runs. That is
    0.24 above their mean and about 8 standard deviations of their spread.
  - The gap opens at steps 25-50 (+0.12) and grows to +0.24 by step 300, while the
    production runs converge to within 0.06 of each other.
- **The efficiency path is therefore not costing AME at 300 steps. Switching it off
  costs.** Under the plan's rule this is a one-directional effect beyond the floor (the
  script prints "harmful: True"), and its sign is the reverse of the risk the audit was
  written for.
- **H0(b) is rejected for A1-A3 together**: switching them off moves a short run's loss
  one-directionally, beyond run-to-run noise.
- **Superseded at 22:45, see "Phase 3: what the bisect found".** G1a computes exactly what
  G1 computes and finished with production, and G0e, a production copy, finished with G1.
  The two-run floor was too small. The G1 gap is the gates' own chaotic spread, and the
  two bullets above are withdrawn.
- **What G1 does differently** (the logged readouts, steps 100 and 200). Production's two
  copies agree with each other in layers 0-2, and G1 departs from both:
  - layer 2's tau is 8.264 against 8.106 and 8.082;
  - layer 2's attention entropy (normalized) is 0.605 against 0.636 and 0.640;
  - layer 2's top-8 mass is 0.486 against 0.442 and 0.435;
  - layer 1 shows the same pattern in alpha_2 and in entropy.

  In G1 the early layers sharpen faster. Layers 3-11 agree across all three runs.
- **The step-10 spike is the same in every AME gate** (47.0-48.6). F3's spike is therefore
  not an artifact of Tier 2. Dense's twins reach 9.5 and 15.5 there.
- **Cost of the ordinary path at h12, one GPU:** 3.45 s/step against 3.38 (+2%).

**Phase 3, started (plan: "If G1 flags").** Two opt-in switches, unset by default, with
`tests/test_parity_flags.py` proving the default bit-identical (`9f4d6384`). A third,
`parity_prep_fp32` (bf16 rows, fp32 prep), was added in `f4b17a5d` and has not been
needed so far:

- **G1r**: G1 again. Is the gap systematic?
- **G1a**: Tier 2 with an fp32 compute copy (`parity_bank_compute_dtype: float32`): A2
  off, A1 and A3 on.
- **G1b**: Tier 2 as run, with the fused gather and its scatter outside the compiled block
  (`parity_bank_eager: true`): A3 off.

1.5 already shows the two optimizer paths step identically from the same state and
gradients. If G1a lands with G1, the bf16 compute copy (A2) is what trains AME faster.
If G1a lands with production, the cause lies in A1 or A3.

### Phase 3: what the bisect found

**The gates** (last-50 mean train loss, `final_ce` in brackets):

| Gate | What | Last-50 | Early bumps (step: rise above the previous 5) |
|---|---|---|---|
| G0a | production | 5.042 (4.871) | 29-33, up to +0.34 |
| G0b | production | 4.981 (4.811) | none |
| G0d | production | 5.020 (4.846) | 29-31, up to +0.22 |
| **G0e** | **production** | **5.223 (5.045)** | 36: +0.16 |
| G0c | production, micro-batch 16 | 5.018 (4.836) | none |
| G1 | ordinary path | 5.253 (5.078) | 31, 37 |
| G1r | ordinary path | 5.284 (5.109) | 27, 30 |
| **G1a** | **Tier 2, fp32 compute copy** | **4.975 (4.805)** | none |
| G1b | Tier 2, eager gather | 4.868 (4.695) | none |

**The gradient twin** (`budget --base ame_std`, job 36345, at init, 256 sequences on
each batch):

- **Tier 2 with an fp32 compute copy computes the ordinary path's gradient exactly.**
  Every group agrees bit for bit except the banks, which differ by 1.8-2.0e-6 relative:
  fp32 summation order, `index_add_` into `main_grad` against `nn.Embedding`'s compiled
  backward. The losses are identical too.
- **The eager gather reproduces production's gradient bit for bit.**
- With 1.5 (one optimizer step identical to 1.3e-7), **A1 and A3 are exact.** The whole
  difference between production and the ordinary path, 0.3-0.4% relative (≤ 4.4e-3 for
  alpha), comes from A2, the bf16 compute copy and the prep it puts in bf16.

**So G1a and G1 are the same computation, yet one finished at 4.975 and the other two
copies at 5.253 and 5.284.** G0e, an identical production copy, finished at 5.223.

- **The 300-step outcome is chaotic.** Runs of one computation spread by about 0.3 nats
  (sd about 0.1-0.15), and nothing in a run's first 50 steps predicts where it ends.
- **G1's +0.21 is that spread**, not an effect of switching Tier 2 off. Under the plan's
  rule it looked "harmful" only because F came from two runs.
- **Correction to the record above:** the "harmful: True" for G1 is withdrawn, and H0(b)
  is *not* rejected for A1-A3.
- **The right test is between computational classes:** A2 on (production and G1b) against
  A2 off (the ordinary path and G1a), with enough runs in each.
  - Replicates are running: G1r2-G1r4 and G0f-G0g.
  - They give about seven runs per class. That bounds A2's effect at 300 steps to roughly
    ±0.1 nats.

### The 220-step set: G-REF (jobs 36319, 36327, 36329, 36330)

| Run | Last-50 | `final_ce` | Early bumps |
|---|---|---|---|
| G0a-220 | 5.664 | 5.480 | 27-45 |
| G0b-220 | 5.596 | 5.402 | 31-41, up to +0.47 at step 38 |
| G0c-220, micro-batch 16 | 5.504 | 5.309 | 42 |
| **G-REF**, A1-A7 off | **5.537** | **5.340** | 29-30 |

- **G-REF lands inside the production spread** (0.17 at 220 steps), at −0.127 against
  G0a. Not harmful: the floor gives a harmful line of 0.478.
- **It sits with the faster production runs**, even though it uses the ordinary bank
  path like G1. That is more evidence that G1's position was noise.
- **G-REF trains at 9.47 s/step against production's 3.36**: the efficiency work buys
  2.8× at h12 on one GPU.
- **Its step-10 spike is 45.4**, against 46.7-48.8 for production. F3's spike is
  therefore not caused by flex, the LSE merge or Tier 2.
- A second G-REF (G-REF2) and a fourth production copy at 220 steps (G0d-220) are
  running.

## Decisions

- **2026-11-06 21:20. Submit the first gate and the G-REF probes before the harness
  exists.** The gates are independent of the harness, and G-REF's micro-batch and
  duration decide whether every gate has to be shortened (plan, Phase 2). So they
  come first.
- **2026-11-06 21:41. G-REF runs 220 steps at micro-batch 8, and its comparison gates
  run 220 steps too.**
  - Micro-batch 16 does not fit (36310), and micro-batch 8 runs at 9.43 s/step
    (36311).
  - 300 steps would be about 47 min of training alone, past the 45-minute limit.
  - 220 steps comes to about 40 min of wall time: 2.5 min startup, 24 s for step 0,
    219 × 9.43 s, three evals of about 36 s each, and 1 min of `final_ce`. That leaves
    a 5-min margin.
  - By the plan's rule, its partners run the same schedule (warmup 38, cosine to 10% at
    step 220): G0a-220, G0b-220 and G0c-220.
  - The 300-step set stays as planned for G1 and for the dense floors.
- **The plain implementation is 2.7× slower.** G-REF's 9.43 s/step against G0a's
  3.49 s/step is the price of AME without its efficiency work at h12 on one H100: the
  reference attention at micro-batch 8, plus fp32 banks in the backbone optimizer.

## Changes to the plan

- **2026-11-06 22:00. The projection flag is set relative to dense, not at a flat 1e-3.**
  - The plan flags a tolerance item when "its signed projection exceeds 1e-3 in
    magnitude".
  - At the final checkpoint, dense's own production path exceeds that in three groups:
    head −6.3e-3, norm +4.2e-3, v_proj −2.2e-3 (job 36321). The flat rule would flag the
    standard stack the audit judges against.
  - The rule used: an AME group is flagged when |projection| is over 1e-3 *and* over twice
    dense's largest |projection| in the same state and batch. This mirrors the plan's own
    rule for the relative error.
  - Both readings are reported.

## Questions for Maciej

Each question is numbered and dated and gives the context, the options and the default
being taken.

**Q1 (2026-11-06 23:05): h100a's `/tmp` is full. May I delete my own stale caches
there?**

- **Context.**
  - h100a's `/tmp` (100 GB) reached 100% at about 22:40 and killed five gates and G-REF2
    with Errno 28 in the Triton cache (jobs 36346-36351).
  - About 10.3 GB of it is `mstefaniak`'s: Triton caches (`/tmp/triton-cache-<job>-rank<n>`,
    `/tmp/triton_gate_<job>`) left by finished jobs, mostly from the 2026-11-05/06
    optimization runs, plus 1.7 GB of `/tmp/torchinductor_mstefaniak`.
  - Most of the rest cannot be attributed from top-level ownership; it may be files held
    open by running processes. Other users' files are not mine to touch.
  - The tool sandbox refused the deletion of even my own files on a shared node without
    your say-so.
- **Options:**
  - (a) Delete `mstefaniak`'s stale `triton-cache-*` and `triton_gate_*` there. That frees
    about 10 GB for every user of the node.
  - (b) Leave them.
- **Default taken: (b).** Every parity job now puts its caches on
  `/storage_nvme_4/nano/mstefaniak/parity_tmp/<job>` (546 GB free), in `add22fd5`, and
  never touches `/tmp`. Other users' jobs on h100a may still fail until someone frees
  `/tmp`.
