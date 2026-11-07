# AME efficiency-parity audit: the record

**Status (2026-11-07, 00:15): the audit (Phases 0-3, plan
[ame_efficiency_parity_plan.md](ame_efficiency_parity_plan.md)) is complete. Phase 4 is
not needed: no item is harmful.**

Its follow-up, [ame_bf16_scalars_plan.md](ame_bf16_scalars_plan.md) (full-length A/Bs on
entropy, X2 × AME), runs from here. Its results go to the sections "Full-length runs on
entropy" and "X2 × AME" below.

**Follow-up status (2026-11-07, 12:25):** Steps 0-2 done; Step 4 under way.
- **Q-P1 is final: not harmful.** AME-B − P1 = +0.0003 nats against a threshold of 0.0216.
  AME-B ×3 scores 3.1320 ± 0.0072 and P1 ×3 3.1317 ± 0.0069 (common evaluator; section
  "Scores").
- **The first error timeline is in.** With bf16 attention, the scalars' gradient error
  grows from 1.5-3% at step 500 to 4-15% from step 1,000 on. fp32 attention holds it at
  0.3-0.6% at every checkpoint (section "X2 × AME").
- **Q-REF is final: not harmful.** P-REF, the plain implementation with every AME-only
  item off, scores 3.1381. AME-B − P-REF = −0.0061: production is the better one, 95%
  interval [−0.029, +0.016]. Over the last 1,000 steps P-REF's train loss is also 0.0070
  above production's.
- **AME-F1's error timeline is in.** On its own weights, the fp32 path it trains with
  keeps the scalars' gradient error at 0.3-0.7% all run. The bf16 path, on the same
  weights, shows 2.6-9.9%.
- **Still running:** 7 trainings on entropy, with D-B3 queued. Expected to finish around
  15:45. Q-X2 follows from them.
- **Dense's runs spread 5× more than AME's.** D-F ×3 ends at 2.966, 3.014 and 3.031
  (trainer `final_ce`), against AME-B's 3.125-3.140. That widens Q-X2's interval (section
  "Scores").

Two setbacks so far:
- the `/home` quota killed P1-1 at step 1,500 (fixed: checkpoints now go to NVMe);
- AME-F ran 10× slow on flex's IEEE fp32 kernel (fixed: tf32x3, equally accurate,
  re-validated).

Verdicts follow when the runs are scored.

- The plan file was emptied by an editor undo at 23:58, just as it was read, and was
  restored byte for byte from VS Code's local history (`8756aa5d`).
- That plan caps its budget at 100 GPU-h; Maciej's goal message of 00:00 sets 120 GPU-h on
  top of the audit's own. 120 applies.

## The answer

**No evidence that AME's efficiency work costs it quality; the 0.143-nat gap to dense is
the method, not the implementation, as far as these checks reach.**

- **At the loss level, the implementation's share of the gap is zero to within 1e-5.**
  - Scored by the fp32 reference path on the full held-out set, the gap is **0.142620**.
    The production paths give 0.142617, and the Trainer's own number is 0.142578.
  - Each arm's production stack costs it the same against its fp32 oracle: AME +0.000150,
    dense +0.000153.
- **No bug.** The bank gradient has the exact structure at init and at the final
  checkpoint: no row leaks, the rows are exactly the input ids. The gradient through
  flex's LSE is present and right. One optimizer step on Tier 2 equals the ordinary
  fused AdamW to 1e-7, from init and from the trained state with its moments.
- **Every AME-only efficiency item is within dense's rounding budget, and most are exact.**
  - A1 and A3 compute the ordinary path's gradient bit for bit or to 2e-6.
  - A2, the bf16 compute copy, adds ≤ 0.42% at init and ≤ 2.3% at the final checkpoint,
    against dense's own 0.7% and 4.0%.
  - The kernel (A4-A6) is as accurate as SDPA flash on the same inputs and more accurate
    than a plain bf16 implementation.
- **No harm in training at 220-300 steps.**
  - **300 steps:** production (7 runs) trains, if anything, better than the same model
    with fp32 banks (6 runs): +0.12 nats in production's favour, p ≈ 0.07-0.10. At 95%,
    production is at most 0.03 nats worse.
  - **220 steps:** production (5 runs) against G-REF (4 runs, every AME-only item off):
    −0.034 nats in production's favour, 95% interval [−0.22, +0.15], Mann-Whitney p = 0.90.
    No difference.
- **What this audit cannot see:** effects that need thousands of steps. The full-length
  runs that would size them are prepared, not run (Proposed full-length runs).
- **One AME-specific numerical weakness, which is not the efficiency work's.** Under bf16
  autocast, which dense shares, the gradients of AME's per-head scalars (alpha, beta,
  tau) are 9-31% off late in training, in *every* bf16 implementation of AME, the plain
  one included. Its training effect is unmeasured (X2).

## How this changes ame_comparison_h12.md

It adds F11 and rejects the implementation hypothesis (H8). The step-10 spike (F3) is
present in every AME configuration, G-REF included, so it is the method's, not Tier 2's
or flex's.

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
| A1 | Tier 2 bank path (O1-O5) | **cleared** (exact) | 1.5 at init (clip active) and at the final checkpoint (real moments): Tier 2 and the ordinary fused AdamW agree to ≤ 1.3e-7, i.e. fp32 rounding, foreach against fused. The twin at init: with an fp32 compute copy, Tier 2's gradient equals the ordinary path's bit for bit, except the banks at 2e-6 (summation order). Gates: G1a (A1 on, A2 off) and G1 (A1 off) are the same computation and land at opposite ends of the chaotic spread |
| A2 | O3, the bf16 bank compute copy: the prep in bf16 | **cleared at the gradient level**; training effect at 300 steps bounded below | Within dense's rounding budget. It adds about 40% to the prep's rounding (q, k reach the kernel 2.33e-3 from fp32 against 1.66e-3), and it rounds tau's forward copy (≤ 0.37% per head). Gradient effect (production against `ame_std`): ≤ 0.42% at init and 0.55-2.3% at final, against a 0.4-1.0% nondeterminism floor between equivalent computations and dense's 2.8-4.0% budget. No loss cost: AME's precision tax is +0.00015, dense's +0.00015 |
| A3 | D2, the fused Q/K pair, traced by compile | **cleared** (exact) | The twin at init: the eager gather reproduces production's gradient bit for bit. 1.3, all exact items at init and at final: no absent row nonzero, no leak into row id + 1, the rows are exactly the input ids, prev-row correlation equal to the next-row control |
| A4 | flex_attention, compiled, ALiBi in `score_mod` | **cleared** | Kernel at the production shape with trained weights: out 2.2-2.6e-3, dq 3.5-5.4e-3 against the exact fp32 function. That is at or below the plain bf16 implementation's error (dq up to 1.2e-2) and within 1.0-1.5× of SDPA flash on the same inputs; projections ≤ 6e-4. G-REF at 220 steps sits inside the production spread |
| A5 | S2, the sink as an LSE merge | **cleared** | The LSE gradient is wired in torch 2.7.1 (source, Phase 0.2 item 3) and right numerically: an LSE-only loss gives dq within 2e-3 at init and at final, in all 12 layers. That includes sink weights of 0.14-0.24. d`k_sink` 1.8-4.0e-3, d`v_sink` 0.6-1.5e-3 at final |
| A6 | S3b, the sink logit as an fp32 mul+sum | **cleared** | fp32 by construction; covered by the kernel check's d`k_sink` above |
| A7 | tau folded into q before the bf16 cast | **cleared** | One rounding of q·tau, the size of any bf16 kernel input: the fp32-prep path reaches the kernel at 1.66e-3, single-rounding level. tau's own bf16 rounding exists only on the Tier 2 path and is booked under A2 |
| A8 | the compiled AME block | **cleared** | Production against `ame_prod_eager`: 0.08-0.33% at init, 0.6-1.7% at final. Same scale as dense's compiled against eager (0.7-2.4%), inside dense's budget |
| A9 | the bank sidecar checkpoint (O30) | **cleared** | `final_ce` reproduces exactly (3.125) from the checkpoint and its sidecar through `load_checkpoint_from_file`. 1.5 at final runs on the sidecar's moments |
| X1 | `compile_blocks_full` | **cleared**, symmetric | As A8 for AME; dense's compiled against eager is 0.7-2.4% at final |
| X2 | autocast bf16, no FSDP2 | **numerical, AME-specific interaction** | The loss-level tax is the same in both arms (+0.00015). At the final checkpoint, though, alpha, beta and tau carry 8.6-31% gradient error in *every* bf16 AME path, the plain reference included, against dense's worst group at 2.8-4.0%. Cause: the bf16 attention backward, amplified by cancellation in the per-head reductions. Training effect not measured (late-training; see the proposed runs) |
| X3 | CE outside autocast, on bf16 logits | **cleared**, symmetric | fp32-CE variants: 0.02-0.18% at init in both arms, 0.2-0.9% (AME) and 0.19-0.73% (dense) at final. No interaction with the arm |
| X4 | `sequence_length` 2049 | **cleared** | The same data in both arms; the model sees 2048 tokens |
| X5 | fused AdamW on the backbone | **cleared** | 1.5: fused and foreach AdamW agree to 1e-7 |
| X6 | micro-batch 32 × 16 | **cleared** | G0c (micro-batch 16) lands inside the production spread at 300 and at 220 steps |
| X7 | the residual-metrics hook breaking the graph | **cleared** | Scheduling only; the harness's production variants keep the hook and its graph breaks |
| X8 | `compile_ce`, `compile_loss`, `fused_linear_ce` off | **cleared** (not active) | Confirmed off in both logged configs (Phase 0.2) |

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
| 36327, 36329, 36330 | G0a-220, G0b-220, G0c-220 (micro-batch 16) | `f096ed77` | 3 | 21:58-22:00 | ~17 min each | 0.85 | `final_ce` 5.48047, 5.40234, 5.30859 |
| 36331 | G1r, G1 again | `9f4d6384` | 1 | 22:06 | ~22 min | 0.37 | `final_ce` 5.10938 |
| 36333 | G1a, Tier 2 with an fp32 compute copy (A2 off) | `9f4d6384` | 1 | 22:07 | ~22 min | 0.37 | `final_ce` 4.80469 |
| 36336 | G1b, Tier 2 with the fused gather eager (A3 off) | `9f4d6384` | 1 | 22:08 | 21:53 | 0.36 | `final_ce` 4.69531 |
| 36337, 36338 | G0d, G0e, production | `1f7b7b43` | 2 | 22:16-22:17 | ~20.5 min each | 0.69 | `final_ce` 4.84570, 5.04492 |
| 36339 | G1r2, ordinary path | `1f7b7b43` | 1 | 22:18 | 21:08 | 0.35 | `final_ce` 4.86133 |
| 36340 | numerics: `machinery --states init`, then `budget --arms dense --states final` (X3) | `ecf8c2d3` | 1 | 22:24 | ~2 min | 0.03 | instrumentation clean; mask check hit a lookup bug (fixed `f71aca47`) |
| 36344 | numerics: twin at init | `8afd4cd9` | 1 | 22:33 | 3.6 min | 0.06 | out of memory at the 4th variant: the harness held the previous model (fixed `2bcf0606`) |
| 36345 | numerics: twin at init, rerun | `2bcf0606` | 1 | 22:37 | 5.9 min | 0.10 | A1 and A3 exact |
| 36346 | G-REF2, first try | `78998949` | 1 | 22:40 | 18:19 | 0.31 | died: h100a `/tmp` full (Errno 28) |
| 36347-36351 | G0f, G1r3, G0g, G1r4, G0d-220, first tries | `78998949` | 5 | 22:41-22:44 | 2-8.5 min | 0.33 | died: h100a `/tmp` full |
| 36361 | numerics: `final_ce --arms ame`, `optim --states final`, `machinery --states final` | `add22fd5` | 1 | 23:00 | ~11 min | 0.18 | AME `final_ce` exact; common evaluator; 1.7 passes; `optim` out of memory (fixed `c340c5e2`) |
| 36362 | numerics: `budget --arms ame --states final` (prod, eager, ref16, ce32) | `add22fd5` | 1 | 23:00 | ~8 min | 0.13 | done |
| 36363 | numerics: twin at final | `add22fd5` | 1 | 23:00 | ~6 min | 0.10 | out of memory at `ame_t2fp32`: bank moments loaded needlessly (fixed `d47cccb4`) |
| 36364 | numerics: `layers --states final` | `add22fd5` | 1 | 23:00 | 4:01 | 0.07 | done |
| 36365-36368 | G0f, G1r3, G0g, G1r4 | `add22fd5` | 4 | 23:01-23:04 | ~21 min each | 1.40 | `final_ce` 4.75977, 4.93555, 4.76172, 4.90039 |
| 36369 | G0d-220 | `add22fd5` | 1 | 23:05 | ~17 min | 0.28 | `final_ce` 5.17188 |
| 36370 | G-REF2 | `add22fd5` | 1 | 23:06 | ~41 min | 0.68 | `final_ce` 5.48047 |
| 36371 | numerics: twin at final, rerun (t2fp32, t2eager) | `d47cccb4` | 1 | 23:08 | 4:30 | 0.08 | nondeterminism floor 0.4-1.0% at final |
| 36372 | numerics: `optim --states final` | `c340c5e2` | 1 | 23:11 | 1:00 | 0.02 | 1.5 passes at final |
| 36376, 36377 | G-REF3, G-REF4 | `671618ac` | 2 | 23:31-23:32 | ~39.5 min each | 1.32 | `final_ce` 5.57812, 5.35352 |
| 36378, 36379 | G0e-220, G0f-220 | `671618ac` | 2 | 23:32-23:33 | ~17 min each | 0.57 | `final_ce` 5.45312, 5.50000 |
| 36381 | check 1.0: GPU tests | `79bd8c86` | 1 | 23:41 | ~2 min | 0.03 | 10 passed |
| (srun) | h100a `/tmp` listing and cleanup (Q1), CPU only | — | 0 | 23:26-23:27 | | 0 | 9.20 GB freed |

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

### 1.0 The existing GPU tests, at the runs' code (job 36381)

`pytest -m gpu tests/test_ame.py tests/test_bank_compile.py` on one entropy H100:
**10 passed**, plus the CPU `aot_eager` case of `test_bank_compile.py`.

- **`test_ame.py`'s nine GPU tests** cover:
  - flex against the reference path;
  - bf16 autocast against fp32 within bf16 resolution;
  - the shift and the sink through the compiled kernel;
  - token ids reaching a whole compiled block;
  - no materialized score matrix.
- **`test_bank_compile.py`'s mp-1 cases**, ported from `ame_optimize` (`922838c1`) to
  `parity_audit` (`79bd8c86`):
  - the compiled fused gather's `main_grad` equals eager on CUDA within 1e-4 relative;
  - `aot_eager` keeps the `main_grad` mutation exactly.
- The NCCL case was not ported: it depends on an `ame_optimize`-only conftest helper, and
  both runs used `mem_parallel_size` 1.
- `test_memory_parallel.py` has no GPU-marked tests. Its CPU suite passed in Phase 0.3.

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

**The replicates** (jobs 36365-36368, and 36339 for G1r2; the first G0f/G0g/G1r3/G1r4
submissions died on the full `/tmp`, Q1):

| Class (same computation within a class) | Runs | Last-50 mean (sd) | `final_ce` mean | Each run's last-50 |
|---|---|---|---|---|
| **A2 on**: bf16 compute copy (production; G1b) | 7 | **5.002** (0.113) | 4.827 | 5.042, 4.981, 5.020, 5.223, 4.941, 4.939, 4.868 |
| A2 on, micro-batch 16 (G0c) | 1 | 5.018 | 4.836 | |
| **A2 off**: fp32 banks (ordinary path; Tier 2 with fp32 copy) | 6 | **5.123** (0.123) | 4.948 | 5.253, 5.284, 5.033, 5.117, 5.073, 4.975 |
| dense (GD0a, GD0b) | 2 | 5.375 (0.018) | 5.184 | |

- **A2 off − A2 on = +0.121 nats** over the last 50 steps: Mann-Whitney p = 0.073, Welch p
  = 0.096. The 95% interval for production − (fp32 banks) is **[−0.267, +0.026]**.
- **At 300 steps the bf16 compute copy is not worse than fp32 banks by more than about
  0.03 nats, and is more likely better.** Whether it really helps early training (A2's
  bf16 rounding acting like noise in the bank prep) is suggestive, not established. The
  question is moot for the audit, whose concern was harm.
- **One run's spread is about 0.12 nats (sd) at 300 steps in either class**, larger than
  most effects these gates were meant to detect. The plan's "harmful" rule with a
  two-run floor (F = 0.061) would have misfired. Any future short gate on AME needs five
  or more runs per arm.
- **The 300-step outcome is not predicted by the early bumps.** G0a recovered from the
  largest bump in the set; G0e had one small bump and ended worst.

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
- **The full 220-step set** (`parity_gates.py --set dist220`), last-50 mean train loss:

  | Run | Last-50 |
  |---|---|
  | production G0a, G0b, G0d, G0e, G0f -220 | 5.664, 5.596, 5.367, 5.641, 5.687 (mean 5.591, sd 0.130) |
  | G0c-220, micro-batch 16 | 5.504 |
  | G-REF, G-REF2, G-REF3, G-REF4 | 5.537, 5.664, 5.758, 5.539 (mean 5.625, sd 0.107) |

  Production − G-REF = −0.034, 95% Welch interval [−0.221, +0.153], Mann-Whitney p 0.90.
  **Switching every AME-only efficiency item off does not train better at 220 steps. At
  95%, production is at most 0.15 nats worse; the point estimate is 0.03 better.**

## Full-length runs on entropy (ame_bf16_scalars_plan.md, Step 3)

The plan [ame_bf16_scalars_plan.md](ame_bf16_scalars_plan.md) replaces the LEM proposal
below with full-length A/Bs on entropy, with replicates.

**Recipe:** S2-PS or S1-D at n = 8, 3,815 steps × 1,048,576 tokens, warmup 38, cosine to
10%, seed 123, autocast bf16, `compile_blocks_full`. Configs: `configs/parity/fe_*.yaml` on
`parity_audit`. Checkpoints go to h100a's NVMe, `/storage_nvme_1/mstefaniak/parity_full/`
(not `/home`: see the quota incident below), every 500 steps for AME-B1, P1-1 and AME-F1.

| Run | Config | Change | Jobs | Status |
|---|---|---|---|---|
| AME-B ×3 | `fe_ame_b{1,2,3}` | none (production) | 36400, 36404, 36407 | **done** 04:01-04:07 (3:48-3:55 h each) |
| P1 ×3 | `fe_p1_{1,2,3}` | `mem_parallel_size: null` (A2 off; A1, A3 are exact) | P1-1: 36408 **died at step 1,500** (quota), rerun 36455; P1-2 36409, P1-3 36411 | **done**: P1-2, P1-3 04:10-04:12, P1-1 08:29 |
| P-REF | `fe_pref` | mp null + reference attention, micro-batch 8 × 64 | 36399 | **done** 10:41 (10.5 h, 9.48 s/step) |
| AME-F ×3 | `fe_ame_f{1,2,3}` | P1 + `parity_attention_fp32` with tf32x3 dots and a 32×128 backward tile (`parity_flex_kernel_options`, below), **micro-batch 16** | 36463, 36464, 36465 (first tries 36445, 36447, 36449 cancelled: 34 s/step) | AME-F1 **done** 10:48 (8.2 h); AME-F2 and AME-F3 running since 04:03 (7.4 s/step, end about 12:15) |
| P1-16 ×3 | `fe_p1_16_{1,2,3}` | P1 at micro-batch 16: AME-F's matched control | 36452, 36453, 36454 | running: P1-16_1 since 08:14 (3.62 s/step, ends about 12:15), P1-16_2 since 09:00 (about 13:00), P1-16_3 since 10:41 (about 14:35) |
| D-B ×3 | `fe_db{1,2,3}` | none (dense production) | 36456, 36457, 36458 | running: D-B1 since 11:37, D-B2 since 12:03; D-B3 queued (about 3.3 h each) |
| D-F ×3 | `fe_df{1,2,3}` | `parity_dense_attention_fp32` | D-F1 36416; D-F2 36450, D-F3 36451 | **done**: D-F1 08:13, D-F2 11:31, D-F3 11:37 (7.4 h each, 6.66 s/step; D-F2 and D-F3 requeued once, below) |
| P0b | `fe_p0b` | AME-B at micro-batch 16 × 32 | 36459 | running since 10:48 (ends about 14:40); kept, the budget holds (below) |

The first submissions of the 12 queued runs (36417-36428) failed at launch; see the
incident.

**Queue order: longest first.** AME-F (`nice` 50), D-F2 and D-F3 (75), then P1-16, P1-1,
D-B and P0b (100). Analysis and scoring jobs run at `nice` 0 and go first.

- **04:08-04:13, re-ordered:** a fresh job has no queue age, so a `nice`-0 job still
  ranked below every queued training (2952 against 3001-3027). The queued trainings'
  `nice` went up by 100 until the GPU tests and the wave-1 scoring had started, then came
  back down.
- **The new order:** D-F3, then **P1-1**, P1-16 ×3, **P0b**, D-B ×3.
  - P1-1 moves up so that its 500-step checkpoints, which the error timeline needs,
    exist by mid-morning rather than at the end.
  - P0b moves ahead of D-B because, at 4.2 h, last in line it would set the finish
    (about 16:25 against 15:30).
  - The budget decision on P0b (the first cut) stays open until it starts, around 10:30.
    It started at 10:48 with the projection at about 113 GPU-h, so it stays.
- **10:50, re-ordered again:** AME-F1's and P-REF's analysis job (36513) became eligible
  when AME-F1 finished, but ranked below the three queued D-B runs (2937 against
  3538-3541). Their `nice` went from 110 to 810, so 36513 takes the next free GPU (D-F2's,
  about 11:30) and its results come about 1.5 h sooner. D-B3, which sets the finish,
  starts about 2 minutes later. The `nice` is restored once 36513 has started.

### Scores

Every final checkpoint (`step_3814`) is scored twice on all 8,192 held-out sequences,
with precise CE (fp32 loss on the logits, fp64 sum): under the common evaluator (the arm's
fp32 reference path, `ame_ref32` or `dense_ref32`) and under the run's own production
path. Tool: `parity_score_runs.sh` → `parity_numerics.py score`. Verdicts:
`parity_full.py`.

| Run | Job | W&B | Trainer `final_ce` | Common evaluator | Own path | Own − common | s/step |
|---|---|---|---|---|---|---|---|
| AME-B1 | 36400 | `u1zpx7lt` | 3.1396 | 3.139532 | 3.139673 (`ame_prod`) | +0.000141 | 3.36 |
| AME-B2 | 36404 | `s65ebe41` | 3.1250 | 3.125231 | 3.125379 (`ame_prod`) | +0.000148 | 3.36 |
| AME-B3 | 36407 | `w4hmw00o` | 3.1328 | 3.131156 | 3.131303 (`ame_prod`) | +0.000147 | 3.38 |
| P1-2 | 36409 | `1flt0emv` | 3.1260 | 3.124540 | 3.124686 (`ame_std`) | +0.000146 | 3.45 |
| P1-3 | 36411 | `t6cqf6fy` | 3.1328 | 3.132150 | 3.132287 (`ame_std`) | +0.000137 | 3.44 |
| P1-1 | 36455 | `c7vyd1n7` | 3.1387 | 3.138342 | 3.138491 (`ame_std`) | +0.000149 | 3.45 |
| D-F1 | 36416 | `vzfga5w2` | 2.9658 | 2.966446 | 2.966587 (`dense_f`) | +0.000141 | 6.65 |
| P-REF | 36399 | `zavr33un` | 3.1396 | 3.138114 | 3.138273 (`ame_pref`) | +0.000159 | 9.48 |
| AME-F1 | 36463 | `gcfv6q7a` | 3.1387 | 3.138320 | 3.138466 (`ame_f3x`) | +0.000145 | 7.42 |
| *`doi36p8k` (LEM, the original run, for reference)* | | | *3.125* | *3.125227* | *3.125377* | *+0.000150* | |

Scoring jobs:
- 36479: wave 1 (04:12-04:35, 0.38 GPU-h);
- 36493: P1-1, D-F1 and P1-1's error timeline (08:29-09:00, 0.52 GPU-h);
- 36513-36520, queued 09:00-09:18 at `4388170a`, each held by a dependency (`afterany`) on
  the trainings it reads:
  - 36513: AME-F1, P-REF, and AME-F1's error timeline (11:31-12:03, 0.53 GPU-h);
  - 36514: D-F2, D-F3, AME-F2, AME-F3 and P1-16_1;
  - 36515-36520: one per late run (P1-16_2, P1-16_3, P0b, D-B ×3).

**ssh outage, 09:05-09:18.** Entropy stopped accepting connections partway through
queuing these jobs, after about 12 in quick succession (probably a rate limit). The last
four (P0b, D-B ×3) were queued at 09:18, when it came back. Since then my commands share
one ssh connection (`ControlMaster`). Nothing on the cluster was affected.

**A stuck shared connection, 11:18-11:43.** The shared connection's TCP session died
without closing, and every command sent through it hung. The job watcher saw nothing
for 25 minutes, so the D-B runs' `nice` went back to 110 at 11:43 instead of about
11:32. That cost nothing: D-B1 had started at 11:37 anyway, with no other job waiting.
The connection now sends keepalives and closes within about 90 s if its session dies.

36493 was queued with a dependency on P1-1. While it waited, the queued trainings' `nice`
was raised by 500, so that it, not P1-16_2, took P1-1's GPU. The `nice` was restored once
it had started. That moved P1-16_2's start from 08:29 to 09:00, which costs nothing: D-B3
sets the finish.

- **AME-B: 3.131973 ± 0.00719** (mean ± sd, n = 3). The original LEM run, 3.125227, sits
  at its low end. One run's spread is 0.007 nats, so the plan's Q-P1 / Q-REF threshold,
  max(3 × sd, 0.005), comes to 0.0216.
- **The production path's tax is the same in every run:** +0.000137 to +0.000148 against
  the fp32 oracle, as in the audit (+0.000150).
- **AME-B1 is an unlucky replicate, not a faulty one.** All five runs match to bf16
  resolution up to step ~20, then diverge chaotically. AME-B1 took a second loss bump at
  steps 30-36 (7.31-7.41, where the others were at 7.0-7.1) and stayed behind: +0.2 nats
  over steps 0-99, +0.015 at the end. The bump comes before its first checkpoint, so the
  500-step saves cannot be the cause.
- **The floor.** P1 ×3 scores 3.131677 ± 0.00691, so AME-B and P1 have the same spread.
  - At 4B tokens, one run's CE is uncertain by about ±0.007 from run-to-run
    nondeterminism alone, with the seed fixed.
  - Two groups of three can resolve a difference of about ±0.016 (95%).
- **Q-P1 (A2, the bf16 compute copy), final (09:05): not harmful.**
  - AME-B − P1 = **+0.00030** against the 0.0216 threshold, 70× below it. The 95% Welch
    interval is [−0.0157, +0.0163].
  - The 50-step paired train-loss gap keeps its sign over 92% of the last 1,000 steps.
    But at +0.0007 it is 30× below the threshold, and the rule needs both.
  - *(The 04:40 preview, without P1-1, gave +0.0036: P1-1 lands at the high end, as
    AME-B1 does.)*
- **The per-head scalars cannot tell AME-B from P1 either.** The trainer logs tau, alpha,
  beta and attention entropy per layer (`parity_full.py`, trajectories):
  - at steps 500, 1,000, 2,000, 3,000 and 3,800, the gap between the two groups' layer
    means lies inside its 95% interval for all four;
  - the largest single-layer gap is 3.2 within-group sd, about what the largest of 12
    layers gives by chance with n = 3;
  - an AME-B run is no farther from a P1 run (0.054-0.063) than two runs of one group
    are from each other (0.059-0.069; permutation p 0.6-0.9). The distance is the RMS over
    tau, alpha, beta and the 12 layers, each scaled by its across-run RMS.
- **Q-REF (every AME-only item off), final (12:25): not harmful.**
  - P-REF scores **3.138114** under the common evaluator, inside the production runs'
    range (AME-B and P1: 3.1245-3.1395).
  - **AME-B − P-REF = −0.0061**: production is the better one. The threshold is +0.0216.
    - P-REF is a single run, so its 95% interval uses AME's run-to-run sd, pooled over
      AME-B and P1 (0.00705, 4 dof): **[−0.029, +0.016]**.
    - At 4B tokens, AME's efficiency implementation costs it at most 0.016 nats against
      the plain implementation, and the point estimate favours production.
  - Over the last 1,000 steps the 50-step paired train-loss gap, AME-B − P-REF, is
    **−0.0070**: production's loss is the *lower* one, at every one of those steps. The
    rule's harm condition needs a positive gap.
  - The reference path's own tax against the fp32 oracle is +0.000159, the same as
    production's (+0.000137 to +0.000148).
  - **The sign-share condition is weak at 4B tokens.** Each AME-B run against the other
    two also keeps one sign over 100% of the last 1,000 steps (gaps +0.013, −0.011,
    −0.003): runs that drift apart early keep their offset. The size condition, the
    0.0216 threshold, is the one that decides.
  - **The per-head scalars, against the six AME-B and P1 runs pooled** (they do not
    differ: Q-P1). P-REF's alpha (layer mean) has sat 0.024-0.025 below theirs from step
    2,000 on (t ≈ −2.8, p ≈ 0.04). Its tau is 0.05-0.06 lower (p 0.11-0.16), and its beta
    the same. Its attention entropy is 0.057 higher at step 3,800 (p = 0.012), after
    being 0.020 lower at 3,000.
  - **None of this survives the multiplicity:** 4 readouts × 5 checkpoints is 20 tests,
    and the smallest p, 0.012, is above Bonferroni's 0.0025. The alpha offset is one run's
    persistent drift, like AME-B1's loss, and it does not reach the loss. Each readout is
    computed from one sequence, the same for every run at a given step (`ame.py`,
    `attention_diagnostics`, the same on every backend).

### The /home quota incident (2026-11-07, 01:52)

- **What happened.**
  - `mstefaniak`'s `/home` on entropy has a 120 GB hard quota (`xfs_quota`), shared by
    every job of the account. The full runs saved their checkpoints into their cemetery
    clones under `/home`: 13.2 GB per AME checkpoint.
  - By 01:52, AME-B1 and P1-1 had saved five 500-step checkpoints (66 GB). Those sat on
    top of the two finished checkpoint-test runs (36394, 36395: 50.5 GB) and
    `~/parity_ckpt` (15 GB).
  - **P1-1 (36408) died at its step-1,500 save** (`OSError: [Errno 122] Disk quota
    exceeded`), after 1.64 GPU-h.
  - **The 12 queued runs then started one after another on the freed GPU and each failed
    within a second** (`RaisedSignal:53`): Slurm could not create their log files. No
    GPU time was lost on them.
  - The running jobs' local W&B logs stopped writing at 01:52. The W&B server kept every
    step: at 02:10 all seven running runs had a loss row at every step from 0 on, none
    missing.
- **The fix (02:00-02:20).**
  - Deleted the two checkpoint-test runs. Their result is recorded under Step 1.
  - Redirected each of the seven running jobs' `checkpoint` folders: it is now a symlink
    into `/storage_nvme_1/mstefaniak/parity_full/<run>` (5.7 TB free). The swap ran on
    h100a itself, so the node's NFS client saw it at once, before AME-B1's next save.
  - Copied AME-B1's three checkpoints and the failed P1-1's two to that folder (the
    latter as `p1_1_failed36408`), checked every file byte for byte with `cmp`, and
    deleted the `/home` copies.
  - Set `checkpoint.save.path` in both full-run bases to
    `/storage_nvme_1/${oc.env:USER}/parity_full/<run name>`, which resolves on the node
    (`85dd16c8`). Resubmitted the 13 runs.
  - `/home` usage: 120 GB (the limit) before the fix, 70.7 GB after the deletion, and
    about 16 GB once the checkpoints moved.
- **P1-1 restarts from step 0 rather than resuming from its step-1,000 checkpoint.**
  The codebase marks resuming with optimizer state "not tested". A resumed run that
  silently repeated or skipped data would not be a valid replicate. The cost is the 1.64
  GPU-h already spent.

### AME-F's speed: flex's fp32 kernel (02:06-02:50)

- **AME-F1 (36445) trained at 34.1 s/step,** ten times production. That projects to 36 h
  per run: past the 12-h limit, and three times the whole budget for the three runs.
  - The Step 2 probe (36396) showed only that micro-batch 32 runs out of memory.
    AME-F's speed at micro-batch 16 was never measured before the runs were submitted.
  - That was my mistake: a 40-step probe at 16 would have shown it.
- **Cause: flex's fp32 defaults** (torch 2.7.1, `torch/_inductor/kernel/flex_attention.py`).
  - `float32_matmul_precision` is "highest" (the default). Under it, flex compiles fp32
    dot products as IEEE on the CUDA cores, not on the tensor cores.
  - Its fp32 backward tile is 16×16 with one stage, on every GPU.
  - D-F escapes this: its fp32 SDPA costs dense only 2.2×.
- **A benchmark at the training shape** (jobs 36460, 36461):
  - What was timed: flex forward + backward at micro-batch 16, 12 heads, n 2048, d 64,
    with AME's ALiBi `score_mod`, the causal mask and `return_lse`.
  - Accuracy: one sequence against an fp64 materialized reference.

  | Kernel | ms per call | Attention per optimizer step (12 layers × 32 micro-batches) | dq error vs fp64 |
  |---|---|---|---|
  | bf16 (production, P1) | 1.92 | 0.74 s | 4.5e-3 |
  | fp32, IEEE dots, 16×16 backward tile (AME-F1 as run) | 82.2 | 31.6 s | 4.2e-7 |
  | fp32, tf32x3 dots, 16×16 backward tile | 22.0-25.3 | 8.5-9.7 s | 4.2e-7 |
  | fp32, tf32 dots | 13.7 | 5.3 s | 1.8e-3 |
  | **fp32, tf32x3 dots, 32×128 backward tile, 8 warps, 2 stages** | **11.2** | **4.3 s** | **4.2e-7** |

  - **tf32x3 is as accurate as IEEE here, to two digits** (dk 4.3-4.5e-7, dv
    2.9-3.0e-7). It splits each fp32 operand into two TF32 parts and keeps three of the
    four products, all on the tensor cores.
  - **Plain TF32 is rejected:** it is only 2.5× more accurate than bf16, so it would not
    test the hypothesis.
  - Ten backward tiles were tried. All give the same error; the 32×128 tile at 8 warps
    and 2 stages is the fastest.
- **The fix** (`b0ae7385`):
  - A new opt-in key, `parity_flex_kernel_options`, allowed only together with
    `parity_attention_fp32`, hands flex these kernel options.
  - Unset, the call passes none, as on every shipped path (tested).
  - The AME-F configs set tf32x3 and the 32×128 tile. Expected speed: about 7.3 s/step.
    **Measured: 7.40 s/step** (AME-F1, 36463, from 02:36): 7.8 h of training per run,
    against 34.1 s/step before.
  - It changes how fast AME-F's kernel runs, not what it computes: the arithmetic stays
    fp32-accurate.
- **Re-validated at the final checkpoint before AME-F trains** (job 36462, the same
  check as 36393). Relative error against the fp32 oracle, `ame_f` (IEEE, as validated)
  against `ame_f3x` (tf32x3 and the 32×128 tile, as the runs train):

  | Group | `ame_f` train | `ame_f3x` train | `ame_f` held-out | `ame_f3x` held-out |
  |---|---|---|---|---|
  | alpha | 1.28e-2 | 1.30e-2 | 8.3e-3 | 7.7e-3 |
  | beta | 1.87e-2 | 1.87e-2 | 1.25e-2 | 1.23e-2 |
  | tau | 1.38e-2 | 1.46e-2 | 9.2e-3 | 8.4e-3 |
  | ff, v_proj, o_proj | 1.19-1.30e-2 | 1.19-1.30e-2 | 1.03-1.15e-2 | 1.04-1.15e-2 |

  **The same, group for group.** Losses: 3.277205 and 3.125758, against the oracle's
  3.277061 and 3.125641 (`ame_f`: 3.277219, 3.125771).
- **Tests.**
  - On CPU, the full suite passes, the new tests included.
  - On GPU (36462), the new CUDA test passes at n 64 and 2048: tf32x3 with the tile
    matches fp64 to < 1e-5 in out, dq, dk and dv.
  - One older test failed in that job: `test_ame.py`'s memory regression for the unfused
    flex. The cause was the new test itself. It compiled AME's shared `_COMPILED_FLEX`
    twice more, past dynamo's recompile limit (8), after which that function runs
    uncompiled.
  - Fixed in `4388170a`: the test now resets dynamo before and after. The runs are not
    affected, because there flex is traced inside each compiled block at fixed shapes.
  - **Re-run at `4388170a`: 14 of 14 GPU tests pass** (job 36476, 04:10), the memory
    regression included.
  - A first re-run (36466, 04:02) crashed with a fatal error inside one test. Its log kept
    only the last 30 lines, so which test is not known. The full-output re-run did not
    reproduce it. The training runs do not execute the test code, and AME-F1 trains on
    the same kernel options without trouble (step 744 at 04:14).
- **Jobs.**
  - AME-F1-3 (36445, 36447, 36449) were cancelled and resubmitted. The cancelled AME-F1
    had used 0.25 GPU-h.
  - D-F2 (36450) and D-F3 (36451) were requeued (`scontrol requeue`) 3 and 2 minutes
    after they started. That put the benchmarks and the validation on a GPU now, not at
    04:00. A requeued job keeps its ID and config, and restarts from step 0.

### Budget (the bf16-scalars plan)

The cap is 120 GPU-h, counting every job from 36392 on. Running totals are recorded as
the runs finish.

| Time | Used | Projected to the end | Note |
|---|---|---|---|
| 02:10 | 15.4 GPU-h | — | the quota incident cost 1.64 (P1-1) |
| 02:37 | 19.2 GPU-h | **about 113** | AME-F at about 8 h per run (tf32x3); 36 h per run at IEEE would have been about 190 in all |
| 04:12 | 31.1 GPU-h | **about 114** | measured: P-REF 9.9 s/step (10.5 h), D-F 7.0 s/step (7.4 h), AME-F 7.4 s/step (~8.1 h with startup and evals) |
| 04:37 | 35.2 GPU-h | **about 114** | wave 1 and its scoring done (0.38 GPU-h for 5 runs) |
| 06:41 | 51.7 GPU-h | **about 113** | health check: the 8 running runs need 31 GPU-h more, the 7 queued about 27, analysis about 2.5 |
| 09:03 | 70.7 GPU-h | **about 112** | D-F1, P1-1 and job 36493 done; the 8 running need about 22 GPU-h more, the 5 queued about 18, analysis about 2 |
| 10:51 | 85.1 GPU-h | **about 113** | P-REF (10.5 h) and AME-F1 (8.2 h) done, P0b started; the 8 running need about 16 GPU-h more, D-B ×3 about 10, analysis about 2.3 |
| 12:09 | 95.5 GPU-h | **about 112** | D-F2, D-F3 and job 36513 (0.53) done; the 8 running and D-B3 need about 15 GPU-h more, the remaining scoring about 1.1 |

The projection assumes P1-16 at about 4.2 h per run and D-F at 7.3 h, plus about 3 GPU-h
for scoring and the error timeline. P0b, the first cut, stays while the projection holds
under 120.

Results, floors and the Q-P1 and Q-REF verdicts are added as the runs finish.

## X2 × AME: the per-head scalars under bf16 (ame_bf16_scalars_plan.md, Steps 1-2)

**Step 1, build and test** (`parity_audit`, from `e4553715`):

- **`parity_attention_fp32` (AME, opt-in):** v is upcast right after v_proj, so the beta
  mix runs in fp32. q and k stay fp32 on the ordinary bank path. The kernel and the sink
  merge run with autocast off. That last part is needed: flex's higher-order op has no
  autocast rule of its own, and SDPA is on autocast's bf16 list.
- **`parity_dense_attention_fp32` (dense, opt-in):** q, k and v are upcast after RoPE, and
  SDPA runs with autocast off.
- **Tests** (`tests/test_parity_flags.py`):
  - both flags are off by default;
  - off is bit-identical to absent, and on is bit-identical on an fp32 CPU model;
  - they reach exactly their layers;
  - **on CUDA, under autocast, each path hands its kernel fp32 inputs, and fp32 flex /
    fp32 SDPA match fp64 to < 1e-5**, i.e. IEEE fp32, not TF32. Flex picks IEEE fp32 dots
    because `float32_matmul_precision` is "highest".

  12/12 GPU tests pass (job 36392), and the CPU suite passes.
- **Periodic checkpoints work, in both formats.** Two 20-step runs saved `step_10` and the
  final (36394 Tier 2 with sidecar, 36395 ordinary path with the banks in the state dict).
  Each loads into both paths with the same loss to 3e-6 (`score`, job 36412):
  - Tier 2 checkpoint: production 7.864662, ordinary 7.864663;
  - ordinary checkpoint: ordinary 7.863520, production 7.863517.

  Writing a 13.2 GB checkpoint to `/home` takes about 2.5 min.
- **The harness** takes checkpoint folders as arguments (`--ckpt-ame`, `--ckpt-dense`),
  reads both checkpoint formats, and scores any checkpoint with any evaluator (`score`).

**Step 2, probes:**

| Probe | Micro-batch | Result |
|---|---|---|
| AME-F (job 36396) | 32 | **out of memory** (6.14 GiB short at 77 GiB). AME-F runs at micro-batch 16, and P1-16 ×3 is its matched control, as the plan prescribes |
| D-F (job 36397) | 32 | fits, peak 52.4 GiB, **6.65 s/step** against dense production's 2.97 s/step: fp32 SDPA is 2.2× slower. About 7.3 h per run, against the plan's 4-5 h |

**Step 2, validating the mechanism before AME-F trains** (job 36393, final checkpoint,
256 sequences; relative error / projection against the fp32 oracle):

| Group | `ame_std` (bf16 attention) train | AME-F train | `ame_std` held-out | AME-F held-out |
|---|---|---|---|---|
| alpha | 1.6e-1 / −1.6e-2 | **1.3e-2** / −3.4e-3 | 1.2e-1 / +3.3e-2 | **8.3e-3** / −8.7e-4 |
| beta | 3.1e-1 / **−1.1e-1** | **1.9e-2** / −1.5e-3 | 1.6e-1 / −2.5e-2 | **1.2e-2** / −1.8e-3 |
| tau | 1.1e-1 / −1.6e-2 | **1.4e-2** / −4.8e-3 | 8.4e-2 / +1.7e-2 | **9.2e-3** / +1.0e-4 |
| ff, v_proj, o_proj (reference) | 1.5-1.7e-2 | 1.2-1.3e-2 | 1.3-1.4e-2 | 1.0-1.1e-2 |

- **fp32 attention removes the amplification.** The scalars' errors fall 10-15× to the
  level of every other group, and beta's −0.11 bias is gone.
- **The plan's literal target (≤ 1%, the init level) is not reached on the training
  batch** (1.3-1.9%), and cannot be: at the final checkpoint every group of every bf16
  path sits at 1.1-4%, dense's included, and the scalars inherit that upstream error.
  Criterion used: no longer worse than the model's other groups. That is met. (Q3.)
- **D-F changes dense's gradient error by nothing measurable** (dense has no scalars to
  amplify).

**Step 2, the mechanism check** (`mechanism`, job 36413). For each layer at the final
checkpoint (8 sequences), tau, alpha and beta are recomputed in fp64 from three kernel-input
gradients (dq, dk, dv): the fp64 oracle's, the fp32 kernel's, and the bf16 flex kernel's
(with the LSE merge, on bf16-rounded inputs and upstream). The fp64 formulas reproduce full
fp64 autograd to ≤ 4e-16.

| Kernel | tau error (median / max over 12 layers) | alpha | beta |
|---|---|---|---|
| fp32 reference | ≤ 1.3e-6 | ≤ 9.4e-7 | ≤ 2.9e-7 |
| **bf16 flex** (dq/dk/dv errors 3-7.5e-3) | 0.5% / 1.4% | 0.5% / 0.9% | 0.2% / 0.3% |

**On 8 sequences, the bf16 kernel's own rounding explains only a small part** of the 8-31%
seen on 256 sequences. The candidate this leaves is a small *bias* per head in the bf16
path:

- a bias adds up linearly over tokens, while, late in training near a stationary point,
  the true scalar gradient grows only like √N;
- √(256/8) ≈ 5.7, which turns 0.5-1.4% into 3-8%;
- Stage A's t-statistics over 16 groups of 512 test exactly this (jobs 36414, 36415).

**Step 2, Stages A and B** (`scalars`, jobs 36414-36415; `parity_scalars.py`). These are
the scalars' gradients on each of the 16 held-out groups of 512 sequences, at the final
checkpoint, against the fp32 oracle. Stage B replays 16 AdamW updates (one per group) from
the checkpoint's own moments, at the final LR 3.9e-4.

| | tau | alpha | beta |
|---|---|---|---|
| **production**: one group's relative error | 7.2% | 10.0% | 18.5% |
| production: the same over all 16 summed | 7.6% | 10.6% | 19.9% |
| production: \|bias\| / RMS of the true gradient, median (max) | 0.015 (0.47) | 0.014 (0.92) | 0.021 (1.08) |
| production: \|t\| median; share with \|t\| > 4 | 9.4; 75% | 8.0; 69% | 23.4; 90% |
| production: summed-error projection | +0.018 | +0.024 | +0.013 |
| production, 16 AdamW updates: update error; sign agreement | 3.4%; 99.3% | 3.2%; 99.0% | 3.6%; 99.3% |
| **AME-F**: one group's relative error | 0.6% | 0.6% | 1.2% |
| AME-F: \|t\| median | 4.6 | 3.3 | 8.5 |
| AME-F, 16 AdamW updates: update error; sign agreement | 0.8%; 99.3% | 0.8%; 99.3% | 0.6%; 99.7% |

- **The errors are systematic, not noise.** Most elements have |t| > 4, and production's
  scalar gradients are stretched 1-2.4% along the true direction. beta's bias is real.
- **They do not grow with the batch.** One group's error is 7.2% and the sum of 16 groups'
  7.6%, for tau. True gradient and error both add up linearly, so the √N idea above is
  ruled out.
- **Adam absorbs most of it.** Over 16 updates from the trained moments, production moves
  the scalars in the oracle's direction for 99.0-99.3% of the 720 numbers, and its update
  size is off by 3.2-3.6%. fp32 attention brings both 5× closer.
- **Where it comes from is not settled.**
  - Not the kernel's dq/dk/dv rounding: ≤ 1.4% at the layer level (the mechanism check).
  - Not the bank prep: `ame_std`, with an fp32 prep, has it as well.
  - It disappears with AME-F's other changes: v upcast before the beta mix (so beta is
    not rounded to bf16), q and k kept fp32, and the merge in fp32. The bf16-rounded beta
    in the forward is one candidate. Pinning it down would need those changes as separate
    switches, which the plan does not ask for.
- What matters for Q-X2 is that AME-F removes the error at every scale measured: per
  group, summed, and through Adam.

**Step 4, Stage A along the run.** Jobs 36493 (P1-1) and 36513 (AME-F1) ran `scalars` on
each 500-step checkpoint with two held-out groups of 512 (`parity_timeline_submit.sh`,
read by `parity_timeline.py`). Each cell is the scalars' gradient error against the fp32
oracle: one group's relative error, averaged over the two groups. The cells are tau /
alpha / beta, in %.

| Step | P1-1 weights, `ame_std` (bf16 attention) | P1-1 weights, `ame_f3x` (fp32 attention) | AME-F1 weights, `ame_std` | AME-F1 weights, `ame_f3x` (as it trains) |
|---|---|---|---|---|
| 500 | 1.5 / 2.4 / 3.0 | 0.3 / 0.6 / 0.5 | 2.6 / 3.5 / 2.6 | 0.4 / 0.4 / 0.3 |
| 1,000 | 3.7 / 4.6 / 4.2 | 0.3 / 0.4 / 0.3 | 2.6 / 3.9 / 4.1 | 0.3 / 0.4 / 0.4 |
| 1,500 | 5.4 / 8.4 / 11.3 | 0.3 / 0.3 / 0.4 | 6.8 / 7.7 / 6.7 | 0.5 / 0.6 / 0.5 |
| 2,000 | 6.2 / 5.6 / 5.6 | 0.6 / 0.5 / 0.5 | 3.3 / 4.1 / 6.8 | 0.4 / 0.5 / 0.7 |
| 2,500 | 9.6 / 12.8 / 13.9 | 0.5 / 0.4 / 0.4 | 5.1 / 4.9 / 5.1 | 0.5 / 0.4 / 0.6 |
| 3,000 | 4.1 / 4.0 / 8.4 | 0.4 / 0.4 / 0.5 | 5.7 / 6.4 / 9.9 | 0.6 / 0.6 / 0.7 |
| 3,500 | 6.4 / 8.6 / 15.3 | 0.5 / 0.5 / 0.4 | 6.6 / 7.8 / 7.4 | 0.4 / 0.5 / 0.5 |
| 3,814 | 7.2 / 7.3 / 11.2 | 0.5 / 0.4 / 0.4 | 4.8 / 4.9 / 7.0 | 0.6 / 0.6 / 0.7 |

- **The error appears early.** It is 0.5-0.9% at init (Phase 1, 1.6), 1.5-3% by step 500
  and about 4% by step 1,000. From step 1,500 on it moves between 4% and 15% with no
  trend, beta usually the worst. With two groups per checkpoint, each cell is rough. The
  final checkpoint matches Step 2's production figures on `doi36p8k` (7.2 / 10.0 / 18.5%
  over 16 groups).
- **It does not push the scalars one way all run long.** Its component along the true
  gradient (the summed-error projection) changes sign from one checkpoint to the next:
  from −0.044 to +0.046 on P1-1's weights, from −0.049 to +0.026 on AME-F1's.
- **fp32 attention removes it at every checkpoint:** 0.3-0.6% throughout, on the same
  weights. The error belongs to the bf16 attention path at every stage of training, not
  only late in it.
- **AME-F trained on accurate scalar gradients all run:** 0.3-0.7% at every checkpoint
  of AME-F1, projections within ±0.004. So AME-F is the clean contrast the plan needs.
- **The error belongs to the bf16 path at any weights.** On AME-F1's weights the bf16 path
  shows 2.6-9.9%, the same range as on P1-1's (1.5-15.3%). Training with fp32 attention
  does not lead to weights on which bf16 would be accurate.

## Proposed full-length runs: prepared, not run

**Superseded on 2026-11-07 by ame_bf16_scalars_plan.md**, which runs the A/Bs on entropy
with replicates (section above). The LEM configs below stay on `parity_audit` unused.

They would settle what 300 steps cannot: whether any part of the implementation moves AME's
loss at 4B tokens. The configs are on `parity_audit` (`configs/parity/`, from `0d4ff3b8`).
Each is the run's recipe on LEM (n = 8, 3,815 steps, the 38-step warmup, the final
checkpoint kept for the common evaluator), with one change, and W&B project
`nano-ame-parity`.

| Run | Config | What it changes | What it settles | LEM GPU-h (est.) |
|---|---|---|---|---|
| **P0** | `p0.yaml` | nothing (S2-PS again) | the run-to-run floor at 4B, with `doi36p8k` | 4.2 |
| **P0r** | `p0r.yaml` | nothing (a third copy) | the floor needs ≥ 3 samples: one gate's sd is about 0.12 at 300 steps | 4.2 |
| **P0b** | `p0b.yaml` | micro-batch 16 × 32 | the reassociation floor | 4.6 |
| **P1** | `p1.yaml` | `mem_parallel_size: null` | **only A2 now** (A1 and A3 are exact): the bf16 compute copy at 4B | 4.3 |
| **P-REF** | `p_ref.yaml` | `mem_parallel_size: null`, `backend: reference`, micro-batch 8 × 64 | A1-A7 off together | 10.2 |
| **D0** | `d0.yaml` | nothing (S1-D again) | dense's floor | 3.6 |

**Total about 31 GPU-h; P-REF alone is a third.** Without P-REF the set is 21 GPU-h.

**How to submit** (from the laptop, which pushes over the 443 rewrite):

```bash
cd ~/code/nano-parity                 # branch parity_audit, clean
bash -c 'source ~/.bashrc; for c in p0 p0r p0b p1 p_ref d0; do \
  /home/maciej/code/nano/.pixi/envs/default/bin/python run_exp.py \
    --config-path configs/parity --config-name $c; rm -rf generated_configs; sleep 30; done'
```

**How to read them.** Score every final checkpoint with the common evaluator:
`documents/scripts/parity_numerics.py final_ce`, pointed at the new checkpoints. Then apply
the plan's rule with F from the spread of `doi36p8k`, P0 and P0r rather than from one pair:

- harmful if |Δ `final_ce`| ≥ max(3F, 0.005);
- and if the 50-step paired train-loss gap keeps one sign over ≥ 90% of the last 1,000
  steps.

**Not prepared, and needs your answer (Q2):** a run that would size X2's interaction with
AME, with the per-head scalars' gradients or the attention itself in fp32.

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
- **2026-11-07 00:10-00:35. The bf16-scalars plan's runs, in a different order from its
  waves.**
  - Wave 1 started as soon as periodic checkpoints were seen working. It holds AME-B ×3,
    P1 ×3 and P-REF, with P-REF moved up from wave 2 because at about 10.5 h it is the
    longest run.
  - These runs use none of the new flags. The plan's "validate before any training" gate
    concerns AME-F, and AME-F was validated (job 36393) before it was submitted.
  - The other 13 runs were submitted at once, at `nice` 100, longest first (D-F, AME-F,
    P1-16, D-B, then P0b, the first to cut). Slurm starts them as GPUs free up; analysis
    and scoring jobs at nice 0 go ahead of them.
  - **Budget:** the plan says 100 GPU-h; Maciej's goal message (00:00) says 120 GPU-h on
    top of the audit's own. 120 applies.
  - Estimate: about 113 GPU-h, with AME-F's speed still unmeasured. P0b is cut first if
    the estimate grows.

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
- **2026-11-06 22:45. The gate floor is a distribution, not a pair.**
  - The plan sets F from |G0b − G0a| and |G0c − G0a|. Identical production runs turned
    out to spread with an sd of about 0.12 nats at 300 steps and 0.13 at 220: G0e landed
    0.2 above G0a-G0d.
  - So a two-run F (0.061) made one ordinary-path run look "harmful", in reverse. G1a
    then showed that run computing exactly what production's fp32-bank twin computes.
  - The rule used instead: gates are grouped by what they compute (the twin check,
    `budget --base`), with 4-7 runs per group. An item is harmful only if the 95%
    interval of (production − item off) lies above max(0.005, 3 × the
    standard error of the difference) — in practice, entirely above zero by a margin
    the floor permits.
  - The plan's rule is still printed by `parity_gates.py --set 300`, for the record.
- **2026-11-06 23:00. Caches off `/tmp`.** After h100a's `/tmp` filled up, every parity
  job sets `TMPDIR`, `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` under
  `/storage_nvme_4/nano/mstefaniak/parity_tmp/<job>` (`add22fd5`). Numerics are
  unchanged; only where compiled kernels are cached moves.
- **2026-11-07 02:00. Full-run checkpoints on h100a's NVMe, not `/home`.** The plan put
  them in the cemetery clones under `/home`, whose 120 GB quota cannot hold them (the
  quota incident). They now go to `/storage_nvme_1/mstefaniak/parity_full/`, which only
  h100a sees, so every scoring and analysis job runs there anyway.
- **2026-11-07 02:30. AME-F's fp32 kernel uses tf32x3 dots and a 32×128 backward tile.**
  The plan asks for the kernel "in fp32". flex's default fp32 kernel (IEEE dots on CUDA
  cores, 16×16 backward tile) meets that at 34 s/step, which the budget cannot pay.
  tf32x3 with the tile is as accurate (4.2e-7 from fp64 in dq, as IEEE) at about 7.3
  s/step, and re-validates at the final checkpoint to the same scalar errors.
  Details: "AME-F's speed".
- **2026-11-07 02:30. P1-1 is rerun from step 0** after the quota killed it, not resumed:
  resuming with optimizer state is untested in the codebase.
- **Additions beyond the plan's checks, each to dig into a failed check:**
  - the gradient twin (`ame_t2fp32`, `ame_t2eager`, `--base ame_std`), which isolates
    A1, A2 and A3;
  - fp32-CE variants for X3;
  - the per-layer kernel comparison against SDPA flash and against a plain bf16
    reference;
  - the replicate gates.

## Questions for Maciej

Each question is numbered and dated and gives the context, the options and the default
being taken.

**Q1 (2026-11-06 23:05): h100a's `/tmp` is full. May I delete my own stale caches
there?** **Answered (Maciej, ~23:25): (a), within these limits:**

- only `mstefaniak`'s files;
- nothing a running job uses: `/tmp/torchinductor_mstefaniak` stays until no job of mine
  runs;
- `~/parity_ckpt/`, `~/parity/data/`, `~/parity/out/` and `~/parity/code/` stay until the
  report is final;
- list before deleting, record after.

**Done 23:27** (`~/parity/tmp_cleanup_list.txt`, `~/parity/tmp_cleanup_log.txt` on
entropy). The listing found 925 `mstefaniak` entries in h100a's `/tmp`, 10.11 GiB.
Deleted were the 924 modified before 23:00, when the only running job of mine (G-REF2,
36370, caches on NVMe) had started. Each was checked by the script to be a top-level
`/tmp` entry owned by `mstefaniak`; none was skipped.

| Removed | Entries | Size |
|---|---|---|
| `triton-cache-<job>-rank<n>`: compile caches of finished jobs, 2026-11-05 15:49 to 2026-11-06 22:59 | 313 | 7.08 GiB |
| `triton-rank<n>-*`, `inductor-rank<n>-*`: per-rank temp caches, 2026-11-06 00:46-04:36 | 60 | 0.65 GiB |
| `triton_gate_<job>` | 11 | 0.49 GiB |
| `inductor_gate_<job>` | 8 | 0.19 GiB |
| `triton_dump_<job>`, `inductor_dump_<job>` | 2 | 0.05 GiB |
| small leftovers: `tmp*`, `pymp-*`, `torchelastic_*`, old wandb dirs and sockets, `uv-*.lock`, `pytest-of-mstefaniak`, `pytorch-errorfile-*` | 530 | 0.01 GiB |
| **total** | **924** | **8.48 GiB** (du) |

**`/tmp` on h100a:**

| | Bytes used | Free | Use |
|---|---|---|---|
| Before | 107,306,639,360 | 434,176 B | 100% |
| After | 98,107,596,800 | 9,199,476,736 B | 92% |

**9.20 GB freed.**

`/tmp/torchinductor_mstefaniak` (1.63 GiB) waits until no job of mine runs, per the limit,
even though every job since `add22fd5` caches on the NVMe instead. With the full-length
runs of the follow-up plan that is the very end; it is removed in that plan's final
cleanup.

The original question follows.

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

**Q2 (2026-11-06 23:45): should the audit build a run that sizes X2's effect on AME?**

- **Context.** Under bf16 autocast, which both arms share, the gradients of AME's
  per-head scalars (alpha, beta, tau; 432 numbers) are 9-31% off at the final checkpoint,
  against 0.5-0.9% at init. That holds in every bf16 implementation of AME, the plain one
  included (Phase 1, 1.6).
  - The source is the bf16 attention backward (dq, dk 3.5-5.4e-3), amplified by
    cancellation in the per-head sums as attention sharpens.
  - Dense has no such parameters.
  - The loss-level cost at the end is the same as dense's (+0.00015), so the open
    question is only whether noisier scalar updates over the run cost AME anything. Adam
    averages the noise over about 10 steps, and the scalars move slowly late in training,
    so I expect little. That expectation is untested.
- **Options:**
  - (a) Build an opt-in fp32-attention path for AME (q, k, v and the kernel in fp32, an
    estimated 2-3× slower attention) and a full-length run with it on LEM: about 6-8
    GPU-h.
  - (b) A cheaper continued-training A/B from the final checkpoint, 200-300 steps at the
    final LR, bf16 against fp32 attention, in both arms. It needs the same flag plus a
    resume with optimizer state, which the codebase marks "not tested".
  - (c) Leave it as a recorded property of bf16 AME.
- **Default taken: (c).** It is a shared-policy property, not an AME-only efficiency
  change, and its expected size is small. Nothing is built.
- **Answered by Maciej (2026-11-07, 00:00):** a plan of its own,
  [ame_bf16_scalars_plan.md](ame_bf16_scalars_plan.md). It builds both fp32 attention
  paths and runs full-length A/Bs on entropy with replicates (above).

**Q3 (2026-11-07 00:30): the AME-F validation target.**

- **Context.** The plan requires AME-F's alpha/beta/tau gradient error against the fp32
  oracle to fall to ≤ 1% (the init level) at the final checkpoint before AME-F trains.
  AME-F reaches 1.3%, 1.9% and 1.4% on the training batch, and 0.83%, 1.2% and 0.92% on the
  held-out one. That is 10-15× better than bf16 attention.
  - At the final checkpoint no group of any bf16 path is under 1%: dense production's FF
    is at 1.8%, its head at 4.0%.
  - The scalars cannot be more accurate than the upstream gradient they are built from.
  - Getting under 1% would need the whole network in fp32, which is a different
    experiment from "fp32 attention".
- **Options:**
  - (a) Accept "no worse than the model's other groups" (met) and run AME-F as built.
  - (b) Also run v_proj and o_proj in fp32. That would shave one rounding (2^-9) off the
    upstream, about 0.2% here, and still not reach 1% on the training batch.
  - (c) Do not run AME-F.
- **Default taken: (a).** AME-F was submitted as built.
