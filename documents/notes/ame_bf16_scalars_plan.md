# Plan: full-length A/Bs on entropy — bf16 per-head scalars (X2 × AME) and the parity runs

Extends the efficiency-parity audit (ame_efficiency_parity_plan.md; record in ame_efficiency_parity_report.md).
It replaces the report's LEM "Proposed full-length runs". Everything runs on entropy.

## Why
- **The audit found no evidence that AME's efficiency work costs quality.**
  - The gap is 0.1426 under an fp32 evaluator.
  - Each arm pays the same loss-level precision tax, +0.00015.
  - 220-300-step gates cannot see effects that need thousands of steps.
- **One open numerical finding (X2 × AME).**
  - At the final checkpoint, the bf16 gradients of alpha, beta and tau are 8.6-31% off the fp32 oracle,
    against 0.5-0.9% at init.
  - Every bf16 AME path shows it, the plain reference included.
  - The source is the bf16 attention backward (dq/dk error 3.5-5.4e-3), amplified by cancellation in the
    per-head sums.
  - beta's projection was negative on both batches measured (-0.11, -0.027): possibly a bias.
  - Both arms run attention in bf16; only AME has these scalars.
- **Short AME runs are chaotic** (sd ~0.12 nats at 300 steps), so every comparison below uses replicates.

## Questions (each ends with a verdict: negligible / harmful with size / inconclusive with bound)
- **Q-X2:** does fp32 attention improve AME's 4B-token loss more than it improves dense's?
- **Q-P1:** does the bf16 bank compute copy (A2, the only numerically active Tier 2 item) change AME's
  4B-token loss?
- **Q-REF:** does switching off all AME-only efficiency items (A1-A7) change it?

## Step 0: cleanup (Q1 answered: option a)
- Delete mstefaniak's stale files on entropy: h100a /tmp (triton-cache-*, triton_gate_*, old torchinductor
  caches) and anything else of mine no longer needed.
- Only mstefaniak's files.
- Nothing a running job uses. /tmp/torchinductor_mstefaniak waits until no job of mine runs.
- Keep ~/parity_ckpt, ~/parity/data, ~/parity/out and ~/parity/code until the report is final.
- First list what goes and its size. Afterwards, record it in the report with /tmp usage before and after,
  and mark Q1 answered.

## Step 1: build and test (branch parity_audit, ~/code/nano-parity)
- **parity_attention_fp32 (AME):**
  - v is upcast right after v_proj, so the beta mix runs in fp32;
  - q, k and v enter flex in fp32 (no `q.to(v.dtype)`);
  - the kernel and the sink merge run in fp32.
- **AME-F** = mem_parallel_size null (fp32 banks, so an fp32 prep) + parity_attention_fp32.
- **parity_dense_attention_fp32 (dense):** q, k and v are upcast to fp32 after RoPE, before SDPA.
- **Tests:**
  - every flag is opt-in, with a test proving the default bit-identical (like tests/test_parity_flags.py);
  - a CUDA test that each fp32 path matches the fp32 reference at a small shape.
- **Periodic checkpoints:** confirm they work, bank sidecar included (checkpoint.save.interval 500), on a
  20-step run.
- **parity_numerics.py:** `final_ce` and `budget` take checkpoint paths as arguments; they are hardcoded now.

## Step 2: probes and the cheap analysis (no training)
- **Probes** (40-60 steps, micro-batch 32): peak memory and s/step for AME-F and for dense fp32 attention.
  - Runs that are compared must share a micro-batch.
  - If AME-F needs micro-batch 16, run its control P1 at 16 as an extra set (P1-16 ×3).
- **Validate the mechanism before any training.** At the existing final checkpoint (~/parity_ckpt), AME-F's
  alpha/beta/tau gradient error against the fp32 oracle must fall to ≤ 1% (the init level).
  - If it doesn't, find what still rounds (for example o_proj or v_proj under autocast, or the merge) and fix
    the flag.
  - Otherwise the A/B would not test the hypothesis.
- **Stage A+B at the final checkpoint**, over the 16 held-out groups of 512 sequences:
  - per element of tau, alpha and beta: mean error over the RMS of the true gradient, the t-statistic, and the
    summed-error projection;
  - then 16 AdamW updates from the checkpoint's moments, with production gradients and with oracle gradients:
    the update error and sign agreement.
  - This says whether beta's bias is real, and whether Adam absorbs the noise.
- **Mechanism check:** recompute the scalar gradients in fp64 from flex's bf16 dq/dk/dv and from the
  oracle's fp32 dq/dk/dv.

## Step 3: full-length runs on entropy
- **Recipe:** S2-PS or S1-D at n = 8: 3,815 steps × 1,048,576 tokens, warmup 38, cosine to 10%, seed 123,
  autocast bf16, compile_blocks_full.
- **Checkpoints:**
  - every 500 steps for AME-B1, P1-1 and AME-F1, for the error timeline;
  - final only for every other run;
  - all of them land in the cemetery clone under /home. AME is 13.2 GB per checkpoint, dense 2.2 GB.

| Run | Change | n | Entropy h each (est.) |
|---|---|---|---|
| AME-B (production) | none | 3 | 3.8 |
| P1 | mem_parallel_size null (A2 off) | 3 | 3.9 |
| AME-F | P1 + parity_attention_fp32 | 3 | 5-7 (from the probe) |
| D-B (dense production) | none | 3 | 3.3 |
| D-F | parity_dense_attention_fp32 | 3 | 4-5 (from the probe) |
| P-REF | mp null + backend reference, micro-batch 8 × 64 | 1 | 10.2 |
| P0b | AME-B at micro-batch 16 × 32 | 1 | 3.9 |

- **Cost:** about 82 GPU-h, about 94 if P1-16 is needed. The cap for everything is 100 GPU-h. Cut order
  if needed: P0b first, then the third D-F, then the third D-B.
- **Waves of at most 8 GPUs:**
  - wave 1: AME-B ×3, P1 ×3, D-B ×2;
  - wave 2: AME-F ×3, D-F ×3, D-B, P-REF;
  - then P0b and any P1-16.
- Preemption is allowed.
- **Compare only runs made on entropy.** doi36p8k and x0xlqroj (LEM) are not samples.

## Step 4: decide
- **Scoring:** every final checkpoint with the common evaluator (fp32 reference path, precise CE, full
  held-out set) and with the production evaluator.
- **Q-X2:**
  - I = [CE(P1) − CE(AME-F)] − [CE(D-B) − CE(D-F)], with a 95% interval from the groups' spreads;
  - harmful if the interval lies above 0 and I ≥ 0.005; negligible if the whole interval lies inside
    ±0.005; otherwise inconclusive, with the bound;
  - supporting evidence: alpha/beta/tau trajectories, AME-F against P1 compared with P1 against P1'; and
    Stage A at each 500-step checkpoint of P1-1 and AME-F1, which shows when the error appears and whether
    fp32 removes it all run long.
- **Q-P1 (P1 against AME-B) and Q-REF (P-REF against AME-B):**
  - harmful if production is worse than the item-off run by more than max(3 × the AME-B spread, 0.005);
  - and the 50-step paired train-loss gap keeps that sign over ≥ 90% of the last 1,000 steps.
- **If anything is harmful:** its size at 4B, the fix candidates, and their throughput price (a 60-step probe).

## Rules
- **Entropy only:**
  - submit as mstefaniak: run_exp.py from the clean worktree for trainings, parity_submit.sh for numerics;
  - caches on NVMe, never /tmp;
  - LEM stays read-only.
- **Code:** no default changes; new code only as opt-in flags with tests proving the default bit-identical.
- **Budget:** 100 GPU-h in total. Past it, add a question for Maciej and submit nothing more.
- **Working alone:** decisions for Maciej go to "Questions for Maciej" with the options and a default.
- **Commits:** commit the report after every step, and at least every 3 hours.

## Output
- In ame_efficiency_parity_report.md:
  - "Full-length runs on entropy": runs, floors, and the Q-P1 and Q-REF verdicts;
  - "X2 × AME: the per-head scalars under bf16": Step 2's tables and the Q-X2 verdict with its size or bound;
  - the verdict table and "The answer" updated.
- A finding in ame_comparison_h12.md.
- Everything committed; no job of yours queued or running.
