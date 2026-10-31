# MFU ablation plan: six candidate fixes, one branch each, measured against baseline

2026-10-31. Follow-up to [mfu_gap_nanogpt_audit.md](mfu_gap_nanogpt_audit.md),
which ranked six gap sources behind our 27.7% h16 MFU vs tuned-stack ~40%+.
Maciej's directive: **implement each improvement in a separate tree off the
current code version, run a training with each, and compare to the baseline** —
so the result is a per-fix MFU delta that names the actual blocker, not a stack
where the first change takes credit for the rest.

**Status: EXECUTED 2026-10-31.** Verdict: **torch.compile was the blocker**
(+18% at every size once a guard bug was fixed); the data path, FSDP2 overhead,
optimizer and RoPE casts were all measured innocent. Final matrix with the
winners' stack (= V1 compile only): **h12 29.8%, h16 33.2%, h20 36.3%** vs
25.2/27.7/30.8% before. nanoGPT itself, compiled, measures **37.9%** on this
same H100 (124M, seq 1024), so the gap to it is now 1–5 points, not 10–17.

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
| nanoGPT reference (124M, seq 1024, compiled) | — | 37.9%² | — |

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

## Parked for Maciej (changes deliberately NOT in the ablation)

- Fused QKV projection (changes parameter layout/init).
- No weight decay on 1-D params, weight tying (change training dynamics).
- Pre-tokenized memmap dataset (bigger build; V6c's dummy-data run measures its
  ceiling first — if the pipeline costs ~0 at h16, it stays parked until
  multi-GPU work needs it).
- Activation checkpointing (a memory knob, not an MFU fix at fixed micro-batch;
  it belongs to the stacking phase where bigger micros become reachable).
