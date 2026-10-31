# MFU ablation plan: six candidate fixes, one branch each, measured against baseline

2026-10-31. Follow-up to [mfu_gap_nanogpt_audit.md](mfu_gap_nanogpt_audit.md),
which ranked six gap sources behind our 27.7% h16 MFU vs tuned-stack ~40%+.
Maciej's directive: **implement each improvement in a separate tree off the
current code version, run a training with each, and compare to the baseline** —
so the result is a per-fix MFU delta that names the actual blocker, not a stack
where the first change takes credit for the rest.

**Status: PLAN — awaiting green light.**

## Method

- **Base commit**: the `pretraining` HEAD this plan lands on. Every variant
  branches from it: branch `mfu-abl/<name>`, built in its own worktree, carrying
  **exactly one change**. Branches are pushed (entropy pulls on the login node);
  worktrees are cleaned after, branches kept (repo policy).
- **The measurement** is identical for every variant: `configs/mfu_audit/h16.yaml`
  exactly as in stage 3 — one entropy H100, seq 2048, batch 512, 30 steps,
  micro×ga = 16×32, window steps 10–29, wandb project `nano-mfu-audit`, tag
  `mfu_ablation` + the variant name. Same seed as stage 3.
- **Baseline (B0)**: stage-3 h16 = 123,367 tok/s, 27.7% MFU
  ([sfupemo2](https://wandb.ai/ideas_cv/nano-mfu-audit/runs/sfupemo2)). One
  fresh re-run at the base commit first, to confirm node parity before any
  deltas are read against it.
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

## Results (filled as measured)

| run | tokens/s | MFU | Δ vs B0 | micro×ga | peak mem | wandb |
|---|---:|---:|---:|---:|---:|---|
| B0 re-run | | | — | 16×32 | | |
| V1 compile | | | | | | |
| V2 enable_gqa | | | | | | |
| V3 fused optim | | | | | | |
| V4 RoPE hygiene | | | | | | |
| V5 plain bf16 | | | | | | |
| V6 data path | | | | | | |
| V6c dummy-data B0 | | | | | | |
| nanoGPT ref (124M s1024) | | | n/a | | | |

## After the ablation: stacking

Interactions are real — V1 (compile) largely subsumes V4 and changes what V2/V3
are worth; V5 competes with FSDP2 rather than composing with it. So once the
per-fix deltas are in: branch `mfu-abl/stack` from base, apply the winners in
descending measured-delta order, re-measuring h16 after each addition (so the
stack is also an ordered attribution), finish with the full h12/h16/h20 matrix,
and update [backbone_mfu_audit.md](backbone_mfu_audit.md) with before/after.
What merges back into `pretraining` is the reviewed stack, not the ablation
branches.

Acceptance target from the gap audit: **h16 ≥ 35–40% MFU** (≈350–400 TFLOP/s).

## Parked for Maciej (changes deliberately NOT in the ablation)

- Fused QKV projection (changes parameter layout/init).
- No weight decay on 1-D params, weight tying (change training dynamics).
- Pre-tokenized memmap dataset (bigger build; V6c's dummy-data run measures its
  ceiling first — if the pipeline costs ~0 at h16, it stays parked until
  multi-GPU work needs it).
- Activation checkpointing (a memory knob, not an MFU fix at fixed micro-batch;
  it belongs to the stacking phase where bigger micros become reachable).
