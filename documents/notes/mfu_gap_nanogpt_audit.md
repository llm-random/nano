# Why nanoGPT's MFU headline beats ours, and the plan to close the real gap

2026-10-31. Prompted by: nanoGPT reports 37–45% MFU (124M GPT-2, A100, bf16,
torch.compile, flash SDPA). Our stage-3 audit
([backbone_mfu_audit.md](backbone_mfu_audit.md)) measured 25.2/27.7/30.8% at
h12/h16/h20 on an H100. This note audits karpathy/nanoGPT (model.py + train.py,
cloned at HEAD) against our stack and decomposes the difference.

## First, the denominator: we are not actually 1.5× slower

MFU divides by the GPU's peak, and the peaks differ by 3.17×:

| | achieved TFLOP/s | basis |
|---|---:|---|
| nanoGPT 124M, A100, 37–45% | **115–140** | 312e12 |
| ours h12 (183M), H100, 25.2% | **249** | 989e12 |
| ours h16 (355M), H100, 27.7% | **274** | 989e12 |
| ours h20 (620M), H100, 30.8% | **305** | 989e12 |

Per GPU we push ~2× nanoGPT's cited absolute throughput. Both use the same PaLM
FLOPs convention (6N + 12·L·H·Q·T, no causal discount), so the numbers are
comparable. H100s are also notoriously harder to fill than A100s (same code
scores lower MFU% on the bigger peak), and nanoGPT measures at seq 1024 where
the (lower-efficiency) attention share of wall-clock is half of ours at 2048.
So "outperformed by simple nanoGPT" is mostly the denominator — **but** tuned
H100 implementations at these sizes do reach ~40%+, so roughly 10–15 points of
our gap are real and recoverable. The decomposition:

## Gap sources, ranked (ours vs nanoGPT, audited line by line)

1. **No torch.compile anywhere** (`grep torch.compile` over main.py/trainer:
   zero hits). nanoGPT's headline number is *with* compile; its README credits
   compile with cutting iteration time ~1.8× on early PT2. Everything below
   that is eager kernel-chain overhead (RoPE, gate, norms, CE) is exactly what
   inductor fuses. Expected +25–50% throughput. **The single biggest item.**
2. **Eager RoPE per layer** ([model.py](../../src/core/model.py) `RoPE.forward`):
   chunk/cat/neg/mul/mul/add plus a per-forward `.to(device, dtype)` of cos/sin
   — ~8 extra memory-bound kernel passes over Q and K per layer per
   micro-forward. nanoGPT's positional cost is one embedding add for the whole
   model. Compile absorbs most of this; buffer-dtype hygiene removes the `.to`.
3. **`repeat_kv` materialization before SDPA**
   ([model.py:270](../../src/core/model.py)): K/V are physically expanded to
   full query width, costing bandwidth and activation memory (directly lowers
   the OOM-probe micro-batch ceiling). Torch 2.7.1 has
   `F.scaled_dot_product_attention(..., enable_gqa=True)` — flash handles GQA
   natively. nanoGPT is MHA so it never pays this.
4. **FSDP2 wrapped at world size 1**: every TransformerBlock + embedding + head
   is `fully_shard`-ed even single-GPU, so each micro-forward pays per-module
   hooks, DTensor dispatch, and the fp32→bf16 parameter recast — at ga=32–64
   the whole model is recast dozens of times per step, with no communication to
   hide behind. nanoGPT runs a raw module under `torch.amp.autocast`. Our
   dtype matrix has a hole here: we measured FSDP2-bf16, plain-fp32, plain-tf32
   — **never plain + autocast-bf16**, which is the exact nanoGPT configuration.
5. **Optimizer and clipping, non-fused** ([main.py:211](../../main.py)): plain
   `torch.optim.AdamW` (nanoGPT: `fused=True`), and our FSDP2 grad-clip path
   ([trainer.py:380](../../src/core/trainer.py)) launches one `.norm()` kernel
   per parameter (~150 serial launches/step) where `clip_grad_norm_`'s foreach
   path uses a handful of multi-tensor kernels. Also: we apply weight decay to
   every parameter including norms; nanoGPT decays only ≥2-D params (a training
   -dynamics difference, not a speed one — parked separately).
6. **Data path**: on-the-fly NeoX tokenization in `num_workers: 2`, int64
   collate, and a synchronous `.to(device)` per micro-chunk (pinned memory but
   no `non_blocking=True`, no prefetch overlap) — vs nanoGPT's pre-tokenized
   uint16 memmap with async prefetch issued *during* the forward. The 30-step
   audit sustained it (h4 pulled 1.24M tok/s), but every micro-step's H2D copy
   sits on the critical path, and 2 workers will not feed a multi-GPU run.
7. **Not inefficiencies, but MFU% depressors to remember when comparing**:
   seq 2048 doubles the attention share (flash runs below dense-GEMM
   efficiency); GQA cuts counted FLOPs ~proportionally more than wall-clock
   (measured in E7); gated FF = 3 smaller GEMMs vs MLP's 2 bigger; separate
   q/k/v projections vs nanoGPT's single fused `c_attn` GEMM.

Exonerated by this audit: the per-layer Residual norm instrumentation (gated to
every 100 steps — fires at step 0 only in a 30-step run, outside the window),
and `F.cross_entropy` usage (same shape as nanoGPT's).

## The plan

### Phase A — equal-footing measurements (cheap: short single-GPU entropy jobs)

| # | Run | Question it answers |
|---|---|---|
| A1 | h16, **plain module + autocast(bf16)**, no FSDP2 (`trainer.distributed=null` + autocast wrapper) | FSDP2's single-GPU overhead in isolation — fills the dtype-matrix hole |
| A2 | h16, FSDP2-bf16 + **torch.compile** (and A1 + compile) | compile's win on each path, before committing code |
| A3 | **nanoGPT itself on our H100** (124M, seq 1024, their defaults, random-token get_batch so no openwebtext prep) | the true reference number on our hardware — kills the A100 apples/oranges for good |

### Phase B — changes, each committed separately and measured (ranked by expected win)

1. `trainer.compile` knob (default off), compile the model: expect +25–50%.
2. `enable_gqa=True` SDPA path replacing `repeat_kv` (torch ≥2.5 available;
   equivalence-tested against repeat_kv at tolerance, fallback kept).
3. Fused AdamW (`fused=True` — numerics-identical, one line) and foreach
   gradient clipping (replace the per-param loop).
4. RoPE hygiene: cos/sin buffers registered in model dtype on device (kill the
   per-forward `.to`); the rest left to compile.
5. Single-GPU path choice: world_size==1 runs plain+autocast when a config asks
   (FSDP2 buys nothing on one GPU); FSDP2 stays the multi-GPU default.
6. **Activation checkpointing knob** (already parked in the stage-3 audit):
   memory lever → bigger micro-batches → MFU at h16+.
7. Pre-tokenized memmap dataset option (nanoGPT prepare.py style) +
   `non_blocking=True` H2D with prefetch of the next chunk during the current
   forward — removes tokenization from the critical path before multi-GPU work.
8. Optional/parked for Maciej (they change params or dynamics): fused QKV
   projection; no weight decay on 1-D params; weight tying.

### Phase C — acceptance

Re-run the stage-3 matrix (h12/h16/h20, bf16) with the winners enabled.
Target: **h16 ≥ 35–40% MFU on H100** (≈ 350–400 TFLOP/s), i.e. nanoGPT-class
utilization against the H100 peak at seq 2048. Update
[backbone_mfu_audit.md](backbone_mfu_audit.md) with a before/after table.

Expected stacking, honestly: compile +25–50%, native-GQA SDPA +2–5%, fused
optim/clip +1–3%, plain-vs-FSDP2 on 1 GPU +3–8% (A1 measures it), AC-enabled
bigger micros a few more at h16+. From 27.7% that lands plausibly at 38–45%.
