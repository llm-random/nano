# Speedup plan: from 44.4% to past nanoGPT, and what to adopt as default

2026-11-02. Follow-up to the nanoGPT comparison record in
[mfu_ablation_plan.md](mfu_ablation_plan.md) (sections 1–5 there hold every
measurement cited here). Standard workload for all numbers: h20, seq 2048,
batch 512, one entropy H100, bf16.

**Where we stand (measured):** eager 30.8% → per-block compile 36.0–36.4% →
whole-model compile **44.4%** MFU (9,451 ms/step, 110,890 tok/s), vs nanoGPT
**47.0%** (9,250 ms) on the identical workload. Our GEMMs and attention are
*faster* than nanoGPT's (−537 ms combined); the remaining +201 ms is eager
cross-entropy outside our compiled graph (230 ms), its bf16↔fp32 boundary
casts over the logits (~270 ms, inside the elementwise row), and
grad-accumulation adds running ~4× off roofline (291 vs their 76 ms).

## The plan, in order of value per effort

### 1. Adopt what is proven (zero new work, +44% tokens/s at h20)

Merge `mfu-abl/whole-compile` into `pretraining`. It carries, in one lineage:
V1 compile knobs, V3 fused AdamW + foreach clipping, V6 async H2D + prefetch,
V5 autocast knob, the Residual `log_name` refactor (logging moved into a
`@torch.compiler.disable` helper; per-block wandb names unchanged), and the
`+compile_model=true` whole-model compile knob.

- Single-GPU pretraining default: `trainer.distributed=null
  +trainer.autocast_bf16=true +compile_model=true`.
- **FSDP2 multi-GPU runs keep `+compile_blocks=true`** (regional compile is
  the pairing FSDP2 documents); whole-model-under-FSDP2 is untested — its own
  experiment when multi-GPU work resumes.
- Verified: knob-off bit-identical; compiled loss tracks eager to ~1e-7;
  zero recompile warnings; peak memory *drops* (66.9 → 60.5 GiB at h20
  micro 16).

### 2. Compile forward+loss as one graph (~+5%, expected to pass nanoGPT)

nanoGPT computes CE inside its compiled model; we compute it eager in the
trainer, so the 16×2048×50257 logits (≈6.6 GB fp32) cross the graph boundary
every micro-step and pay twice: eager log-softmax fwd+bwd (230 ms/step) plus
boundary casts (~270 ms/step). Fix: wrap the trainer's forward+CE computation
(the body of `_hack_for_python_garbage_collection` in
`src/core/trainer.py`) in `torch.compile`, so model and loss are one graph —
no model-API change.

- Watch for: graph breaks at the autocast context; the backward staying
  fused; eval path (which calls the same code) triggering one extra compile.
- Expected: −400–500 ms/step → ≈46.5–47% MFU.

### 3. Raise the micro-batch at h20 (one free run, ~+2–3%)

Whole-model compile dropped peak memory to 60.5 GiB at micro 16, so micro
32 × ga 16 plausibly fits. Halves every per-micro overhead at once —
grad-accumulation adds (291 ms), boundary casts, loop Python — and fattens
the GEMMs. One h20 run answers it; an OOM costs ten minutes.

**Landing zone for 2+3 combined: ~8,800 ms/step ≈ 47–48% MFU — ahead of
nanoGPT's 47.0%** on the same workload, with our GEMM+attention advantage
finally visible in the bottom line.

## Parked: the diminishing-returns shelf

- **Liger-style fused linear+CE** — never materializes the logits at all;
  beats nanoGPT's design and saves memory, but adds a kernel dependency.
  Revisit if step 2 disappoints or memory gets tight at bigger sizes.
- **Flat/bucketed grad accumulation** (DDP-style contiguous grad buffer) —
  worth ~150–200 ms; real plumbing.
- **Vocab padding 50257 → 50304** — measured minor once CE is compiled.
- **Activation checkpointing** — purely as a memory knob to buy larger
  micro-batches at h24+ sizes, not an MFU fix by itself.
- **Whole-model compile under FSDP2** — required before the multi-GPU
  configs can adopt item 1's default.

## Bookkeeping before this becomes the paper's default config

One longer overlay run (a few hundred steps, compiled vs eager, same seed)
confirming the ~1e-7 per-step drift does not grow into a visible loss-curve
divergence. Cheap, and it closes the numerics question for good.

## Execution record (2026-11-02): all three items done

1. **Merged** — `mfu-abl/whole-compile` → `pretraining` (`34bf3c17`); gates
   green, knobs opt-in, FSDP2 configs keep `compile_blocks`.
2. **Compiled loss** — branch `mfu-abl/compiled-loss` (`733f455`),
   `+trainer.compile_loss=true`: h20 step 9,451 → **9,046 ms**, 115,862
   tok/s, **46.4% MFU — 2.2% faster wall-clock than nanoGPT's 9,250 ms** at
   matched workload (job 32023; kernel table in
   [mfu_ablation_plan.md](mfu_ablation_plan.md) §4b, now four columns).
3. **Micro 32** — OOMs at h20 (73.6 GiB used + 6.1 GiB requested, job
   32024); 16×32 stays.

**Numerics overlay (jobs 32025/32031)**: 300 steps at h16, compiled
(`compile_loss`) vs eager twin, same seed, loss 11.2 → 5.17. Mean |Δloss|
0.0074, max 0.066 (step 148, the steep mid-run region), last-50 mean 0.0036
(0.07% of final CE), no divergence trend — benign bf16 fusion-order noise.
**Numerics question closed; the compiled single-GPU default is safe to
adopt.** Incidental datum from the overlay: compiled-loss at h16 runs
~5.4 s/step ≈ 194k tok/s ≈ 43.7% MFU (vs 33.6% parity).
