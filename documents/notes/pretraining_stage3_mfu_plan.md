# Stage 3: backbone audit — does it all work, and how fast (MFU)

Redefined by Maciej 2026-10-30: before any new attention implementations, validate the
current backbone end to end with **real single-GPU trainings on entropy** and measure
implementation efficiency as **MFU**. Models **h12, h16, h20**; sequence length
**2048**; **512 sequences per batch** (= 1,048,576 tokens/step); few-step runs; a
**new wandb project**. The point is to exercise everything that has not run on a GPU
since the reset and to catch bugs precisely, not to train anything real.

**Status: PLAN — awaiting green light.**

## Ground rules

- Single-GPU jobs only, on entropy, **one job at a time** (shared quota; it usually
  has one free GPU, not several).
- Direct commits on `pretraining`; pytest (CI filter) + black green after every
  commit; the local CPU smoke configs stay runnable throughout.
- MFU instrumentation must be a **numerics no-op**: seeded local runs bit-identical
  before/after the logging lands. The one deliberate numerics change is bf16
  (Phase 1.3), behind a config knob that defaults off.
- No pixi changes. New wandb project proposed: `ideas_cv/nano-mfu-audit`
  (override of `infrastructure.metric_logger.project_name`) — **parked for naming**.

## Definitions, fixed up front (so every number means the same thing)

- **Tokens/step** = 512 × 2048 = 1,048,576. `processed_tokens` in wandb must equal
  exactly `step × 1,048,576` — if it does not, the accounting is buggy (test E-c3).
- **FLOPs/token (training)** = `6·P + 12·L·s·d` (PaLM convention: fwd 2P + 4Lsd,
  backward 2× fwd; causal masking NOT discounted), where
  `P = 15·L·d² + V·d` (all matmul params: attention 3d²/layer under GQA-H/2, gated
  FF 12d²/layer, untied head; embedding lookup excluded), `s = 2048`, `V = 50304`.
  Worked example, h16 (d=1024, L=16): P = 303.2M → **2.22 GFLOPs/token**.
- **MFU** = (tokens/s × FLOPs/token) / peak-FLOPs(GPU, dtype). Peak table:
  A100 bf16 312e12, H100 SXM bf16 989e12, H100 PCIe bf16 756e12; fp32 peaks are the
  same tensor-core numbers at tf32 (A100 156e12, H100 SXM 495e12). Unknown GPU name
  → log tokens/s and a warning, mfu stays null rather than wrong.
- **Timing window**: steps 10–29 of a 30-step run (step time from the existing step
  timer); step 0–9 excluded (allocator warmup, dataloader spin-up, cache effects).

## Phase 0 — preflight (local + entropy, no GPU experiments yet)

| # | Test | Pass criterion |
|---|---|---|
| P0.1 | Add `h12.yaml`, `h20.yaml` (3-line instance files); extend `test_h_balanced` parametrize to 12 and 20 | exact closed-form param counts: h12 = 183.4M, h20 = 620.2M class (formula incl. norms) |
| P0.2 | New runnable configs `configs/mfu_audit/{h12,h16,h20}.yaml`: `_cluster: entropy`, `_dataset: fineweb`, `_model: h_balanced/hNN`, seq 2048, batch 512, n_steps 30, cosine warmup 10, lr 2^-12, eval off for the MFU runs, save off (no checkpoint I/O inside the timing window) | hygiene suite composes + resolves them (automatic) |
| P0.3 | Stage the NeoX tokenizer on entropy (scp into the HF cache, same route as models--gpt2 before) and prove it loads with `HF_HUB_OFFLINE=1` on a compute node | tokenizer loads offline; ids ≤ 50303 |
| P0.4 | Run the full pytest suite on entropy (one CPU/GPU job) — first suite run there since the stage-1/2 rewrite | suite green on the cluster env |
| P0.5 | Verify fineweb paths at seq 2048: compose the h12 config on entropy, pull 3 batches in a tiny CPU-side probe | batches shaped (…, 2048), max id ≤ 50303, holdout train/eval disjointness test already covers the rest |

## Phase 1 — instrumentation (local, gated bit-identical)

1. **MFU metric** in the trainer: compute FLOPs/token once at init from the live
   model (via `get_model_dimensions()`, so it is the *measured* architecture, not
   the config's claim), detect GPU name + peak from the table, and log per step:
   `train/step_time`, `train/tokens_per_s`, `train/mfu`, plus once per run:
   GPU name, dtype, `torch.cuda.max_memory_allocated` (logged at end),
   world_size (already logged). Unit tests: the FLOPs closed form against a hand
   count for h4; the MFU arithmetic with a stubbed timer and a fake device name;
   the unknown-GPU fallback (mfu null, loud warning, tokens/s still logged).
2. **Loss sanity hook** (test-side, not trainer): first logged train/loss of a
   fresh run must be ln(50304) = 10.826 ± 0.15 — wrong init, a vocab mismatch or a
   head-wiring bug all show up here before a GPU-hour is spent.
3. **bf16 knob**: audit what dtype the single-GPU path actually matmuls in today
   (suspicion: fp32 — the bf16 policy lives in the FSDP path). Add
   `trainer.autocast_bf16: false` default; when true, wrap forward+loss in
   `torch.autocast("cuda", bfloat16)` (optimizer/grads stay fp32). Tests: knob off
   → seeded CPU run bit-identical to today; knob on → parameters still fp32, loss
   finite. The MFU runs then measure **both** settings once (E3 decides which is
   the project default — parked decision).

Gate: pytest + black + bit-identical seeded tiny_local & smoketest with the knob off.

## Phase 2 — entropy experiments (single GPU each, sequential)

Per model, two jobs:

- **(a) micro-batch probe**: 5-step run that doubles the per-forward micro-batch
  (gradient_accumulation halves: start ga=64 ⇒ micro 8·2048 tokens) until OOM, then
  backs off one notch. Records the chosen `(micro_batch, ga)` and peak memory. MFU
  improves with bigger micro-batches, so measuring at an arbitrary ga would
  understate the implementation.
- **(b) measurement run**: 30 steps at the probe's setting, batch 512×2048, wandb
  project `nano-mfu-audit`, run name `hNN_s2048_b512_<dtype>`.

| # | Run | What it proves / pass criteria |
|---|---|---|
| E1 | h4 pipeline smoke (30 steps, same batch spec) | the whole path end to end before burning time on big models: submission, offline NeoX, wandb logging to the new project, rc=0. Criteria: loss stays finite, **first loss 10.826 ± 0.15**, loss at step 29 < loss at step 0, `processed_tokens = step·1,048,576` exactly, step-time std in the window < 10% of mean |
| E2 | h12 (a) probe + (b) measurement, fp32/tf32 **and** bf16 | all E1 criteria + MFU recorded for both dtypes |
| E3 | h16 same | same; this is the reference model — if bf16 MFU < **25%** (A100) / **20%** (H100), open an investigation before proceeding (prime suspects: dataloader/tokenizer stall — NeoX must tokenize ~1M tokens/step on CPU workers; fp32 leakage; unintended host-device syncs). Investigation tool: 5-step rerun under `torch.profiler`, plus one rerun with the local pre-tokenized dummy data to isolate input pipeline from compute |
| E4 | h20 same | same criteria; also confirms memory headroom scaling (bigger model, smaller micro-batch) |
| E5 | h16 + eval path: eval_interval=10, final_eval over the fineweb holdout at seq 2048 (eval was off in E2–E4) | eval/loss logged at steps 10,20; `final_ce over <N> samples` with N = the configured holdout cap; run rc=0 — first GPU exercise of the stage-1 final_eval rewrite at real seq length |
| E6 | h16 resume: 30-step run saving at step 15, then a second job resuming from it | resumed loss curve continues the original within noise (no re-start spike, no jump > 0.5); the GPU twin of the CPU resume test |
| E7 | registry on GPU (h16, 30 steps each): `mha`, `mqa`, and `mix: {type: mqa, layers: every_2nd}` | all train; MFU per variant recorded (mqa should be ≥ gqa ≥ mha in tokens/s — if not, repeat_kv or kv-width wiring is suspect); the mixed run proves per-layer switching under a real optimizer step |

Every job: `gres=gpu:1`, submitted via `run_exp.py`, one at a time, ~5–10 min each
(h16 ≈ 7 s/step at 30% MFU on H100 → ~4 min + queue). Expected total: ~15 jobs.

## Phase 3 — analysis and close

1. Results note `documents/notes/backbone_mfu_audit.md`: the MFU table
   (model × dtype × GPU × micro-batch × tokens/s × MFU × peak mem), wandb run
   links, every bug found and its fix commit.
2. Any bug found mid-phase is fixed forward in its own commit (test first where
   reproducible locally), then the affected experiment re-runs.
3. Log updates throughout; final push; summary with the parked list.

## Parked for Maciej (so far)

- wandb project name (`nano-mfu-audit` proposed) and entity (`ideas_cv` assumed).
- Whether bf16 autocast becomes the single-GPU default after E2–E4 report both.
- MFU acceptance bar beyond the E3 investigation threshold (first measurement sets
  the baseline; thresholds above are tripwires, not targets).
- h24 excluded per the stated model list (h12/h16/h20) — say the word to add it.

## Risks / honest unknowns

- Entropy compute nodes have no hub access: anything not staged (tokenizer) fails
  only at runtime — hence P0.3 before any experiment.
- The current single-GPU dtype story is unaudited (Phase 1.3 exists because of it);
  fp32 MFU numbers will look halved against bf16 peaks — the table keys on dtype.
- Dataloader tokenization throughput at 1M tokens/step is the most likely MFU
  ceiling and is explicitly isolatable (E3's dummy-data rerun).
- Queue time on the shared quota is the schedule risk; jobs are short and sequential.

## Amendments during execution

- **2026-10-30 (token accounting)**: `processed_tokens` counts tokens the model
  consumes — `batch × (width − 1)` = 512 × 2047 = **1,048,064** per step — not the
  raw 512 × 2048 the plan stated. The criterion becomes
  `processed_tokens = step × 1,048,064`, exact. MFU is unaffected: tokens/s and
  FLOPs/token both use the consumed-token basis (seq 2047), so the ratio is
  consistent. Whether the *spec* should count raw or consumed tokens is parked
  for Maciej; the audit reports both.
- **2026-10-30 (submission)**: direct sbatch from the pinned ~/nano-mfu clone
  instead of run_exp.py (entropy :22 block; compute nodes cannot reach GitHub at
  all — pulls happen on the login node). Provenance: the job logs the clone's
  commit; wandb logs the config.
- **2026-10-30 (dtype)**: no autocast knob — FSDP2's hard-coded
  MixedPrecisionPolicy (bf16 compute / fp32 reduce) is the production path and is
  what the audit measures. fp32 comparison runs dropped.
