# Task: PC vs HP training-time overhead vs batch size (paper Fig. 2)

Started 2026-10-06. Session "pc_time".

## Goal
Empirically back Fig. 2 ("Amortized relative computational overhead of PC across increasing batch
sizes"): PC's extra work (materialize W_c, projection backward) is a fixed cost per optimizer step,
independent of batch size b, so its relative overhead vs HP should shrink as b grows.
Plan (Maciej): Llama 1B, HP vs PC, on 1 H100, batch 100k → 2M tokens (rounded to seq-len multiples),
grad_acc = 1, ~10 steps per point for timing. First find max batch that fits on 1 GPU for PC 50%,
then 70% and 90%.

## Code facts (verified)
- No activation checkpointing anywhere in src/ → all activations for the full batch must fit.
- PC overhead runs ONCE PER OPTIMIZER STEP, outside the grad-accum loop:
  `prepare_compressed_weights()` at start of each step (projected_compression/trainer.py:59),
  `pass_gradient_to_projections()` after accumulation (:72). So grad accumulation still measures
  amortization correctly (overhead per step / tokens per step).
- Step time is only logged as a 100-step average (`steps/100/time`) → for 10-step probes use wandb
  per-row `_timestamp` deltas.
- Vocab 128,256 → logits dominate activation memory: ~4 GB bf16 per 16k tokens just for logits,
  more with fp32 loss/grad copies. This caps tokens/GPU regardless of compression ratio.

## Probe setup
Configs `configs/pc_project_mas/llama_1B_pc_bs_probe{,_70p,_90p}_entropy.yaml`: plain PC (no distill),
1×H100 (`gpu:h100:1`), grad_acc 1, n_steps 10, warmup 2, no checkpoint, wandb `ideas_cv/pc_time`.
Grid over `common.batch_size` (sequences of 1024). NB run_exp resolves interpolations before grid
expansion → need a plain `batch_size` placeholder next to `^batch_size` (caret wins).

## Results
### PC 50% (job 23350) — DOES NOT FIT on 1 GPU at any batch
- batch 16 seqs (16k tokens) OOM in forward (calculate_loss): 73.6 GiB already allocated, tried +3.9 GiB (logits).
- Static memory ≈ 70 GB: ~3B trainable projection+W_b params (ff P ≈ 8192×5376 per matrix) in fp32 ×
  (weight+grad+2 Adam) ≈ 48 GB, + frozen 1B source + compressed model. On 4 GPUs FSDP2 shards this.
- Larger tasks cancelled (would OOM too).

### PC 90% (job 23357), PC 70% (job 23361) — 1×H100, grad_acc 1, 10 steps (wandb ideas_cv/pc_time)
Max batch that fits: **90% → 32 seqs (32,768 tok)**, OOM at 48. **70% → 24 seqs (24,576 tok)**, OOM at 32.

| ratio | seqs | tokens | s/step (median) | tok/s |
|---|---|---|---|---|
| 90% | 4 | 4,096 | 0.373 | 10,991 |
| 90% | 8 | 8,192 | 0.382 | 21,437 |
| 90% | 16 | 16,384 | 0.437 | 37,531 |
| 90% | 24 | 24,576 | 0.479 | 51,280 |
| 90% | 32 | 32,768 | 0.544 | 60,198 |
| 70% | 4 | 4,096 | 0.690 | 5,938 |
| 70% | 8 | 8,192 | 0.713 | 11,497 |
| 70% | 16 | 16,384 | 0.776 | 21,126 |
| 70% | 24 | 24,576 | 0.951 | 25,840 |

- s/step = median of wandb `_timestamp` deltas after dropping 2 warm-up steps (~9 deltas) — rough.
- Peak GPU mem from wandb system metrics is sampled ~15 s → under-sampled, not reliable (e.g. 70% bs24 shows 51 GiB < bs16 65 GiB).
- Linear fit (rough): 90% ≈ 0.35 s fixed/step + 6.1 ms/seq; 70% ≈ 0.64 s fixed/step + 13 ms/seq. The fixed
  part grows with projection size (70% has larger P than 90%) — consistent with a per-step PC overhead,
  but it also contains optimizer step / launch costs; needs an HP baseline to attribute.
- Illustrative only (extrapolation): with grad accumulation at 32-seq microbatches, 2M tok ≈ 61 microbatches
  × ~0.195 s ≈ 12 s compute + ~0.35 s fixed → fixed ≈ 3% at 90%.

## Conclusions
- grad_acc=1 on 1 GPU caps batch at ~24–32k tokens (70/90%) and PC 50% does not fit at all → the planned
  100k–2M sweep is impossible as specified. Logits (128k vocab) + no activation checkpointing are the limit.
- Options: (1) grad accumulation with a fixed microbatch (valid: PC overhead is once per optimizer step);
  (2) multi-GPU FSDP (makes 50% fit, but adds comm to timings; still needs accum for 2M);
  (3) cpu_offload_projections (fits 50% on 1 GPU but timing dominated by PCIe — not representative).
- Awaiting Maciej's decision.

## n_blocks = 1 timing model (agreed with Maciej 2026-10-06)
Idea: T(n, b) ≈ F0 + F1·n + (c0 + c1·n)·b — measure n=1 and n=2 blocks at feasible batch sizes, fit,
extrapolate to n=16 and b=100k..2M. Head/embedding/loss are NOT negligible (≈26/30/35% of fwd FLOPs at
50/70/90% due to the 128k vocab) → captured by the n-independent terms, not ignored.

Truncated source checkpoints (no code change needed; get_sharded_sd iterates source keys, so a 16-block
checkpoint crashes a 1-block model): `/storage_nvme_4/nano/j321m/pc/trunc/llama1b_{1,2}blk_in_nano.pt`
(first n blocks + embedding + head) and `importances_{1,2}blk.pt` (dff_importances[:n]). Projections take
n_blocks from len(target_model.encoder.blocks) → consistent.

Configs `configs/pc_project_mas/llama_1B_pc_1blk_bs_probe_{50,70,90}p_entropy.yaml` (common.n_blocks: 1).
Jobs 23381 (50%), 23385 (70%), 23386 (90%). 1×H100, grad_acc 1, 10 steps.

**Max batch with n_blocks=1: 64 seqs = 65,536 tokens for ALL three ratios; OOM at 80 seqs (81,920).**
Same limit for every ratio → the 128k-vocab logits/loss are the binding constraint (~1 MB/token), not width.
(Caveat learned: wandb prints "View run" even on crash and the OOM trace can be flushed later — judge
success by wandb state=finished + step count, not by grepping the slurm log mid-run.)

| ratio | seqs | tokens | s/step | tok/s |
|---|---|---|---|---|
| 50% | 32 | 32,768 | 0.220 | 149,139 |
| 50% | 48 | 49,152 | 0.267 | 183,843 |
| 50% | 64 | 65,536 | 0.329 | 199,066 |
| 70% | 32 | 32,768 | 0.163 | 201,303 |
| 70% | 48 | 49,152 | 0.207 | 237,146 |
| 70% | 64 | 65,536 | 0.246 | 266,314 |
| 90% | 32 | 32,768 | 0.105 | 311,055 |
| 90% | 48 | 49,152 | 0.157 | 314,048 |
| 90% | 64 | 65,536 | 0.182 | 359,660 |

Next: n_blocks=2 PC at {32,48,64}; same for HP (n_blocks 1/2) to get F_HP; then fit + extrapolate.

## Single-block synthetic benchmark — no embedding / no head (2026-10-06, Maciej's idea)
Script `benchmarks/bench_block.py` (repo classes, same dims as the probe configs; random weights/data):
- PC step: materialize W_c=P_out·W·P_in+W_b (fp32, no_grad) → block fwd/bwd under bf16 autocast (≈ FSDP2
  param_dtype=bf16 used by PC training) → projection backward (fp32, = backward_compressed_weights) →
  clip_grad_norm(proj, 0.25) + AdamW(proj); clip(norms, 1.0) + AdamW(norms). W_c not stepped (as in code).
- HP step: block fwd/bwd (bf16 autocast) → clip 1.0 → AdamW(block).
- Input x: random bf16 (b, 1024, d_c), requires_grad=True (middle layer also computes dL/dx). Dummy loss = out.mean().
- Exact peak memory via torch.cuda.max_memory_allocated; median of timed steps; ascending grid, stop at first OOM;
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. Run with PYTHONPATH=repo root.
- Validation (50%, 3 steps): PC 8→32 seqs 0.053→0.067 s; HP 0.007→0.022 s → same per-seq slope (~0.6 ms/seq),
  PC fixed ≈ 0.045 s/step/block. Activations ≈ 108 KB/token (50%).
- Full sweep (1×H100): 50/70/90 × pc/hp over seqs 98,196,391,586,782,977,1172,1368,1563,1758,1954,2442,2930,3907
  (= 100k..4M tokens rounded up to 1024 multiples). Log: entropy ~/bench_sweep1.log

### Full sweep results (srun job 23422, 1×H100, 1 block, 10 timed steps)
**Max batch (same for PC and HP): 50% → 586 seqs = 600k tok (OOM 800k); 70% → 782 seqs = 800k tok (OOM 1M); 90% → 1563 seqs = 1.6M tok (OOM 1.8M).** 2M fits for none at grad_acc=1.
PC peak memory ≈ HP + ~2 GiB (projections + AdamW state); e.g. 50%@600k: PC 65.8 vs HP 63.8 GiB.

| ratio | tokens | HP s/step | PC s/step | PC−HP (s) | PC overhead |
|---|---|---|---|---|---|
| 50% | 100,352 | 0.0634 | 0.1098 | +0.0464 | 73.1% |
| 50% | 200,704 | 0.1261 | 0.1723 | +0.0462 | 36.6% |
| 50% | 400,384 | 0.2496 | 0.2951 | +0.0454 | 18.2% |
| 50% | 600,064 | 0.3747 | 0.4169 | +0.0422 | 11.3% |
| 70% | 100,352 | 0.0464 | 0.0757 | +0.0293 | 63.2% |
| 70% | 200,704 | 0.0918 | 0.1208 | +0.0290 | 31.6% |
| 70% | 400,384 | 0.1829 | 0.2106 | +0.0277 | 15.1% |
| 70% | 600,064 | 0.2743 | 0.3021 | +0.0277 | 10.1% |
| 70% | 800,768 | 0.3674 | 0.3956 | +0.0282 | 7.7% |
| 90% | 100,352 | 0.0312 | 0.0424 | +0.0111 | 35.7% |
| 90% | 200,704 | 0.0612 | 0.0724 | +0.0112 | 18.3% |
| 90% | 400,384 | 0.1181 | 0.1294 | +0.0113 | 9.6% |
| 90% | 600,064 | 0.1787 | 0.1896 | +0.0109 | 6.1% |
| 90% | 800,768 | 0.2383 | 0.2491 | +0.0109 | 4.6% |
| 90% | 1,000,448 | 0.3002 | 0.3103 | +0.0100 | 3.3% |
| 90% | 1,200,128 | 0.3442 | 0.3554 | +0.0112 | 3.2% |
| 90% | 1,400,832 | 0.4023 | 0.4128 | +0.0105 | 2.6% |
| 90% | 1,600,512 | 0.4596 | 0.4705 | +0.0108 | 2.4% |

- PC−HP is CONSTANT per ratio, independent of batch: ≈0.045 s (50%), ≈0.028 s (70%), ≈0.011 s (90%) per block per step → direct empirical confirmation of the Fig. 2 claim (fixed per-step overhead, amortized by batch). HP throughput flat (~1.60M / 2.18M / 3.45M tok/s), PC approaches it.
- Extrapolated to 2M tokens (HP time = 2M / HP tok/s): 50% 0.045/1.25 s ≈ 3.6%; 70% 0.028/0.92 s ≈ 3.1%; 90% 0.011/0.58 s ≈ 1.9%.
- Scope: per-BLOCK numbers. Full model adds head/embedding: HP-equal per-token compute (lowers the relative overhead) + PC head/embedding projection fixed cost (raises it) — to measure separately if the paper figure should be whole-model.

## 100-step sweep @ 90% (2026-10-06) — the Fig. 2 data
16 batch sizes every ~100k tok (98..1563 seqs = 100k..1.6M), PC and HP, each a separate process on the SAME
1×H100 (srun job 23428), PC/HP interleaved per size. 100 steps, **mean step time over last 50**.
Raw: `benchmarks/results/bench_block_90p_100step.jsonl` (incl. per-step times). Plots: `mastest_pc_time.ipynb`
(plotly express; run in conda env `nanollmr`).

Result: PC − HP ≈ **10–12 ms per step, constant** across 100k → 1.5M tokens; relative overhead
35.0% @100k → 9.5% @400k → 4.6% @800k → 3.4% @1.0M → **2.5% @1.5M**. Step-time std < 1.1 ms everywhere.
HP throughput ~3.2–3.5M tok/s (small kernel-selection jump between 1.0M and 1.1M, same for PC and HP).

**31/32 runs OK. PC @ 1563 seqs (1.6M tok) fails reproducibly** (HP @1.6M OK; PC @1.6M passed in the earlier
10-step in-process sweep): `CUDA error: illegal memory access` in loss.backward() after the warm-up steps
(x2), and without expandable_segments `CUBLAS_STATUS_EXECUTION_FAILED` in a bf16 cublasGemmEx. Activation
tensors there exceed 2^31 elements (attention q ≈ 3.3e9) → likely a PyTorch/cuBLAS large-tensor edge case,
not our code. Not pursued; curve uses 15 complete pairs (100k–1.5M) + HP-only point at 1.6M.

## Paper Fig. 2 updated locally (2026-10-06) — NOT pushed, awaiting Maciej's review
- Original generator: `~/code/nano_local/nano/mastest.ipynb` cell 1 (analytic 6rd(d+r)/(6br^2), d=2048, r=960, plotly gold).
- New generator: `benchmarks/plot_overhead_fig2.py` = same analytic gold curve + measured blue (90%, 1 block,
  31 PC/HP pairs 50k..1.55M from both sweeps). Output `pc_paper/<id>/Figures/projection_compression_overhead_measured.png`
  (old PNG kept). mainAAAI.tex: \includegraphics → new PNG, new caption (both curves described), +1 sentence in
  "Amortization at Scale" paragraph citing ≈11 ms constant / 2.5% at 1.5M. Local PDF builds clean (19 pages).
- Midpoint sweep (srun 23449): 32/32 OK incl. PC @1.55M (only PC @1.6M fails, see above).
- Caveat for the paper: blue is 90% (r=448) while gold is r=960 (70%); measured ≈4–5× above analytic at ≥1M
  (likely fp32 projection math vs bf16 per-token compute — not profiled).

## Measurement window decision (Maciej, 2026-10-06)
Per-step times over the measured window (steps 51–100) are flat for all runs → method accepted:
100 steps, mean over steps 51–100. The 300-step fully-timed warm-up check (job 23453) was cancelled — not needed.
`bench_block.py` now also stores `all_step_times` (warm-up included) for future runs.

## 70% sweep — matches the analytic Fig. 2 setup (d=2048, r=960) (2026-10-06)
srun job 23454, 1×H100, 1 block, 19 batch sizes every 50k (49..928 seqs = 50k..950k), PC+HP, 100 steps,
mean of steps 51–100. **38/38 OK** — 70% one block fits ≥950k tokens (the earlier 10-step probe OOM was at 1M).
Raw: `benchmarks/results/bench_block_70p_100step.jsonl`.
PC − HP ≈ **25–29 ms/step, ~constant** (drifts 29→25 ms); overhead 121% @50k, 61.8% @100k, 30.5% @200k,
14.6% @400k, 9.4% @600k, 7.3% @800k, **6.0% @950k**. Analytic (r=960) at 950k = 0.68% → measured ≈ 9× analytic.

Paper Fig. 2 (local only, not pushed): now matched — gold analytic r=960 + blue measured 70%, ymax 20%.
`Figures/projection_compression_overhead_measured.png` = 70% version (included); `..._measured_90p.png` = earlier
90% version (not included). Caption + text sentence updated to 70% numbers (≈26 ms, 62% @100K → 6.0% @950K).
Notebook `mastest_pc_time.ipynb` now loads 70% + 90% (101 runs), plots faceted/colored by ratio.

## Fig. 2 switched to 90% (Maciej, 2026-10-06) — local only, not pushed
Gold analytic now r=448 (Llama 1B at 90%), blue measured 90% (31 pts, 50k–1.55M), ymax 20%.
Included file: `Figures/projection_compression_overhead_90p.png`. Kept for switching back:
`..._70p.png` (gold r=960 + measured 70%) and the original `projection_compression_overhead.png`.
Caption + in-text sentence reverted to 90% numbers (≈11 ms constant; 35% @100K → 2.5% @1.5M).
At 90%, measured ≈ 3× analytic at ≥1M (vs ≈9× at 70%). 70% vs 90% comparison: 70% relative overhead ≈1.6×
higher (fixed cost 2.4× larger ≈ analytic 2.6×; HP per-token time only 1.5× larger vs r² model's 4.6×).

## grad_acc = 4 sweep @ 90%, 1 block (2026-10-06)
Same method (100 steps, mean of 51–100), totals 52..1516 seqs (multiples of 4, ≈50k..1.55M), microbatch = total/4 (13..379 seqs).
bench_block.py --grad-acc K mirrors PCTrainer: W_c once → K micro fwd/bwd (accumulate W_c.grad) → projection bwd + AdamW once.
Raw: benchmarks/results/bench_block_90p_gacc4_100step.jsonl (named so the grad_acc=1 figure/notebook globs do NOT pick it up). 62/62 OK.
**PC−HP stays ≈11 ms/step (10.9–12.9) → overhead once per optimizer step confirmed empirically; relative overhead ≈ same as grad_acc=1 at equal total batch (±0.5 pp from 200k; lower at 50k: 57.8 vs 69.4% because HP slows with small microbatches). PC peak mem ≈3.8× lower (70.2→18.3 GiB @1.55M).**

| target | gacc=1 tokens | HP ms | PC ms | Δ ms | **ovh** | gacc=4 tokens (micro) | HP ms | PC ms | Δ ms | **ovh** | peak GiB PC g1→g4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50k | 50,176 | 16.2 | 27.5 | 11.3 | **69.4%** | 53,248 (13) | 18.9 | 29.9 | 10.9 | **57.8%** | 3.2→1.5 |
| 100k | 100,352 | 31.4 | 42.4 | 11.0 | **35.0%** | 102,400 (25) | 34.4 | 45.5 | 11.1 | **32.2%** | 5.4→2.1 |
| 150k | 150,528 | 46.1 | 57.2 | 11.1 | **24.2%** | 151,552 (37) | 49.4 | 60.7 | 11.2 | **22.7%** | 7.6→2.6 |
| 200k | 200,704 | 61.3 | 72.4 | 11.1 | **18.1%** | 200,704 (49) | 64.3 | 75.6 | 11.3 | **17.6%** | 9.9→3.2 |
| 250k | 250,880 | 75.5 | 87.0 | 11.5 | **15.2%** | 253,952 (62) | 80.3 | 91.5 | 11.2 | **14.0%** | 12.1→3.8 |
| 300k | 300,032 | 88.8 | 100.0 | 11.1 | **12.5%** | 303,104 (74) | 93.1 | 104.3 | 11.2 | **12.0%** | 14.3→4.3 |
| 350k | 350,208 | 104.6 | 116.0 | 11.5 | **11.0%** | 352,256 (86) | 109.9 | 120.8 | 10.9 | **9.9%** | 16.6→4.9 |
| 400k | 400,384 | 118.3 | 129.5 | 11.2 | **9.5%** | 401,408 (98) | 124.0 | 135.2 | 11.3 | **9.1%** | 18.8→5.4 |
| 450k | 450,560 | 133.2 | 144.4 | 11.2 | **8.4%** | 450,560 (110) | 136.7 | 148.2 | 11.5 | **8.4%** | 21.1→6.0 |
| 500k | 500,736 | 150.6 | 162.2 | 11.6 | **7.7%** | 503,808 (123) | 152.1 | 163.3 | 11.3 | **7.4%** | 23.3→6.6 |
| 550k | 550,912 | 163.9 | 175.1 | 11.2 | **6.8%** | 552,960 (135) | 167.6 | 178.9 | 11.3 | **6.8%** | 25.5→7.1 |
| 600k | 600,064 | 178.5 | 189.6 | 11.1 | **6.2%** | 602,112 (147) | 183.1 | 194.3 | 11.3 | **6.2%** | 27.7→7.7 |
| 650k | 650,240 | 193.1 | 204.3 | 11.2 | **5.8%** | 651,264 (159) | 198.2 | 209.4 | 11.2 | **5.7%** | 30.0→8.2 |
| 700k | 700,416 | 208.5 | 219.3 | 10.8 | **5.2%** | 700,416 (171) | 206.8 | 218.2 | 11.5 | **5.5%** | 32.2→8.8 |
| 750k | 750,592 | 226.6 | 237.8 | 11.2 | **4.9%** | 753,664 (184) | 222.9 | 234.2 | 11.2 | **5.0%** | 34.5→9.4 |
| 800k | 800,768 | 238.0 | 249.0 | 11.0 | **4.6%** | 802,816 (196) | 243.7 | 254.9 | 11.2 | **4.6%** | 36.7→9.9 |
| 850k | 850,944 | 252.6 | 263.8 | 11.2 | **4.4%** | 851,968 (208) | 254.5 | 265.9 | 11.4 | **4.5%** | 39.0→10.5 |
| 900k | 900,096 | 270.0 | 281.1 | 11.1 | **4.1%** | 901,120 (220) | 267.3 | 278.5 | 11.2 | **4.2%** | 41.2→11.0 |
| 950k | 950,272 | 285.6 | 296.3 | 10.6 | **3.7%** | 950,272 (232) | 285.2 | 296.9 | 11.8 | **4.1%** | 43.4→11.6 |
| 1000k | 1,000,448 | 300.0 | 310.1 | 10.1 | **3.4%** | 1,003,520 (245) | 300.8 | 312.8 | 12.1 | **4.0%** | 45.6→12.1 |
| 1050k | 1,050,624 | 301.4 | 311.9 | 10.6 | **3.5%** | 1,052,672 (257) | 310.5 | 321.7 | 11.2 | **3.6%** | 47.9→12.7 |
| 1100k | 1,100,800 | 315.8 | 326.4 | 10.6 | **3.4%** | 1,101,824 (269) | 333.1 | 344.9 | 11.8 | **3.6%** | 50.1→13.2 |
| 1150k | 1,150,976 | 329.9 | 340.7 | 10.9 | **3.3%** | 1,150,976 (281) | 348.1 | 359.8 | 11.7 | **3.4%** | 52.4→13.8 |
| 1200k | 1,200,128 | 344.4 | 355.4 | 11.1 | **3.2%** | 1,200,128 (293) | 353.2 | 364.5 | 11.2 | **3.2%** | 54.6→14.3 |
| 1250k | 1,250,304 | 360.6 | 371.0 | 10.4 | **2.9%** | 1,253,376 (306) | 369.6 | 380.6 | 11.0 | **3.0%** | 56.8→14.9 |
| 1300k | 1,300,480 | 372.4 | 384.2 | 11.8 | **3.2%** | 1,302,528 (318) | 384.0 | 395.3 | 11.3 | **2.9%** | 59.1→15.5 |
| 1350k | 1,350,656 | 387.0 | 398.0 | 11.0 | **2.8%** | 1,351,680 (330) | 398.4 | 409.5 | 11.1 | **2.8%** | 61.3→16.0 |
| 1400k | 1,400,832 | 402.1 | 412.4 | 10.3 | **2.6%** | 1,400,832 (342) | 417.0 | 428.8 | 11.7 | **2.8%** | 63.5→16.6 |
| 1450k | 1,451,008 | 416.3 | 427.0 | 10.7 | **2.6%** | 1,454,080 (355) | 435.5 | 447.1 | 11.6 | **2.7%** | 65.8→17.2 |
| 1500k | 1,500,160 | 430.9 | 441.5 | 10.6 | **2.5%** | 1,503,232 (367) | 449.7 | 462.7 | 12.9 | **2.9%** | 68.0→17.7 |
| 1550k | 1,550,336 | 445.0 | 456.2 | 11.2 | **2.5%** | 1,552,384 (379) | 455.8 | 467.3 | 11.5 | **2.5%** | 70.2→18.3 |

## Fig. 2 in the NEW paper repo (AISTATS, `pc_paper/6abc2d4dc7c9cbe423eadab0/Main.tex`) — local only, not pushed
Old AAAI clone (6a3e…) was deleted by Maciej; all Fig. 2 edits re-applied here.
- `Figures/projection_compression_overhead_90p.png`: 3 lines — gold analytic r=448, blue measured grad_acc=1,
  red measured grad_acc=4 (both 1 block, 90%, 1×H100). Script: `benchmarks/plot_overhead_fig2.py --r 448 --ymax 20
  --series "LABEL=GLOB" ...` (labels may contain "="; parser splits on the LAST "=").
- Caption rewritten for 3 curves; +1 sentence in "Amortization at Scale" (≈11 ms constant, 2.5% @1.5M, grad-acc
  leaves it unchanged since projections update once per optimizer step).
- PDF built outside the repo: `/home/maciej/code/pc_paper_build/Main.pdf` (22 pages, 0 errors, 0 undefined).
  Repo left clean apart from Main.tex + new PNG.

## Full-model sweep, 90%, grad_acc 64, 10 steps (2026-10-06 evening)
PC = SLURM 23484 (`pc90_gacc64_sweep`), HP = 23485 (`hp90_gacc64_sweep`), 20 sizes each (131k..2.03M tok, micro 2..31),
1×H100 on a fully loaded h100a. Submitted manually (GitHub port 22 blocked from entropy; cloned over 443 — see memory).
Llama-3.2-1B pre-downloaded into /storage_nvme_4/nano/hf_cache for HP. Analysis: `benchmarks/full_model_timing.py`
→ `benchmarks/results/full_90p_gacc64_10step.jsonl`; notebook section "Full Llama-1B model…".
- 40/40 finished. HP very stable (std7 ≤ 0.09 s). PC has sporadic slow steps in ~6/20 runs (e.g. 459k: 18.1/10.2/10.1 s
  vs ~6 s typical; 1.25M, 1.64M spikes to 16–19 s) → use MEDIAN of last 7.
- Median overhead: 21.8% @131k, 25.9% @524k, 21.7% @1.05M, 17.6% @1.31M, 16.6% @1.70M, 17.3% @1.84M, **14.1% @2.03M**
  (noisy 17–58% below 1M). PC−HP NOT constant: ~0.8 s @131k → ~2.0–2.5 s @1.5–2M → PC has a per-token/per-microbatch
  extra cost in the full model (not seen in the 1-block benchmark) on top of a ~1 s fixed cost.
- Consistent with the paper's Appendix D end-to-end 14–19%, but much higher than the 1-block 2.5% @1.5M.
- Not yet explained (needs profiling): candidates = embedding projection path per microbatch (2048-wide source lookup
  + projection fwd/bwd), FSDP2 hooks/reductions on PC's extra modules per microbatch, host-side per-block optimizer
  loop / .item() syncs; spikes may be host contention on the fully loaded node (HP unaffected).

## Fig. 2 now 4 lines (local, not pushed)
Gold analytic r=448; blue 1-block gacc1; red 1-block gacc4; green FULL model gacc64 (median of last 7/10 steps,
`benchmarks/results/full_90p_gacc64_10step_median7.jsonl`). ymax 40 (459k full-model point 58% off-chart), xmax 2.1M.
Caption extended with the green line (14% @2M). PDF: `pc_paper/6abc2d4dc7c9cbe423eadab0/Main.pdf` via the repo's
`compile.sh` (build/ + PDF excluded via .git/info/exclude). Repo diff: Main.tex + Figures/projection_compression_overhead_90p.png.

## Seed-averaged 1-block sweep, 90%, grad_acc 4, 50k..2M every 50k (2026-10-06 night) — FINAL Fig. 2 data
40 SLURM jobs (one per size, submitted 10 s apart from entropy login node via ~/submit_seed_jobs.sh), each: 5 seeds × {PC,HP}
back-to-back on the same GPU; 100 steps, mean of steps 51–100. bench_block.py --seed seeds weights, projection idx, data.
Raw: benchmarks/results/bench_block_90p_gacc4_seeds.jsonl (400/400 OK). Logs on entropy: ~/bench_seeds_gacc4/.
Result: PC−HP ≈ 11 ms constant (9.9–12.5); overhead 57.9% @53k, 32.8% @102k, 9.0% @401k, 4.0% @1.0M, 2.4% @1.55M, **1.96% @2.0M**;
seed std ≤ 0.08 pp (1.33 pp @53k). Paper Fig. 2 now = gold analytic r=448 + this series (mean ± std), ymax 20, xmax 2.1M;
caption + in-text sentence updated (33% @100K → 2.0% @2M, std < 0.1 pp). PDF: pc_paper/6abc2d4dc7c9cbe423eadab0/Main.pdf. Not pushed.

| tokens | HP ms (mean±std) | PC ms (mean±std) | PC−HP ms | **overhead % (mean ± std, n=5)** |
|---|---|---|---|---|
| 53,248 | 18.8 ± 0.1 | 29.7 ± 0.0 | 10.9 ± 0.2 | **57.85 ± 1.33** |
| 102,400 | 34.3 ± 0.0 | 45.6 ± 0.0 | 11.2 ± 0.0 | **32.77 ± 0.08** |
| 151,552 | 49.3 ± 0.0 | 60.6 ± 0.1 | 11.3 ± 0.1 | **22.87 ± 0.14** |
| 200,704 | 64.0 ± 0.0 | 75.2 ± 0.0 | 11.2 ± 0.0 | **17.49 ± 0.08** |
| 253,952 | 80.0 ± 0.0 | 91.2 ± 0.0 | 11.2 ± 0.1 | **14.03 ± 0.07** |
| 303,104 | 92.7 ± 0.0 | 103.9 ± 0.0 | 11.2 ± 0.0 | **12.08 ± 0.03** |
| 352,256 | 109.4 ± 0.0 | 120.5 ± 0.0 | 11.1 ± 0.1 | **10.15 ± 0.06** |
| 401,408 | 123.8 ± 0.0 | 134.9 ± 0.0 | 11.2 ± 0.1 | **9.02 ± 0.05** |
| 450,560 | 136.6 ± 0.0 | 147.9 ± 0.0 | 11.3 ± 0.0 | **8.27 ± 0.03** |
| 503,808 | 151.9 ± 0.0 | 163.1 ± 0.0 | 11.1 ± 0.0 | **7.33 ± 0.03** |
| 552,960 | 167.5 ± 0.0 | 178.8 ± 0.0 | 11.3 ± 0.0 | **6.75 ± 0.02** |
| 602,112 | 183.1 ± 0.1 | 194.4 ± 0.0 | 11.3 ± 0.1 | **6.17 ± 0.05** |
| 651,264 | 198.1 ± 0.0 | 209.3 ± 0.0 | 11.2 ± 0.0 | **5.63 ± 0.01** |
| 700,416 | 206.9 ± 0.0 | 218.2 ± 0.1 | 11.4 ± 0.1 | **5.50 ± 0.02** |
| 753,664 | 223.0 ± 0.0 | 234.2 ± 0.0 | 11.2 ± 0.0 | **5.02 ± 0.02** |
| 802,816 | 243.4 ± 0.0 | 254.6 ± 0.2 | 11.2 ± 0.2 | **4.61 ± 0.08** |
| 851,968 | 254.6 ± 0.0 | 266.1 ± 0.0 | 11.5 ± 0.0 | **4.52 ± 0.02** |
| 901,120 | 267.5 ± 0.1 | 278.6 ± 0.1 | 11.2 ± 0.1 | **4.17 ± 0.03** |
| 950,272 | 285.2 ± 0.0 | 297.1 ± 0.0 | 11.8 ± 0.0 | **4.14 ± 0.02** |
| 1,003,520 | 301.0 ± 0.2 | 312.9 ± 0.1 | 11.9 ± 0.2 | **3.96 ± 0.07** |
| 1,052,672 | 311.1 ± 0.1 | 322.0 ± 0.0 | 10.9 ± 0.1 | **3.49 ± 0.04** |
| 1,101,824 | 333.1 ± 0.0 | 344.9 ± 0.0 | 11.8 ± 0.1 | **3.53 ± 0.02** |
| 1,150,976 | 348.0 ± 0.0 | 359.9 ± 0.3 | 11.9 ± 0.2 | **3.41 ± 0.07** |
| 1,200,128 | 353.2 ± 0.0 | 364.5 ± 0.1 | 11.3 ± 0.1 | **3.19 ± 0.03** |
| 1,253,376 | 369.8 ± 0.2 | 381.0 ± 0.1 | 11.2 ± 0.2 | **3.03 ± 0.06** |
| 1,302,528 | 384.2 ± 0.1 | 395.3 ± 0.1 | 11.0 ± 0.1 | **2.87 ± 0.02** |
| 1,351,680 | 399.9 ± 0.1 | 409.9 ± 0.1 | 9.9 ± 0.1 | **2.49 ± 0.02** |
| 1,400,832 | 417.1 ± 0.0 | 428.6 ± 0.1 | 11.5 ± 0.1 | **2.75 ± 0.02** |
| 1,454,080 | 435.5 ± 0.2 | 447.0 ± 0.2 | 11.5 ± 0.3 | **2.63 ± 0.07** |
| 1,503,232 | 449.9 ± 0.1 | 462.4 ± 0.1 | 12.5 ± 0.1 | **2.77 ± 0.02** |
| 1,552,384 | 456.7 ± 0.1 | 467.7 ± 0.2 | 11.0 ± 0.3 | **2.42 ± 0.07** |
| 1,601,536 | 471.9 ± 0.0 | 483.0 ± 0.1 | 11.0 ± 0.1 | **2.34 ± 0.03** |
| 1,650,688 | 498.1 ± 0.2 | 509.8 ± 0.1 | 11.6 ± 0.1 | **2.34 ± 0.03** |
| 1,703,936 | 502.1 ± 0.1 | 513.0 ± 0.1 | 11.0 ± 0.1 | **2.18 ± 0.01** |
| 1,753,088 | 517.2 ± 0.1 | 527.9 ± 0.0 | 10.7 ± 0.1 | **2.06 ± 0.02** |
| 1,802,240 | 531.4 ± 0.1 | 542.7 ± 0.1 | 11.3 ± 0.2 | **2.13 ± 0.04** |
| 1,851,392 | 545.3 ± 0.4 | 556.0 ± 0.1 | 10.8 ± 0.3 | **1.97 ± 0.06** |
| 1,900,544 | 560.0 ± 0.0 | 571.5 ± 0.1 | 11.5 ± 0.1 | **2.05 ± 0.02** |
| 1,953,792 | 586.5 ± 0.1 | 597.1 ± 0.1 | 10.7 ± 0.1 | **1.82 ± 0.02** |
| 2,002,944 | 602.0 ± 0.1 | 613.9 ± 0.1 | 11.8 ± 0.1 | **1.96 ± 0.01** |
- 2026-10-07: Maciej couldn't see the variance bars (true seed std 0.01–0.08 pp ≈ few px on a 0–20% axis). Fix without
  inflating: black wide error-bar caps in the main plot + zoomed inset (1.0–2.05M) where ±1 std is clearly visible
  (`plot_overhead_fig2.py --inset 980000 2050000`). Caption mentions the inset. PDF rebuilt, 0 errors. Not pushed.
- 2026-10-07: Fig. 2 now uses 5%-trimmed means (per run: drop 2 fastest + 2 slowest of the 50 timed steps, scipy
  trim_mean convention) → `benchmarks/results/bench_block_90p_gacc4_seeds_trim5.jsonl`. Effect is negligible (median
  per-run change 0.003%, max 1.4%); only 53k changes (57.85±1.33 → 59.14±0.67%). 100k: 32.74±0.04, 1M: 3.96±0.07,
  2M: 1.97±0.01. Caption describes the trimming. PDF rebuilt, 0 errors. Not pushed.

## PUSHED to Overleaf (2026-10-07): commit 4d85189 on 6abc2d4dc7c9cbe423eadab0
Fig. 2 = gold analytic (r=448) + "Measured, 1 layer" (5-seed mean, 5%-trimmed, no error bars, ymax 15, xmax 2.1M);
short caption; one sentence in "Amortization at Scale" (≈11 ms constant, 2.0% @2M, std < 0.1 pp over 5 seeds).
No upstream changes at pull time (7e8e880 → 4d85189, fast-forward). Files: Main.tex + Figures/projection_compression_overhead_90p.png.
Regenerate figure: python benchmarks/plot_overhead_fig2.py --r 448 --ymax 15 --xmax 2100000 --no-errorbars \
  --series "Measured, 1 layer=benchmarks/results/bench_block_90p_gacc4_seeds_trim5.jsonl" --out <paper>/Figures/projection_compression_overhead_90p.png
