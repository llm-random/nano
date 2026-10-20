# Task: SliceGPT baseline for the PC paper

Branch: `mas/nips` · Started: 2026-10-06 · Status: **planning (nothing implemented yet)**

## Goal

Add SliceGPT (Ashkboos et al., ICLR 2024) as a new baseline next to HP/HPD/PC/PCD/SVD:
compress Llama 1B (Llama-3.2-1B) and 8B (Llama-3.1-8B) by **50 / 70 / 90 %**, then run recovery
training (RFT) in our framework with the same data and token budgets as HP, and compare on the same metric.

**Decision (2026-10-06, Maciej): the main baseline is the HYBRID "SliceGPT + FFN pruning"**
(SliceGPT on the residual width d, HP-style channel pruning on dff). It's preferable for our comparison.
Optionally also run faithful SliceGPT (residual only) at 1–2 points so reviewers can't say we
weakened their method.

Sources (local):
- Paper: `slicegpt/slice_gpt_paper.pdf` (arXiv 2401.15024v2)
- Official repo: `slicegpt/TransformerCompression` (microsoft, HEAD `183cd92`), MIT license

## How SliceGPT works (paper)

- **Computational invariance (Thm 1):** in an RMSNorm-connected transformer, `RMSNorm(XQ)Qᵀ = RMSNorm(X)`
  for orthogonal Q, so the residual stream can be rotated by Q if the weights are rotated to match:
  `W_embd·Q`, `Qᵀ·W_in`, `W_out·Q`, `Qᵀ·W_head`.
- **LayerNorm → RMSNorm:** fold the norm scale `diag(α)` into the following linear; models that are
  already RMSNorm (Llama) only need the weight folded.
- **Per-layer Q_ℓ:** one global Q works poorly, so each block gets its own Q from PCA of the calibration
  signal (`C_ℓ = Σ XᵀX`, eigenvectors sorted by eigenvalue, fp64). Different Q per block requires an
  extra linear map **`shortcut_Q = Q_{ℓ-1}ᵀQ_ℓ`** in each residual connection (2 per layer, d×d after slicing).
- **Slicing:** keep the top-d principal directions. This deletes columns of the embedding, q/k/v and
  gate/up inputs, and rows of o_proj/down_proj outputs; the lm_head is sliced optionally.
- **Only the residual width d shrinks.** FFN hidden dim (dff) and attention inner dims (heads × head_dim)
  stay full.
- Paper only goes up to 30 % slicing. Llama-2-7B one-shot at 30 %: avg zero-shot 69.0 → 51.5. RFT uses LoRA
  (r=32, α=10). Calibration ablation: ≥128 samples, longer sequences are better; main runs use 1024 × 2048.
  Fp32 PCA is noticeably worse than fp64 (App. A.2).
- Conclusion explicitly lists combining with structural pruning (LLM-Pruner) as future work, which
  motivates our hybrid.

## Why dff can't be sliced by rotation (supports the hybrid)

The invariance holds only for the residual stream (through RMSNorm). dff sits behind elementwise
ops (`silu(gate(x)) * up(x)`), and rotations don't commute with elementwise functions. Attention inner
dims sit behind RoPE/softmax, same problem. Only **permutations / channel selection** commute with
elementwise ops, so dff can only be reduced by pruning, i.e. exactly HP's dff branch.

## Code map (official repo, `src/slicegpt/`)

| File | What it does |
|---|---|
| `rotate.py` | `rotate_and_slice_sequential` (main loop), `pca_calc` (fp64 cov + 1 % mean-diag damping, `eigh`), `rotate_*`/`slice_*` per weight, `random_orthogonal_upper_left` |
| `layernorm_fusion.py` | `fuse_ln_linear` (fold norm weight into next linear), `fuse_modules`, `replace_layers` |
| `modules.py` | `RMSN` — weightless RMSNorm, divides by `mean_dim = hidden_size` (ORIGINAL D) |
| `adapters/llama_adapter.py` | `CompressedLlamaDecoderLayer` (residual `x @ shortcut_Q + f(norm(x))`), module mapping |
| `slicing_scheduler.py` | `ConstSlicingScheduler` (same d everywhere), per-layer schedulers (worse on Llama-2 per App. A.4) |
| `model_utils.py` | `get_layer0_inputs` (Catcher), `get_signals` (hook on 2nd norm input + layer output) |
| `data_utils.py` | wikitext2/ptb/c4/alpaca calibration loaders (packed to max_seqlen) |
| `experiments/run_slicegpt.py` | CLI: sparsity → `d = int((1-s)·D)` rounded down to `round_interval` |

Main loop per layer: rotate+slice attn inputs with Q_prev → re-run layer to get pre-MLP-norm signal →
PCA → Q_mid → set `attn_shortcut_Q`, rotate+slice o_proj and MLP inputs → re-run → PCA on layer
output → Q_next → set `mlp_shortcut_Q`, rotate+slice down_proj. Activations are kept on CPU; one layer
on GPU at a time.

## Gotchas found in their code (must handle in our port)

1. **Can't run on our models as-is.** Pinned `transformers==4.41` can't load Llama-3.1/3.2 (`llama3` rope
   type needs ≥4.43). Our pixi env has **4.55** (torch 2.7.1), where their `CompressedLlamaDecoderLayer.forward`
   breaks (attn now returns 2 values, rotary passed as `position_embeddings`). The adapter name check only
   accepts `meta-llama/Llama-2*` / `meta-llama/Meta-Llama-3*`.
2. **RMSN divides by the original D**, not the sliced d. Our `src.core.model.RMSNorm` divides by
   `normalized_shape` = d. Port must set norm weight = **√(D/d)** after slicing (exact equivalence up to eps),
   otherwise activations get rescaled (×3.4 at d=178).
3. **Embedding mean-centering is unconditional** in `fuse_modules` (meant for LayerNorm models). For RMSNorm
   Llama it changes the function. Measure loss after fusion (their `--eval-fused-model` exists for this)
   and probably skip it.
4. **Defaults vs paper:** CLI default `final_orientation="random"` (random rotation inside the kept subspace;
   function-equivalent but matters for Adam RFT / fp16), paper describes PCA. `round_interval=8`.
   `do_slice_head=False` by default, and model fp16 + fp64 PCA. Default calibration is 128 × 2048 wikitext2,
   but the paper uses 1024 × 2048.
5. **The head must be sliced** (`do_slice_head=True`): an unsliced V×D head alone is 267M (1B) / 540M (8B),
   more than the 90 % budget (189M for 1B).
6. **Llama-3.2-1B has tied embeddings.** SliceGPT unties them (different Q at input and output). nano
   already trains untied (`copy_llama_model_weights_from_HF` loads lm_head as a separate param), so it's
   consistent with HP/PC.

## Our framework (relevant facts)

- HP pipeline = `checkpoint.load.type: huggingface` → `copy_llama_model_weights_from_HF` → `apply_functions`
  (`src.projected_compression.pruning.prune`, runs on each rank before FSDP) → standard trainer.
- nano `TransformerBlock` = `Residual(norm, attention)` + `Residual(norm, LlamaFeedForward)`; Llama FF
  modules: `ff_pre_act` (up), `gate`, `ff_post_act` (down). Attention: `q_proj/k_proj/v_proj/o_proj`.
  The config has separate `dmodel`, `datt`, `dff`, so a sliced model is `dmodel=d`, `datt=original`.
- HP dims (from `configs/pc_project_mas/llama_{1B,8B}_prune.yaml`):
  1B 50/70/90 % = d/dff 1344/5376, 960/3840, 448/1728; 8B = 3072/9216, 2160/6480, 1056/3168.
- Precedent: the SVD baseline in the paper = SVD-LLM compression + **full-parameter** recovery training
  (instead of LoRA). Do the same for SliceGPT.

## Parameter matching (computed 2026-10-06, untied emb+head as in nano, shortcut_Q counted)

**Hybrid (MAIN): same dff as HP, d chosen to match HP total params**

| | 50 % | 70 % | 90 % |
|---|---|---|---|
| 1B HP d / dff | 1344 / 5376 (802M) | 960 / 3840 (502M) | 448 / 1728 (189M) |
| 1B hybrid d / dff | **1256** / 5376 (800M) | **912** / 3840 (503M) | **432** / 1728 (188M) |
| 8B HP d / dff | 3072 / 9216 (4513M) | 2160 / 6480 (2606M) | 1056 / 3168 (938M) |
| 8B hybrid d / dff | **2744** / 9216 (4513M) | **1960** / 6480 (2610M) | **984** / 3168 (936M) |

The hybrid differs from HP only in how the residual stream is compressed: PCA rotation + slicing
(+ shortcut_Q) instead of channel selection. That's a very clean comparison.

**Faithful SliceGPT (optional): dff full, d matched on total params**

| | 50 % | 70 % | 90 % |
|---|---|---|---|
| 1B d (dff 8192) | 1048 (−49 % of D) | 664 (−68 %) | 256 (−88 %) |
| 8B d (dff 14336) | 2152 (−48 %) | 1272 (−69 %) | 472 (−89 %) |

- shortcut_Q overhead: up to 35M (1B 50 %) / 296M (8B 50 %).
- Params ≠ FLOPs: faithful SliceGPT spends less on embeddings → ~1.7× HP's non-embedding params at 1B 90 %.
  Matching non-embedding params instead would give 1B d=904/520/152, 8B d=2032/1152/384.
- Param counting convention for "50/70/90 %" in the paper still to confirm. HP models keep 0.535/0.335/0.126
  of the 1B params counted untied (0.509/0.306/0.106 counted tied).

## Decided approach (2026-10-06, Maciej)

1. **Hybrid SliceGPT + FFN pruning** is the main baseline: prune dff exactly like HP (same dff, same
   activation importances), slice the residual width d with SliceGPT only as far as the param budget requires.
2. **Implement everything in our repo**, based on their code: copy their algorithm (PCA, rotate/slice, norm
   folding) nearly verbatim with MIT headers; adapt the layer loop to nano; recovery training (RFT) runs through
   our standard pipeline. Their repo is used only to validate our port.
3. **Full-parameter RFT, nothing frozen** (incl. embedding, lm_head, shortcut_Q, norms), on the same data and
   budgets as HP/HPD. No LoRA, same argument as the SVD baseline in the paper.
4. **Match total params** to the HP model at each ratio (hybrid d in the param-matching table above).

## Implementation plan (FOR REVIEW — not started)

### Step 0 — prerequisites
- Assets on entropy: `llama1b_in_nano.pt` + 1B importances are already there (`/storage_nvme_4/nano/j321m/pc/`).
  8B needs `llama8b_in_nano.pt` + 8B importances copied from lem (same route as the 1B copy, see notes/no_p.md)
  or loaded from HF `meta-llama/Llama-3.1-8B` (gated; HF_TOKEN is forwarded by run_exp).
- No new dependencies (their code needs only torch; `datasets` is already in our env for wikitext2 in validation).

### Step 1 — model support for sliced models (small, backward compatible)
- `src/projected_compression/model.py`:
  - `Residual` gets optional `shortcut_dims=None`; if set, it owns `self.shortcut_Q = nn.Parameter(d_in×d_out)` and its
    forward becomes `x @ shortcut_Q + layer(norm(x))` (otherwise unchanged).
  - `TransformerBlock` gets `shortcut: bool = False` and passes the dims to both Residuals. It stays the same class,
    so the FSDP2 wrap policy (`modules_to_shard: TransformerBlock`) still covers the new params.
- New model configs `configs/_model/llama_3_1_sliced.yaml` and `llama_3_8_sliced.yaml` (copies of `llama_3_1.yaml` /
  `llama_3_8.yaml`): `dmodel=d` (residual, embedding, head, norms), `datt` = original (2048 / 4096), `dff` = HP's dff,
  `block_fn.shortcut: true`.
- Param-count unit test: instantiate each sliced config on the meta device and assert the count equals the
  param-matching table (±0.5 %).

### Step 2 — SliceGPT port: `src/projected_compression/slicegpt/`
| File | Content | Origin |
|---|---|---|
| `rotate.py` | `pca_calc` (fp64 cov, 1 % damping, eigh), `rotate_*` / `slice_*` per weight, `random_orthogonal_upper_left` | copied |
| `fusion.py` | `fuse_ln_linear`: fold `attention_layer.norm` → q/k/v, `ff_layer.norm` → gate/ff_pre_act, `head.norm` → head.linear; then set norm weights to 1 | copied, adapted |
| `nano_adapter.py` | mapping: embedding / q,k,v / o / ff_pre_act,gate / ff_post_act / norms / head.linear | new (replaces their HF adapter) |
| `slice.py` | `rotate_and_slice(model, calib_batches, d, orientation)` — their `rotate_and_slice_sequential`, adapted: layer-0 input = `model.embedding(ids)`, block forward = `block(x)` (RoPE/causal internal, packed seqs → no masks), pre-MLP signal via forward hook on `ff_layer.norm` input | adapted |
| `run_slicing.py` | offline entry point (below) | new |

Details to get right (from the code review):
- **dff pruning first, then SliceGPT on d.** Apply HP's dff pruning (dff branch of `pruning.prune`, same
  importances / indices) to the full-width model, then calibrate PCA on that dff-pruned model, so the PCA
  sees the actual network.
- After slicing, set every RMSNorm weight to **√(D/d)** (their `RMSN` divides by the original D; ours by d).
- **Skip embedding mean-centering** (LayerNorm-only step); confirm with validation 1 below.
- `do_slice_head=True`, const d for all layers incl. head; d rounded to multiple of 8.
- fp32 model on CPU, one block at a time on GPU; PCA in fp64. Calibration activations kept on CPU (fp32 for 1B ≈ 17 GB
  per signal list; for 8B store bf16 or use 512 samples to stay ≲ 150 GB RAM; h100a has 1 TB).
- Orientation: default `random` (repo default); `pca` behind a flag (open decision).

Offline slicing job (`run_slicing.py` + `configs/slicegpt/llama_{1B,8B}_slice_{50,70,90}.yaml`), 1 GPU, via run_exp on entropy:
1. Instantiate nano Llama (calibration seq len 2048 → RoPE length 2048), load base weights.
2. Prune dff (HP indices) → fuse norms → calibrate on **FineWeb-Edu, 1024 seqs × 2048 tokens**
   (our dataloader, seed different from the training seed 123) → rotate + slice to hybrid d → set norm weights.
3. Log to wandb `pc_v2`: param count, per-layer eigenvalue spectra, one-shot FineWeb-Edu loss (held-out batches,
   seq 1024 like training) = **0-token point**.
4. Save `slicegpt_hyb_{1b,8b}_{50,70,90}.pt` (+ JSON: d, dff, orientation, calib info) to `/storage_nvme_4/nano/j321m/pc/slicegpt/`.

### Step 3 — validation (gates Step 5)
| # | Test | Pass criterion |
|---|---|---|
| V1 | Invariance: fuse + rotate with d=D, dff full, no slicing, fp32 | FineWeb loss equal to original within ~1e-4; check the centering on/off effect |
| V2 | Norm-divisor fix: slice to d, compare `RMSNorm(weight=√(D/d))` vs a reference `RMSN(mean_dim=D)` | outputs equal up to eps effect |
| V3 | Reference cross-check: their **unmodified** repo in a separate venv (transformers 4.41, datasets 2.18) on `meta-llama/Meta-Llama-3-8B` (supported by their adapter), sparsity 0.25, wikitext2, 1024 × 2048, `final_orientation=pca`, fp16; our port on the same model (nano `llama_3_8` with `rope_scale_freqs: false`), same settings, faithful mode (no dff pruning) | WikiText2 ppl within ~1 % of theirs (compute ppl with a copy of their `prepare_test_dataloader` for identical tokens) |
| V4 | Sliced checkpoint loads into `llama_3_1_sliced` config and gives the same loss as the in-memory sliced model | exact match |
| V5 | Short RFT smoke run (100 steps, 4 GPUs, FSDP2) | runs, loss decreases, shortcut_Q receive grads |

### Step 4 — RFT plumbing + configs
- `main.py`: new `checkpoint.load.type: sliced_nano` branch (like `finalized_pc`): instantiate model from config, load the
  sliced state dict (strict), then `get_model_optimizer_scheduler`.
- Configs (mirroring `llama_1B_prune.yaml` / `llama_1B_prune_distillation.yaml`, entropy version like the validated repro):
  `configs/pc_project_mas/llama_{1B,8B}_slicegpt_hyb.yaml` (trainer `llama`) and
  `llama_{1B,8B}_slicegpt_hyb_distillation.yaml` (distillation trainer, teacher Llama-3.2-1B / 3.1-8B as in HPD).
- Identical to HP/HPD: FineWeb-Edu, seed 123, seq 1024, batch 512 (accum 8), `^n_steps: [2048, 4096, 8192, 12288]`,
  cosine schedule as in the HP trainer, wd 0.1, clip 1.0, all params trainable, `save.type: nano` (not HF-exportable).
- wandb project `pc_v2`, tags `slicegpt`, `hybrid`, ratio, budget.

### Step 5 — experiments (in this order)
| Stage | Runs | Purpose |
|---|---|---|
| E1 | slicing jobs 1B × {50,70,90} (+ 0-token loss) | sliced checkpoints |
| E2 | LR sweep 1B: SG-H at 1BT, LR ∈ {10, 11, 12, 13} × 3 ratios (12 runs) | per-ratio LR, as for other methods |
| E3 | 1B SG-H + SG-H-D × 3 ratios @ 2BT (6 runs) | Table 2-style column first |
| E4 | 1B remaining budgets 1/4/6.4BT (18 runs) | full Table 1 rows |
| E5 | 8B: slicing × 3, short LR sweep, then same grid (priority 2BT) | Table 1 8B rows |
| X1 | faithful SliceGPT (no dff pruning, d = 1048 / 256) 1B 50 % and 90 % @ 2BT | show the hybrid isn't weakening their method |
| X2 | their recipe: LoRA r=32 α=10 (default targets) 1B 50 % @ 2BT | show full-param ≥ LoRA (same argument as SVD) |

Compute estimate: 1B RFT ≈ 2 h/BT on 4×H100 (PCD repro: 2.1BT in 4.25 h; the hybrid ≈ HP cost) → E2 ≈ 25 h, E3 ≈ 25 h,
E4 ≈ 100 h, X ≈ 15 h (node-hours, 2 runs fit in parallel on the 8-GPU h100a). 8B ≈ 5–8× per token.

### Step 6 — evaluation & paper
- Loss tables: same metric as Table 1 (final `steps/100/train/loss`), rows "SliceGPT+FFN-P" / "…-D".
- Downstream (9 benchmarks): needs a nano-model lm-eval wrapper or an export to their format (decide after E3).
- Paper text: method description, "+FFN pruning" labelled and motivated by their conclusion; note full-param RFT
  (like SVD), param-matched widths, calibration data.

### Acceptance / what you review now
- [ ] approach & decisions 1–4 above
- [ ] module-level changes in Steps 1, 2, 4 (model.py, new package, main.py loader, configs)
- [ ] validation criteria V1–V5 (esp. V3 as proof of a faithful port)
- [ ] experiment order + compute (E1–E5, X1–X2)

## Open decisions
- [x] Hybrid as main baseline (Maciej, 2026-10-06)
- [x] Full-param RFT, nothing frozen incl. shortcut_Q / embedding / head (agreed 2026-10-06)
- [x] Implement in our repo based on their code; their repo only for validation (agreed 2026-10-06)
- [x] Match total params (hybrid d = HP d minus shortcut overhead)
- [ ] Orientation: `random` (repo default) vs `pca` (paper)
- [ ] Calibration: FineWeb-Edu 1024 × 2048 (proposed) vs wikitext2
- [ ] Param-counting convention for 50/70/90 % (untied as in nano, or tied) — affects d slightly
- [ ] Run faithful SliceGPT at a few points too?
- [ ] Scope / order given compute
- [ ] Paper label, e.g. "SliceGPT + FFN pruning", citing their conclusion

## Log
- 2026-10-06: read paper + official code; findings, param matching and plan above. Maciej chose the hybrid. Nothing implemented.
- 2026-10-06: decided full-param RFT (nothing frozen), param matching, in-repo port based on their code. Detailed implementation plan written for review.

## Implementation status (2026-10-06) — Steps 1–4 implemented, testing in progress
Worktree **`/home/maciej/code/nano-slicegpt`**, branch **`slicegpt-baseline`** (from `mas/nips` cb035132), uncommitted.
Code mirrored on entropy at `/home/j321m/slicegpt_dev/nano-slicegpt` (smoke tests only).

| Item | Files |
|---|---|
| Residual `shortcut_Q` (optional, backward compatible) + `TransformerBlock(shortcut_dim)` | `src/projected_compression/model.py` |
| Port: adapter, fusion (+ fixed-divisor `RMSN`, `finalize_norms` → RMSNorm(d) weight √(D/d)), PCA/rotate/slice, layer loop, dff pruning | `src/projected_compression/slicegpt/{nano_adapter,fusion,rotate,slice}.py` |
| Offline slicing job (`init_model_opt_sched_fn`) — FineWeb-Edu calib, 0-token loss, saves `.pt` + `.json` | `src/projected_compression/slicegpt/run_slicing.py`, `configs/slicegpt/llama_{1B,8B}_slice_hyb_entropy.yaml` |
| Sliced model configs | `configs/_model/llama_3_{1,8}_sliced.yaml` |
| Loader `checkpoint.load.type: sliced_nano` | `main.py`, `src/core/checkpointing.py` |
| RFT configs (SG-H, SG-H-D; 1B + 8B; `slicegpt_ratio` r50/r70/r90; LR = HP/HPD placeholder) | `configs/pc_project_mas/llama_{1B,8B}_slicegpt_hyb[_distillation]_entropy.yaml` |
| Unit tests (CPU, tiny model) | `test_slicegpt.py` (`python test_slicegpt.py`) — **9/9 pass** |
| Reference cross-check (V3) | `scripts/slicegpt_crosscheck.py` |
| Smoke configs + entropy env runner | `smoke_configs/` |

Unit tests cover: V2 norm equivalence; fusion is function-preserving; embedding centering changes RMSNorm models;
V1 invariance at d=D (fp64, both orientations, shortcut_Q orthogonal); `slice_head=False` mode; dff selection == HP;
V4 save → strict load into `llama_3_1_sliced` → identical logits & params; hybrid param counts == HP (±0.5 %, 1B+8B);
all 6 configs compose for r50/r70/r90.

Notes from implementation:
- slicing configs needed `trainer.n_steps: 1` (trainer warmup interpolation must resolve for the wandb config dump).
- `rotate_and_slice(slice_head=...)`: True for our baseline; False reproduces their CLI default for the cross-check.
- 8B configs: assets (llama8b_in_nano.pt, 8B importances) still on lem only; 8B HP setup (seq 2048, accum 32,
  1M tokens/step, `^n_steps: [1024, 2048, 4096, 6144]`) copied from `llama_8B_prune.yaml` — verify vs wandb before runs.

### Smoke / validation results on entropy (h100a)
- **S1 / V1 (real Llama-3.2-1B, fp32, 64 calib seqs, d=D, no dff pruning, pca): PASS.** FineWeb-Edu eval loss
  base 2.5243111 → fused 2.5243108 → rotated 2.5243107 (|Δ| ≈ 4e-7). Params 1.4985B → 1.6327B (= +2·16·2048² shortcut_Q). ~2.5 min.
- **S2 (real hybrid r50 slicing, 64 calib seqs, random orientation): PASS.** 799.70M params (target 800M; 0.534 of untied base =
  same as HP 50 %). FineWeb-Edu eval loss: base 2.524 → after HP-dff pruning (8192→5376) 4.218 → fused 4.218 → **one-shot sliced
  (d=1256) 5.129** (0-token point, small calibration). 221 s. Saved to `/storage_nvme_4/nano/j321m/pc/slicegpt_smoke/`.
- Env notes: j321m's entropy pixi env has **transformers 5.12.1 / huggingface_hub 1.20.1 / datasets 3.1.0**, no `accelerate`
  (local env: transformers 4.55). Official SliceGPT runs from isolated `~/slicegpt_dev/ref_libs` (transformers 4.41, tokenizers 0.19.1,
  huggingface_hub 0.25.2, accelerate 0.30.1, ml-collections) layered on the same torch 2.7.1. With hub ≥1.0 WikiText is `Salesforce/wikitext`.
- V3 uses token ids dumped by THEIR `data_utils` in their env (`scripts/slicegpt_dump_ref_tokens.py`) → identical calibration/test tokens.

## PLAN CHANGE (2026-10-06): one-shot SliceGPT first, no recovery training
Maciej: focus on **one-shot** SliceGPT (no RFT for now). Faithful method (no dff pruning): compress **dmodel to exactly HP's width**,
keep dff = 8192 → same dmodel as HP, more params than HP. RFT jobs (LR sweep 23490, jump diagnostic 23488) cancelled;
V3 cross-check (23489, validates the one-shot slicing vs the official repo) kept.

### Llama-3.2-1B: one-shot SliceGPT (same dmodel as HP, dff full) vs HP
Params counted as in nano (untied embedding + head, RMSNorm weights; attention inner dims unchanged for both). Base = 1498.5M.
SliceGPT slices the head too (head in_features = d) and adds 2 × 16 shortcut_Q (d×d).

| Compression | Method | dmodel | dff | shortcut_Q | Params | Kept vs base | vs HP |
|---|---|---|---|---|---|---|---|
| 50 % | HP | 1344 | 5376 | – | 801.7M | 0.535 | 1.00× |
| 50 % | SliceGPT (one-shot) | 1344 (−34.4 % of D) | 8192 | 57.8M | 1041.2M | 0.695 | 1.30× |
| 70 % | HP | 960 | 3840 | – | 501.9M | 0.335 | 1.00× |
| 70 % | SliceGPT (one-shot) | 960 (−53.1 % of D) | 8192 | 29.5M | 731.9M | 0.488 | 1.46× |
| 90 % | HP | 448 | 1728 | – | 188.8M | 0.126 | 1.00× |
| 90 % | SliceGPT (one-shot) | 448 (−78.1 % of D) | 8192 | 6.4M | 334.2M | 0.223 | 1.77× |

SliceGPT param breakdown: 50 %: emb+head 344.8M, attn 110.1M, mlp 528.5M · 70 %: 246.3M / 78.6M / 377.5M · 90 %: 114.9M / 36.7M / 176.2M.
Run (when approved) = existing slicing job with `slicegpt.target_d={1344,960,448}`, `slicegpt.target_dff=8192` (no dff pruning),
full calibration (1024 × 2048 FineWeb-Edu), metric = one-shot FineWeb-Edu loss (+ params, kept variance).

### RESULTS: one-shot SliceGPT vs one-shot HP, Llama-3.2-1B (job 23527, 2026-10-06; wandb pc_v2 `llama1_slicegpt_oneshot_*`)
FineWeb-Edu CE, 100 × 1024 eval samples (seed 7), no recovery training. Base 2.548 (fusion: 2.548001 → 2.548001).
SliceGPT: faithful (dff 8192), random orientation, calib 1024 × 2048 FineWeb-Edu (seed 42), head sliced; ~20 min/ratio on 1 H100.
HP reference = `pruning.prune` (same dmodel, HP dff) evaluated in the same job on the same batches.

| Compression | dmodel | HP one-shot | HP params | SliceGPT one-shot | SliceGPT params |
|---|---|---|---|---|---|
| 50 % | 1344 | 8.800 | 801.7M | **3.706** | 1041.2M (1.30×) |
| 70 % | 960 | 10.571 | 501.9M | **4.690** | 731.9M (1.46×) |
| 90 % | 448 | 11.404 | 188.8M | **6.531** | 334.2M (1.77×) |

Checkpoints: `/storage_nvme_4/nano/j321m/pc/slicegpt/llama1b_slicegpt_oneshot_{r50_d1344,r70_d960,r90_d448}.pt` (+ .json).
8B: job 23566 submitted (d 3072/2160/1056, dff 14336, HF Llama-3.1-8B, 8B importances copied to `~/slicegpt_dev/assets/importances/8b/`), pending on j321m GPU QOS.

### RESULTS: one-shot SliceGPT vs one-shot HP, Llama-3.1-8B (job 23566, 2026-10-06/07; wandb pc_v2 `llama8_slicegpt_oneshot_*`)
Same protocol as 1B (100 × 1024 eval, seed 7). Base 8.03B params, loss 2.218 (fusion 2.2184839 → 2.2184839). ~80–90 min/ratio on 1 H100.

| Compression | dmodel | HP one-shot | HP params | SliceGPT one-shot (dff 14336) | SliceGPT params |
|---|---|---|---|---|---|
| 50 % | 3072 | 6.753 | 4.51B | **2.769** | 6.63B (1.47×) |
| 70 % | 2160 | 10.114 | 2.61B | **3.791** | 4.53B (1.74×) |
| 90 % | 1056 | 12.119 | 0.94B | **5.342** | 2.14B (2.28×) |
- 2026-10-07: one-shot SliceGPT rows added to paper Table 1 (`tables/table_loss.tex`, Overleaf project 6abc2d4dc7c9cbe423eadab0,
  commit 5e0ff7a, pushed): loss in the first token-budget column, dashes elsewhere; caption notes one-shot / d_model-only / param overhead.
