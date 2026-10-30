# Pretraining reset — running log

Short timestamped reminders for the multi-stage reset of the `pretraining` branch.
One line per event; details live in the stage plan documents, not here.

## Context (read once)

- `pretraining` branched from `main` at `e894422f` (post-#199 audit). `main` keeps the
  full PC project; this branch strips everything not needed for the new project:
  **pretraining models with different attention types** (GQA/MHA/MQA/MLA/Mamba-2/Gated
  DeltaNet + our AME), H-balanced lineage, mixable per-layer attention patterns.
- Key structural fact: the *live* generic transformer classes sit in
  `src/projected_compression/model.py`; `src/core/model.py` holds a second, dead stack
  no config targets. Every base llama config instantiates the PC-package classes.
- Work happens in the worktree `/home/maciej/code/nano-pretraining`, branch `pretraining`.

## Put aside for the future (do NOT lose these)

- **Eval harness**: removed, to be rewritten from scratch later (old one had problems).
- **MoE**: removed, to be reimplemented later with efficient granularity CUDA kernels.
- **Model/training configs**: current llama lineage goes; replaced by **H-balanced**
  lineage in stage 2 (`documents/concepts/h-balanced-model.md` stays as the spec).
- **Stage 2 ingredients** (user's words, not yet started): H-balanced architectures,
  GPT-NeoX tokenizer (~50k vocab), gated FF, cosine scheduler, fineweb-edu dataset,
  switchable GQA/MHA/MQA/MLA/Mamba-2/Gated-DeltaNet, AME attention, layer-pattern
  mixing (every 4th / every 2nd / all / selected layers).
- **pixi.toml dependency pruning** (lm_eval etc.): deferred — touching deps means
  re-solving `pixi.lock`, which must stay v6 (hygiene test guards it; re-solve only
  with pixi ≤0.67.x). Do as its own step later.
- Anything deleted here still exists on `main` — resurrect with
  `git checkout main -- <path>` if a removal turns out premature.

## Log

- **2026-10-30 10:40** Branch `pretraining` created from `main@e894422f`, pushed,
  tracking fixed to `origin/pretraining` (push over SSH; `origin` URL is https).
- **2026-10-30 10:45** Worktree `/home/maciej/code/nano-pretraining` added. Stale
  `nano-baseline` worktree removed (branch kept, PR #199 merged tree-identical).
- **2026-10-30 10:50** Stage-1 removal plan written:
  `documents/notes/pretraining_stage1_removal_plan.md`. **Waiting for green light.**
- **2026-10-30 13:02** Green light received as /goal. Override vs plan: **no cluster
  GPU runs in stage 1** (entropy smoke skipped); local CPU checks only. Decisions I'd
  otherwise ask about get logged under "For Maciej" below and batched at the end.
