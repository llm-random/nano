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
- **2026-10-30 13:20** Phase 0 baseline: 145 tests pass (CI filter; 6 deselected),
  seeded tiny_local(20 steps) step19 loss 12.162640571594238, fineweb smoketest
  final_ce 12.290434122085571.
- **2026-10-30 13:35** Phase A done: live transformer moved PC→core verbatim;
  `ProjectedLlamaFeedForward` renamed `GatedFeedForward` (only deviation; body and
  init order untouched). PC model.py is now a re-export shim. Gate passed: 145
  tests, both seeded runs **bit-identical**. Side effect to know: main.py's
  `isinstance(module, Residual)` metric-logger hookup used to compare against the
  dead core class and never matched; it matches now, so residual update-norm
  logging actually engages (logging only, loss unaffected — verified by the gate).
- **2026-10-30 13:35** Ordering fix vs plan: configs targeting the PC package
  (smollm/, llmrandom/, mastest/old/) and configs orphaned by pc_project deletion
  must go in the SAME commit as the package, or composition/orphan tests break —
  C.1/C.3/C.4/C.6 collapse into the Phase-B sweep.
- **2026-10-30 14:05** Phase B done (one sweep commit): removed projected_compression
  (incl. pruning, mem_eff, SliceGPT) and product_keys packages, distillation
  (core trainer + 3 _trainer configs), all PC/PK/smollm/llmrandom configs, smoke/,
  3 slicegpt scripts, 3 PC test files; main.py lost the sliced_nano/finalized_pc
  branches; Residual/TransformerBlock lost the SliceGPT shortcut_Q; exp.yaml (the
  main.py default config) rewritten to tiny+fineweb, fresh init. test_distributed
  kept its 2 generic tests; save-types contract re-pinned on a stub subclass.
  Orphan list now empty. Gate: 122 passed, both seeded runs bit-identical, black ok.
  NOTE configs/mastest/ was never tracked (untracked scratch in the main checkout) —
  the plan's "personal scratch" item shrinks to configs/_mas/ + _checkpoints/mas/.
