# GOAL: remove dead code from the pretraining branch

2026-11-03, surveyed at `7c0b3384`. Every item below was verified by a
whole-repo reference scan (py + yaml + sh + toml, so Hydra `_target_` strings
count as references) plus `ruff --select F401,F811,F841`. Paste this file to
an agent as the goal; it is self-contained.

## Objective

Delete the dead code listed in Findings, leave every anti-target untouched,
and land the result on `pretraining` with all gates green.

## Scope and constraints

- Touch only: `src/core/`, `src/definitions.py`, `main.py`, `run_exp.py`.
- Deletions only — no refactors, no renames, no behavior changes. If a
  deletion would require editing a *caller*, the symbol was not dead; stop
  and re-verify.
- Before deleting any symbol, re-run its reference scan at HEAD (grep the
  bare name across `*.py *.yaml *.sh *.toml`, excluding its own def line and
  `.pixi/`, `pc_paper/`, `__pycache__/`). A finding below is a claim about
  `7c0b3384`, not about HEAD.
- Two decision-gated items (D1, D2) need Maciej's confirmation before
  deletion; everything else is pre-approved by this goal.
- Commit in the three clusters below (reviewable units), each message citing
  this file; end commits with the Claude co-author line. Push to
  `git@github.com:llm-random/nano.git pretraining` (origin https cannot
  auth). Never delete branches.

## Findings

### Cluster 1 — dead file

- `src/definitions.py` (82 lines): imported nowhere. Aspirational typed-config
  layer (pydantic `BaseModel`s + dataclasses) that was never wired to Hydra;
  also internally broken — `from attr import dataclass` (line 12) shadows the
  stdlib `dataclass` imported on line 9, and `attr` is not a declared
  dependency pattern elsewhere in the repo. **D1 — ask Maciej first**: the
  `solve_config_lr` TODO in `src/core/utils.py:49` says "move to devinitions
  eval+ when created", so this may be a parked plan rather than residue.
  **Resolved 2026-11-03: DELETE** — Maciej wants the capability (Pylance
  typing + submit-time validation), but built properly; the replacement is
  specified in [typed_config_goal.md](typed_config_goal.md), which runs
  after this goal. Also remove the stale "move to devinitions" remark from
  the `solve_config_lr` comment.

### Cluster 2 — dead symbols in live modules (pre-approved)

| symbol | where | note |
|---|---|---|
| `LLamaFeedForward` | `src/core/llama.py:55` | superseded by `src/core/model.py` FF classes; keep `LlamaRoPE` + `repeat_kv`, which are live |
| `LlamaAttention` | `src/core/llama.py:98` | same — model.py's attention + registry is the live path |
| `cast_state_dict_to_tensors` | `src/core/utils.py:21` | never called; its one import (trainer.py:24) is itself flagged unused |
| `finish_exp` | `src/core/utils.py:41` | two-line stub returning `False`, unreferenced |
| `_find_latest_checkpoint` | `src/core/checkpointing.py:114` | unreferenced; checkpoint paths are explicit everywhere |
| `get_composition_file_path` | `src/core/metric_loggers.py:139` | unreferenced; its only plausible consumer was main.py's dead `upload_config_file` |
| `upload_config_file` | `main.py:73` | unreferenced wandb-era config upload |
| `log_environs` | `main.py:145` | unreferenced env-var scraper |
| `ensure_remote_config_exist` | `run_exp.py:66` | **D2 — ask Maciej first**: run_exp.py is the cross-cluster submission tool; confirm these two aren't kept for interactive/REPL use |
| `get_experiment_components` | `run_exp.py:210` | part of D2 |

### Cluster 3 — import cruft (pre-approved; 15 of 16 ruff-autofixable)

- `main.py`: dead `grid_generator` imports (lines 6–7 — the live users are
  run_exp.py and tests); duplicate `import os` (1, 12); triplicate
  `import logging` (9, 15, 17).
- `src/core/metric_loggers.py`: unused `torch` (5) and `torch.distributed`
  (9); duplicate `from abc import ABC, abstractmethod` (7, 8).
- `src/core/trainer.py`: unused imports `get_full_checkpoint_path` (18) and
  `cast_state_dict_to_tensors` (24).
- `run_exp.py:192`: `except ... as e` with `e` unused.
- `src/definitions.py` findings are moot if D1 deletes the file.
- Apply via `ruff check --select F401,F811,F841 --fix` **after** adding the
  noqa below, then eyeball the diff — do not trust `--fix` blind on main.py's
  import block.

## Anti-targets (do NOT touch)

- `main.py:8` `import resolver as _` — side-effect import; it registers
  OmegaConf resolvers. Mark `# noqa: F401` with a comment saying why, so the
  lint gate passes without deleting it.
- `src/core/attention.py` placeholder mechanisms that raise on forward —
  deliberate registry stubs (the file says so; they fail loudly by design).
- All trainer profiling/compile knobs (`profile_timers`, `kernel_profile`,
  `compile_model`, `compile_blocks`, `compile_loss`) and the
  `configs/mfu_audit/` family — live experiment instrumentation.
- `LlamaRoPE` and `repeat_kv` in llama.py (live: model.py:278, tests).
- Anything under `src/product_keys/` if it reappears via merge — other
  team's code.
- Config yamls: out of scope for this goal entirely
  (`test_no_new_orphan_model_configs` + `test_every_runnable_config_composes_and_resolves`
  already police them).

## Verification gates (all must pass before push)

1. `pixi run pytest tests/ -q` — full suite green, notably
   `test_every_src_module_imports`, `test_entry_point_imports`,
   `test_call_sites_match_the_function_signature`,
   `test_every_runnable_config_composes_and_resolves`.
2. `pixi run ruff check src main.py run_exp.py conftest.py update_pixi.py
   --select F401,F811,F841` — zero findings.
3. Re-run the dead-symbol reference scan on the final tree: every deleted
   name appears nowhere; no *new* zero-reference symbols were created by the
   deletions (e.g., a helper whose only caller was deleted).
4. CPU smoke: `pixi run python main.py --config-name tiny_local` composes and
   takes at least one training step (it ran on CPU for the overlay work; if
   the local box can't, say so explicitly rather than skipping silently).

## Acceptance criteria

- Clusters 2 and 3 deleted; D1/D2 resolved per Maciej's answer (deleted or
  explicitly recorded here as "kept on purpose" with one line why).
- All four gates green; diff contains deletions, noqa line, and nothing else.
- Pushed to `pretraining`; this file updated in the same push with an
  execution record section (date, commits, what D1/D2 decided).
