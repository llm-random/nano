# GOAL: wire in src/definitions.py — recover its typed-config functionality

2026-11-03. Companion to [dead_code_cleanup_goal.md](dead_code_cleanup_goal.md),
whose D1 keeps `src/definitions.py` precisely so this goal can rewire it.
Run this goal AFTER the cleanup goal. Motivation (Maciej, 2026-11-03): IDE
autocomplete/live typo-catching and submit-time rejection of mistyped sweep
keys are high-value; the declare-a-knob-first friction is accepted.

## Objective

Rewrite `src/definitions.py` in place into a working typed-config layer:
stdlib dataclasses for the `common` and `trainer` sections, registered in
Hydra's ConfigStore and merged under every runnable config, entry-point `cfg`
annotated with them, plus a `validate(cfg)` recovering the file's original
pydantic intent (value-range checks). Result: (a) Pylance autocompletes and
typo-checks `cfg.common.*` / `cfg.trainer.*` in code; (b) a mistyped key or
wrong-typed value in a config or sweep fails at compose/submit time on the
laptop, not on the cluster; (c) type-valid but nonsensical values (negative
lr, warmup > n_steps) fail at the same moment.

## Why the file must be rewritten, not just imported

The current contents cannot be wired as-is (verified 2026-11-03 at `748209ab`):

- `from attr import dataclass` (line 12) shadows the stdlib `dataclass`
  (line 9), so the five "@dataclass" classes are attrs classes — Hydra
  structured configs require stdlib dataclasses.
- Its pydantic models (`TrainingConfig` etc.) carry `extra="forbid"` yet
  describe an older config shape missing most live trainer keys (distributed,
  checkpoint, compile/profiler knobs), so validating today's configs against
  them fails immediately.
- Nothing registers anything in a ConfigStore; no yaml references these
  classes; nothing imports the module.

"Recover its functionality" therefore means: keep the file and its role
(the repo's config schema; `src/core/utils.py:49` already points here), keep
the *intent* of its range types (`PositiveInt`, `NonNegativeFloat`, …), and
replace the implementation.

## Design decisions (pre-made, do not relitigate)

- The file stays `src/definitions.py`; contents fully replaced.
- **stdlib dataclasses only** for the schema — no pydantic, no attrs imports
  remain. `omegaconf.MISSING` for required fields; `omegaconf.II("...")` for
  interpolated defaults (e.g. kv_heads tied to `${common.q_heads}`).
- Range checks live in an explicit `validate(cfg)` function in the same file
  (plain asserts/raises with readable messages), called after composition in
  `main.py` and `run_exp.py:submit_experiment`. This recovers the pydantic
  `PositiveInt`-style intent without a second class hierarchy that would
  drift. (OmegaConf does not run `__post_init__` on merged values, so
  validators cannot live on the dataclasses themselves.)
- **Scope: `common` and `trainer` only.** The model tree stays untyped — it
  is `_target_`-instantiated against real class `__init__`s (already de-facto
  key-checked) and recursive. Dataset / cluster / infrastructure sections:
  out of scope this round; `RootSchema` holds them as permissive `Any`
  fields so strictness lands only where scoped.
- Strict mode stays ON for schema'd sections. Policy change this implies:
  new trainer knobs are declared in `src/definitions.py` (with a default)
  BEFORE first use; the `+trainer.X=...` ad-hoc pattern is retired for
  schema'd sections — once declared, plain `trainer.X=...` works without
  the `+`.
- The `Trainer` class's own typed attributes remain the implementation; the
  schema mirrors them. A hygiene test (below) keeps the two from drifting.

## Steps

1. **Enumerate the live shape**: compose every runnable config
   (`configs/*.yaml`, `configs/mfu_audit/*.yaml` — same list
   `test_every_runnable_config_composes_and_resolves` uses) and union the
   keys/types under `common` and `trainer`. The schema must admit every one
   of today's green configs unchanged.
2. **Rewrite `src/definitions.py`**: `CommonSchema`, `TrainerSchema` (nested
   dataclasses for `scheduler`, `checkpoint`, `evaluation` as needed),
   `RootSchema`, the `validate(cfg)` range checks, and the ConfigStore
   registration (`ConfigStore.instance().store(name="base_schema",
   node=RootSchema)`) executed on import.
3. **Merge into configs**: prepend `- base_schema` to the defaults list of
   every runnable config; import `src.definitions` where composition happens
   so registration runs (main.py, run_exp.py, conftest/test fixtures).
4. **Annotate entry points**: `def main(cfg: RootSchema)` in main.py,
   `def submit_experiment(cfg: RootSchema)` in run_exp.py (duck-typed —
   runtime object stays DictConfig; the annotation is for Pylance). Call
   `validate(cfg)` in both, right after composition/missing-key check.
   Update the stale "move to devinitions" remark at `src/core/utils.py:49`
   to point at what now exists.
5. **Grid path check**: confirm grid overrides from `create_grid_config` are
   applied by OmegaConf merge/update onto the schema-backed cfg, so a bogus
   grid key fails at generation. If any path dumps yaml without composing
   against the schema, add a compose-check in that path.
6. **Tests**:
   - sync test: `TrainerSchema` field names == the typed class attributes of
     `Trainer` (so neither can drift without CI noticing);
   - negative tests: `trainer.n_stepz=5` (typo key), `+trainer.bogus=1`
     (undeclared add), `trainer.n_steps=abc` (type) each fail composition
     with a clear error; one bogus grid key fails generation; one range
     violation (e.g. negative learning rate) fails `validate(cfg)`;
   - positive test: every runnable config still composes (existing hygiene
     test covers this — it must stay green unmodified).

## Anti-targets

- No behavior changes to training; this is schema + annotations + validation
  only.
- Do not type the model tree, datasets, or clusters this round.
- Do not weaken or special-case `test_every_runnable_config_composes_and_resolves`
  — if a config fails against the schema, fix the schema (it must describe
  reality, not prescribe it), unless the config is genuinely wrong, in which
  case say so in the execution record.
- Do not touch `src/product_keys/` if present.

## Verification gates (all must pass before push)

1. `pixi run pytest tests/ -q` green, including the new tests.
2. `pixi run python main.py --config-name tiny_local` composes, passes
   `validate(cfg)`, and takes a step on CPU.
3. The three negative overrides and the range violation above each fail with
   a readable error.
4. Manual IDE check, recorded in the execution record: in main.py,
   `cfg.common.` autocompletes and a deliberate `cfg.trianer` gets a Pylance
   error squiggle.

## Acceptance criteria

- Schema live under every runnable config; entry points annotated and
  validating; gates green; mfu_audit experiment configs unaffected (same
  composed values as before — spot-check one with `--cfg job`).
- `src/definitions.py` contains no attrs/pydantic imports and is imported by
  main.py and run_exp.py (no longer dead).
- Execution record appended to this file (date, commits, any config the
  schema exposed as wrong).
- Pushed to `git@github.com:llm-random/nano.git pretraining`. Never delete
  branches.
