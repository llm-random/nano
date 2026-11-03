# GOAL: typed config schema — Pylance autocomplete + submit-time validation

2026-11-03. Follow-up to [dead_code_cleanup_goal.md](dead_code_cleanup_goal.md):
that goal deletes the old, never-wired `src/definitions.py` (D1); this goal
builds its replacement properly. Run this goal AFTER the cleanup goal.
Motivation (Maciej, 2026-11-03): IDE autocomplete/live typo-catching and
submit-time rejection of mistyped sweep keys are high-value; the
declare-a-knob-first friction is accepted.

## Objective

A stdlib-dataclass schema for the `common` and `trainer` config sections,
registered in Hydra's ConfigStore and merged under every runnable config, with
`cfg` annotated in entry points — so that (a) Pylance autocompletes and
typo-checks `cfg.common.*` / `cfg.trainer.*` in code, and (b) a mistyped key
or wrong-typed value in a config or sweep fails at compose/submit time on the
laptop, not on the cluster.

## Design decisions (pre-made, do not relitigate)

- New file `src/core/config_schema.py`. Do NOT resurrect `src/definitions.py`
  (deleted by the cleanup goal; it was attrs/pydantic-based, which Hydra
  cannot use, and described an outdated config shape).
- **stdlib dataclasses only** — no pydantic, no attrs. `omegaconf.MISSING`
  for required fields; `omegaconf.II("...")` for interpolated defaults
  (e.g. kv_heads defaulting to `${common.q_heads}` style values).
- **Scope: `common` and `trainer` only.** The model tree stays untyped — it
  is `_target_`-instantiated against real class `__init__`s (already de-facto
  key-checked) and recursive; schema cost exceeds benefit there. Dataset /
  cluster / infrastructure sections: out of scope this round.
- Strict mode stays ON. Policy change this implies: new trainer knobs are
  declared in the schema dataclass (with a default) BEFORE first use; the
  `+trainer.X=...` ad-hoc pattern is retired for schema'd sections — once
  declared, plain `trainer.X=...` works without the `+`.
- The `Trainer` class's own typed attributes remain the implementation; the
  schema mirrors them. A hygiene test (below) keeps the two from drifting.

## Steps

1. **Enumerate the live shape**: compose every runnable config
   (`configs/*.yaml`, `configs/mfu_audit/*.yaml` — same list
   `test_every_runnable_config_composes_and_resolves` uses) and union the
   keys/types under `common` and `trainer`. The schema must admit every one
   of today's green configs unchanged.
2. **Write `src/core/config_schema.py`**: `CommonSchema`, `TrainerSchema`
   (nested dataclasses for `scheduler`, `checkpoint`, `evaluation` as
   needed), and a `RootSchema` holding `common`, `trainer`, and
   explicitly-permissive `Any`/dict fields for the untyped sections (model,
   infrastructure, …) so strictness lands only where scoped.
3. **Register + merge**: `ConfigStore.instance().store(name="base_schema",
   node=RootSchema)`; prepend `- base_schema` to the defaults list of every
   runnable config.
4. **Annotate entry points**: `def main(cfg: RootSchema)` in main.py,
   `def submit_experiment(cfg: RootSchema)` in run_exp.py (duck-typed —
   runtime object stays DictConfig; the annotation is for Pylance).
5. **Grid path check**: confirm grid overrides from `create_grid_config` are
   applied by OmegaConf merge/update onto the schema-backed cfg, so a bogus
   grid key fails at generation. If any path dumps yaml without composing
   against the schema, add a compose-check in that path.
6. **Tests**:
   - sync test: `TrainerSchema` field names == the typed class attributes of
     `Trainer` (so neither can drift without CI noticing);
   - negative tests: `trainer.n_stepz=5` (typo key), `+trainer.bogus=1`
     (undeclared add), `trainer.n_steps=abc` (type) each fail composition
     with a clear error; one bogus grid key fails generation;
   - positive test: every runnable config still composes (existing hygiene
     test covers this — it must stay green unmodified).

## Anti-targets

- No behavior changes to training; this is schema + annotations only.
- Do not type the model tree, datasets, or clusters this round.
- Do not weaken or special-case `test_every_runnable_config_composes_and_resolves`
  — if a config fails against the schema, fix the schema (it must describe
  reality, not prescribe it), unless the config is genuinely wrong, in which
  case say so in the execution record.
- Do not touch `src/product_keys/` if present.

## Verification gates (all must pass before push)

1. `pixi run pytest tests/ -q` green, including the new tests.
2. `pixi run python main.py --config-name tiny_local` composes and takes a
   step on CPU.
3. The three negative overrides above each fail with a readable error.
4. Manual IDE check, recorded in the execution record: in main.py,
   `cfg.common.` autocompletes and a deliberate `cfg.trianer` gets a Pylance
   error squiggle.

## Acceptance criteria

- Schema live under every runnable config; entry points annotated; gates
  green; mfu_audit experiment configs unaffected (same composed values as
  before — spot-check one with `--cfg job`).
- Execution record appended to this file (date, commits, any config the
  schema exposed as wrong).
- Pushed to `git@github.com:llm-random/nano.git pretraining`. Never delete
  branches.
