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

## Execution record (2026-11-03)

Executed at `6874efe7`, after [dead_code_cleanup_goal.md](dead_code_cleanup_goal.md),
in one working tree under the master goal (one commit per plan, single push).

**Implementation notes** — deviations forced by measured Hydra/OmegaConf
behavior; the design intent (strict common+trainer, open everything else,
submit-time rejection) is unchanged:

- The stored `base_schema` node is a plain-dict root holding structured
  `CommonSchema`/`TrainerSchema` children, not `OmegaConf.structured(RootSchema)`:
  struct *flags* inherit downward, so flag-based strictness on `trainer` also
  closed its untyped subtrees (`train_dataloader`, `distributed.fsdp2`), which
  merge freely today. Object-type strictness scopes exactly to common/trainer.
  `RootSchema` remains the entry-point annotation type for Pylance.
- Hydra's `+key=` force-add bypasses struct validation at compose (measured),
  so `+trainer.bogus=1` is caught one layer later by `check_point()` — a
  schema merge + `validate()` run by `main()` on every composed config and by
  `submit_experiment` on every expanded grid point. `check_point` converts its
  input to a plain container first, because merging a node that already
  carries the schema's object_type skips per-key validation (measured).
- `validate()` runs per expanded grid point, not on the raw sweep config: a
  caret sweep (`^learning_rate: [...]`) legitimately leaves the plain key
  null until expansion.
- `^key` grids cannot sit under `common`/`trainer` in a config whose defaults
  include `- base_schema` (compose rejects them as unknown keys). A caret
  sweep config simply omits that line and loses nothing: every expanded point
  is schema-checked at submit by `check_point`, which is where sweep typos
  surface. Cluster-side runs compose the generated flat configs (no defaults
  list, no schema) and are re-checked by `main()`'s `check_point`.
- Configs in a group directory (`configs/mfu_audit/*.yaml`) reference the
  node as `- /base_schema`; the leading slash is absolute. A bare
  `- base_schema` there is resolved relative to the config's own group and
  Hydra looks for `mfu_audit/base_schema`. This only ever affected
  `--config-name mfu_audit/h20`, which has never been a usable invocation —
  before the schema it composed everything nested under a single `mfu_audit:`
  key, with no top-level `common`/`trainer`, and died on resolve with
  `InterpolationKeyError`. The absolute spelling is correctness hygiene, not
  a repair: that invocation is still unusable, and the repo's convention
  (`--config-path configs/mfu_audit --config-name h20`) composes identically
  before and after the change, verified on all five configs.
- `~trainer.X` deletion overrides no longer work on schema'd keys (typed
  nodes do not support deletion); plain `trainer.eval_dataloader=null` (or
  `=None` for the historical string idiom) replaces them. The eval-off
  subprocess test in tests/test_core_framework.py was updated to the plain
  spelling — same purpose, the literal string "None" still flows through
  main.py. Likewise `+trainer.X=...` is retired for schema'd sections: the
  four instrumentation knobs (`autocast_bf16`, `profile_timers`,
  `kernel_profile`, `compile_loss`) are declared in the schema and spelled
  plain (`trainer.compile_loss=true`). Top-level `+compile_model=true` /
  `+compile_blocks=true` still work (open root).

**Config the schema exposed as wrong:** `configs/tiny_remote.yaml` and
`configs/tiny_remote_wandb.yaml` do not compose at HEAD at all — their
defaults reference `_eval/basic`, which `efe41424` (lm-eval harness removal)
deleted. Pre-existing, invisible to the hygiene test (its glob only sweeps
`configs/*/*.yaml`). Outside both plans' scope: left broken on purpose,
`- base_schema` prepended like the rest, flagged here for a separate fix.

**Composed-values spot check** (mfu_audit/h20, fully resolved, HEAD vs this
tree): identical except the declared defaults materialize —
`trainer.{autocast_bf16,profile_timers,kernel_profile,compile_loss}: false`,
`trainer.distributed.fsdp: null` and `common.attention.mix: null` (also
mirrored into the model tree via `${common.attention}`;
`build_attention` treats a null mix as absent) — the values the code
already assumed. The `tiny_local` smoke after all changes reproduces the
pre-schema run bit-identically — same final_ce before and after, 3/3
repeats. The absolute figure is environment-specific and is not a portable
gate value: this box (torch 2.7.1+cu128, 6 CPU threads) yields
8.451301097869873, while a reviewer on different CPU threading measured
8.451300477981567. Compare before-vs-after on one machine rather than
against a number copied from here.

**Fresh-eyes review** (independent subagent over the combined diff, per the
master goal) confirmed three bugs, all fixed here and pinned in
tests/test_typed_config.py:

1. `scripts/expand_config.py` composes runnable configs but did not import
   `src.definitions`, so every schema-backed config failed there with
   `Could not load 'base_schema'` — the one composer the rewire missed.
   Fixed with the registration import; pinned by a subprocess test.
2. `validate()` rejected `trainer.n_steps=0`, which is how a
   finalize/conversion pass is configured (src/core/schedulers.py allows it
   explicitly, and the very next warmup check carved it out, unreachably).
   Now `>= 0`.
3. `AttentionSchema` omitted `mix` — the live per-layer attention override
   (`attention: {type: gqa, mix: {type: ame, layers: every_4th}}`,
   configs/_model/h_balanced/base.yaml), which no route could submit any
   more since check_point guards grid points too. Declared `mix: Any = None`.

**IDE gate (verified against live Pylance):** main.py was
opened in VS Code via the remote CLI (Pylance only analyzes open files) and
diagnostics were read back through the editor API. The deliberate
`config.trianer` on the annotated entry point produced the Pylance error
`Cannot access attribute "trianer" for class "RootSchema"`
(reportAttributeAccessIssue), and the valid `config.common.sequence_length`
access drew no diagnostic — the same symbol table drives `cfg.common.`
completion. Making the gate honestly green also meant clearing the Pylance
errors the new annotations introduced at DictConfig boundaries:
`initialize_training_components`/`run` now take `cfg: RootSchema`;
`typing.cast` marks the runtime-is-DictConfig seams (create_grid_config,
run's local path, the wandb full_config, the already-validated
learning_rate); `get_metric_logger`'s full_config is `Optional[DictConfig]`;
and `RootSchema` gained the annotation-only `init_model_opt_sched_fn` field
(absent from the stored schema node, so composed configs are unchanged).
Annotations and casts only — zero runtime change (tiny_local final_ce
unchanged). main.py's one remaining Pylance error (`"parameters" is
not a known attribute of "None"`) predates this work.

**Gates (re-run after the review fixes):** full pytest green — 188 tests, 16
new in tests/test_typed_config.py (schema↔Trainer sync, negative compose
tests, grid-point checks, range checks, top-level config composition pins,
registry idempotence, the three review pins); `tiny_local` CPU smoke through
`check_point` trained 200 steps; the three negative overrides and the range
violation each fail with readable errors; ruff F401/F811/F841 clean over
the gate's scope (`src main.py run_exp.py conftest.py update_pixi.py
scripts`), definitions.py included. That is the scope the cleanup goal
specifies, not the whole repo — 6 findings remain repo-wide (3 in
`grid_generator/generate_configs.py`, 3 across two test files), all in
files neither commit touched, down from 22 at 6874efe7.

## Follow-up (2026-11-03): second review round

A second independent review of the committed range raised three headline
findings. Verified against both commits before acting:

- **`mfu_audit` configs and `- base_schema`** — mechanism real, severity not.
  Applied `- /base_schema` to all five (see the implementation note above).
  The claim that the configs "no longer compose" does not hold: the only
  affected invocation never worked, and the proposed fix does not make it
  work either — with `- /base_schema` it still composes to a nested
  `mfu_audit:` tree and dies on resolve with `InterpolationKeyError`.
- **"ruff zero everywhere"** — fair; the claim was scoped-gate-clean, not
  repo-clean. Corrected in the gates paragraph above.
- **"final_ce figure wrong"** — does not reproduce. This box yields
  8.451301097869873 on 3/3 repeats, before and after. The figure is
  environment-dependent, so it is now labelled as such rather than changed.

The review also flagged stale `+trainer.X=` repro commands in the MFU notes,
and the flag was correct — `+trainer.autocast_bf16=true` does error with
`An item is already at 'trainer.autocast_bf16'`. That has since been resolved
upstream rather than here: `b9189d1c` merged those five notes into
[training_efficiency.md](training_efficiency.md), which uses the plain
spelling throughout and spells out that `compile_model` keeps its `+` as a
top-level key. Nothing left to fix; the only `+trainer.X=` strings remaining
in the repo are the deliberate `+trainer.bogus=1` negative-test examples and
dated entries in [pretraining_reset_log.md](pretraining_reset_log.md), which
record what was run at the time.

Tests hardened: `TrainerSchema` defaults are now compared against `Trainer`'s
attrs defaults (names alone could not catch a knob defaulting differently on
each side — mutation-checked by flipping `compile_loss`), and the
caret-sweep rule is pinned (a `^key` under a schema'd section is rejected,
which is why a sweep config omits `- base_schema`).

**Gates:** 190 passed, 0 failed (188 + the 2 new tests); ruff clean over the
gate scope; `tiny_local` smoke final_ce 8.451301097869873, unchanged; all
five `mfu_audit` configs compose byte-identically to before the change under
the repo's convention. Note for anyone verifying in a fresh git worktree:
`data/` and `data_eval/` are gitignored, so 21 dataset-dependent tests skip
there unless those directories are linked in — symlink them, or the suite
looks green while skipping the training smokes.
