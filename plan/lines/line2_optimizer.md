# Line 2: optimizer switches

## Human

New optimizer options, each off by default. Now: task 3/4, Muon, not started.

- [x] Param groups: no weight decay on gains and embeddings
- [x] muP for AdamW, kept as simpleP (muP's LR rule only): best LR stays put across widths
- [ ] Muon: Muon for matrices, AdamW for the rest
- [ ] muP for Muon: best LR stays put with Muon

**Needs Janek:**
- The h12 param-groups run is written and ready to launch.
- The MoE width sweep is written and ready to launch: the same widths, no muP vs simpleP, 5 LRs per width around the dense best LRs. 50 short runs, about 6–8 GPU-hours.
- The h12 simpleP run is written and ready to launch: about 4 GPU-hours, at the tuned LR shifted so the big layers train at exactly the baseline's LR.
- Not line 2: 11 model-size tests still expect the old FFN width (4× instead of 3×), so they fail without this change too. Left alone.

**Last change:** full muP dropped, simpleP kept. In the width sweep both kept the best LR at the same value from the smallest width to the largest (without them it fell 16×), and simpleP is simpler: it only scales each layer's LR with width. With it off, losses are unchanged.

## AI

Plan for line 2 of the pretraining switches (`plan/today.md`): param groups → muP for
AdamW → Muon → muP for Muon. Each task builds on the one before. The per-switch survey
is `documents/notes/pretraining_switches.md`.

### How to use this file

You are a line-2 task agent. The Human section above is Janek's 30-second view, so keep
it short and in plain words: no file paths, no sub-steps.

1. Find your task in the status table.
2. Read Shared decisions, then the section and the "What landed" entry of every task
   before yours: they say what the code looks like now, which may differ from this plan.
3. When you finish, in the same diff as your code:
   - set your row to `done`, and fill "What landed" (files, config keys, group roles,
     anything that departs from the plan)
   - correct your section where the plan turned out wrong
   - update the Human section: tick your task, say where the line is now, set "Needs
     Janek" (one line per open decision or blocker, or "nothing") and "Last change" (one
     line)
4. Fix small or obvious problems yourself and record each in your row or section. That
   covers a wrong detail or line number, a missing step, the order of steps within your
   task, and a choice between equivalent implementations.
5. Stop and ask Janek only for:
   - a blocker
   - a change to a shared decision (parameter roles, the root `optimizer` config surface)
   - anything that touches another line's code or contract
   - any behaviour change with the switch off
   - a new dependency
   - a change to what a switch means
   - cluster cost beyond the inherited h12 config
   - re-ordering or dropping tasks

   Lead with the decision you need, then the options, with your recommendation first.
   Also add the open question to "Needs Janek" in the Human section.
6. Your final report starts with the decisions Janek needs to know about, then what's in
   the diff, with a reading order.

### Status

| # | Task | Status | What landed |
| --- | --- | --- | --- |
| 1 | Param groups | done | `src/core/optimizers.py`: `param_roles` and `build_optimizer_and_scheduler`, called once from `main.get_model_optimizer_scheduler`. Groups in `ROLES` order (vector, embedding, head, hidden), each with `role`. Options `configs/_trainer/_optimizer/{adamw,adamw_groups}.yaml` fill root `optimizer.roles`: null is one group; otherwise `optimizer.roles.<role>.weight_decay` for all four roles. Typed as `OptimizerSchema`. Cosine takes several groups if they share one LR. `only_weights` now loads the model only, as Task 1 allowed (see "As landed" there). Configs `tiny_local_param_groups`, `jm/h12_param_groups`. Cluster: not run yet. |
| 2 | muP for AdamW | done | simpleP only (`documents/notes/simplep_over_mup.md`): option `configs/_trainer/_optimizer/adamw_simplep.yaml`, hidden and head LR ÷ m, m = `common.dmodel / optimizer.base_dmodel` (base 256, h4). One new key, `optimizer.base_dmodel`: null in `adamw`/`adamw_groups`, 256 in `adamw_simplep` (which is `adamw_groups` plus that key). When it is set, the builder multiplies the hidden and head groups' `lr` by `base_dmodel / dmodel`. Cosine still rejects distinct group LRs; `train/lr` logs the base LR; `trainer.py` and the model config are untouched. Full muP (`adamw_mup`) was built, swept and dropped (`455421d2`). Configs in `configs/jm/feature/`: `tiny_local_simplep`, `h12_simplep` (one LR, 2^-8.415), `mup_width_{sp,simplep}` (dense), `mup_width_moe_{sp,simplep}`. Script: `scripts/mup_coord_check.py`; plots: `jupyter_notebooks/mup_lr_plots.ipynb`. Cluster: dense width sweep done, 122 runs with the dropped muP arm; MoE sweep and `h12_simplep` not yet. |
| 3 | Muon | todo | |
| 4 | muP for Muon | todo | |

### Today's code (at `a3f4efdb`, with task 2)

- `main.py:209` calls `build_optimizer_and_scheduler` (`src/core/optimizers.py`) once.
  `optimizer.roles: null` (`adamw`, the default) is one AdamW group over
  `model.parameters()`; otherwise one group per non-empty role, each with its own `lr`
  (LR, or for hidden and head LR × `base_dmodel / dmodel` under simpleP) and
  `weight_decay`. `only_weights` loads the model only.
- `get_cosine_scheduler_with_warmup` asserts that all groups share one LR
  (`src/core/schedulers.py:83`), because its `eta_min` is absolute, taken from group 0.
  `tiny_local` uses cosine (`_trainer/llama.yaml`); `_trainer/nano.yaml` and
  `h12_chinchilla` use WSD.
- `WSDScheduler` chains `LinearLR`/`ConstantLR`, which scale each group's own LR, so it
  handles several groups, including the decay to `final_lr_fraction` (line 7). Its
  `load_state_dict` replays steps.
- `train/lr` logs `scheduler.get_last_lr()[0]`, i.e. group 0 (`src/core/trainer.py:813`).
- Gradient clipping is global over `model.parameters()` and doesn't see groups.
- The H family has dff = 3·dmodel and QK-norm on (`common.attention.qk_norm: true`).
- The optimizer is built after `compile` (in place, names unchanged) and `fully_shard`
  (DTensor params with global `ndim`; module classes are subclassed, so `isinstance`
  still holds).

### Shared decisions

#### Parameter roles

Every parameter gets exactly one role, from its shape and owning module, never from its
name. The first matching rule wins:

| Role | Rule | In today's model |
| --- | --- | --- |
| `vector` | `p.ndim < 2` | RMSNorm and QK-norm gains: 2 + 2 per block, plus the head norm |
| `embedding` | owned by an `nn.Embedding` | `embedding.weight` |
| `head` | inside a `TransformerHead` | `head.linear.weight` |
| `hidden` | everything else | q/k/v/o_proj, ff_pre_act, gate, ff_post_act |

The h4 `tiny_local` model has 47 parameters: 28 hidden, 17 vector, 1 embedding, 1 head.

This is the contract with the other lines: they add ordinary `nn.Parameter`s and never
edit the optimizer.
- Line 1: QK-norm gains, per-head gate scalars and nGPT's step sizes and logit scale are
  `vector`; the output gate's projection and the indexer's projections are `hidden`.
- Line 3: the router and expert weights are `hidden`, whether experts are 2D linears or
  one 3D tensor.
- Line 6: `Float8Linear` subclasses `nn.Linear`, so roles don't change.
- If a parameter fits no rule (say, small n×n residual-mixing matrices that shouldn't be
  `hidden`), whoever adds it adds a rule and a row here. Rules key on module type or shape.

#### Where the optimizer is built

- New `src/core/optimizers.py`, owned by line 2: the role classifier and one builder for
  the optimizer and scheduler. Both `main.py` sites call it, and nothing else constructs
  an optimizer.
- Always one optimizer object. The trainer, scheduler and checkpointing hold a single
  `optimizer`, so Muon goes in as one class holding Muon groups and AdamW groups.
- In grouped modes, each param group carries its role name (`group["role"]`).

#### Config surface

- A root key `optimizer`, filled from a new group `configs/_trainer/_optimizer/`, the way
  `_trainer/_scheduler` fills `trainer.scheduler`. `_trainer/llama.yaml` and
  `_trainer/nano.yaml` both default to `adamw`, which is today's optimizer.
- Root, not `trainer.optimizer`, because every `cfg.trainer` key is passed to `Trainer`'s
  constructor (the comment at `main.py:160`).
- `trainer.learning_rate` and `trainer.weight_decay` stay where they are, as base values.
  The `optimizer` node holds the choice and per-role settings under
  `optimizer.roles.<role>`.
- Each task's switch is a new option in the group, chosen in a config's defaults list
  with an `override` entry placed after the group entries:
  - `- override _trainer/_optimizer@optimizer: <option>` in `configs/`, as
    `tiny_local_wsd.yaml` does for the scheduler
  - `- override ../_trainer/_optimizer@optimizer: <option>` in `configs/jm/`, as
    `h16_chinchilla_wsd.yaml` does
  - `- override ../../_trainer/_optimizer@optimizer: <option>` in `configs/jm/feature/`, where
    feature test configs go
  - on the CLI, `_trainer/_optimizer@optimizer=<option>`, only for configs that don't
    reach `_trainer` through `..`
- Schema typing in `src/definitions.py` (decided in task 1): one `OptimizerSchema` (`roles`:
  null, or all four roles, each `OptimizerRoleSchema`) in both `RootSchema` and
  `_base_schema_node`, so `optimizer` is strict at compose like `common` and `trainer`.
  `Trainer`/`TrainerSchema` are untouched. A later option adds its keys as fields there;
  if options stop sharing keys, add a class key and type by it in `check_point`, as
  `SCHEDULER_SCHEMAS` does. Task 1 has no class key, because AdamW is the only class.
- Task 2 added `optimizer.base_dmodel` (every option sets it; null is off). The width
  rule is in the builder, not in config: no per-role LR keys. Options don't touch model
  keys: full muP's head-init key went with it.

#### Acceptance check (every task)

Two stages, in this order. Copy `data/` and `data_eval/` from the main checkout before
running anything. Every config for a feature test run, local or cluster, goes in
`configs/jm/feature/`. Include the parent with `@_global_` (`../../tiny_local@_global_`,
`../h12_chinchilla@_global_`); without it the parent lands in a relative package and
interpolations and overrides fail.

**Stage 1, local:**
1. **Off is unchanged.** Before editing, record the losses of
   `pixi run python main.py --config-path configs --config-name tiny_local infrastructure.metric_logger.type=stdout`
   on CPU. After your change, the same command gives identical `train/loss` lines. Use
   CPU, because GPU runs aren't bit-reproducible.
2. **On trains.** `configs/jm/feature/tiny_local_<switch>.yaml` holds the overrides
   (defaults: `../../tiny_local@_global_`, the `override` entry, `_self_`), and
   `main.py --config-path configs/jm/feature --config-name tiny_local_<switch>` trains with it
   on CPU.
3. **Tests.** `pixi run python -m pytest -m "not gpu and not slow"` passes. New unit tests
   go in `tests/test_optimizers.py`.
4. Your task's own local checks, listed in its section.

**Stage 2, cluster config** (only after stage 1 passes): write
`configs/jm/feature/h12_<switch>.yaml` on top of `h12_chinchilla`, following the pattern of
`h16_chinchilla_wsd.yaml` (`git show ba679a70:configs/jm/h16_chinchilla_wsd.yaml` if it
has since been deleted):
- The defaults list is `../h12_chinchilla@_global_`, then the
  `- override ../../_trainer/_optimizer@optimizer: <option>` entry, then `_self_`.
- The body sets `infrastructure.metric_logger.name` and tags, and nothing else unless
  the switch needs it:
  - FLOPs-matched: if the switch changes FLOPs per token (`src/core/mfu.py`), set
    `trainer.n_steps` so total FLOPs match h12_chinchilla's 50_000 steps × 64 × 1024
    tokens, a multiple of 10, with the arithmetic in a one-line comment.
  - One LR, no grid: h12_chinchilla's tuned 2^-10 (`learning_rate: 10`) is inherited.
    Change it only for a known LR effect of the switch (muP's base-width shift, an
    update scale with its own LR range), with the reason in a one-line comment.
- Everything else is inherited, so the run compares directly with the h12_chinchilla
  baseline.

Janek launches it with
`run_exp.py --config-path configs/jm/feature --config-name h12_<switch>`; agents don't submit
cluster jobs. A task lands when stage 1 passes and its h12 config is written. Cluster
results are added to its row when they come in. Task-specific cluster runs, such as the
muP width sweeps, come on top of the h12 config, not instead of it.

### Task 1: param groups

Scope:
- `src/core/optimizers.py`: the role classifier and the builder. `main.py`: both AdamW
  sites call the builder. Whether the `only_weights` path stops building twice is your
  call, as long as behaviour stays the same.
- `_trainer/_optimizer/adamw.yaml`: today's single group over `model.parameters()`. Keep
  `fused=torch.cuda.is_available()`.
- `_trainer/_optimizer/adamw_groups.yaml`: one AdamW group per non-empty role, each with
  its own `weight_decay`: `${trainer.weight_decay}` on `hidden` and `head`, 0 on `vector`
  and `embedding`.
- `src/core/schedulers.py`: the cosine assert accepts several groups that share one LR.
  Groups with different LRs stay rejected until task 2.
- `configs/tiny_local_param_groups.yaml`.

Checks:
- `adamw_groups` with every role's `weight_decay` set to `trainer.weight_decay` gives
  losses identical to `adamw` on CPU. Grouping by itself must change nothing: CPU AdamW
  updates each tensor independently.
- Unit tests: every parameter of the h4 model lands in exactly one role, matching the
  table above; empty roles make no group.
- Save and resume with `adamw_groups`: non-sharded on CPU, and sharded through the gloo
  FSDP2 harness in `tests/test_distributed.py`.
- A checkpoint saved with `adamw`, loaded with `only_weights: true` into an
  `adamw_groups` run, trains. Today that path loads the old optimizer state before
  rebuilding, which may fail on the group mismatch. If it does, make `only_weights` skip
  optimizer and scheduler state.
- Stage 2: `configs/jm/h12_param_groups.yaml`.

As landed:
- The mismatch did fail on both paths: non-sharded with "loaded state dict has a different
  number of parameter groups", sharded with "Missing key in checkpoint state_dict:
  app.optim.param_groups.0.role". So `load_checkpoint_from_file` now skips optimizer and
  scheduler state under `only_weights` (sharded through a new `ModelState`), and
  `main.py` builds the optimizer once. The trained state is the same as before, because
  the loaded optimizer was thrown away.
- `fused=torch.cuda.is_available()` stays in the builder; `adamw.yaml` is just `roles: null`.
- Stage 1 results: switch off, all 202 `train/loss` lines byte-identical to the baseline.
  `adamw_groups` with every role's decay at 0.1 is identical too. Save/resume and the
  `only_weights` load are covered by `tests/test_optimizers.py` (CPU and gloo FSDP2) and by
  `main.py` runs on CPU.
- Pre-existing, left alone: `test_every_runnable_config_composes_and_resolves` fails at
  `581e2f73`, because `configs/jm/old/branch_testing/*.yaml` (moved in `eea91953`) can't
  find `../_misc/default`. It is the only failure in the suite, before and after.

### Task 2: muP for AdamW

Scope:
- Per-role LR multipliers and head init as functions of the width multiplier
  `m = common.dmodel / <base dmodel>` (you name the key). Follow the Adam column of Table 3
  in Tensor Programs V (Yang et al., arXiv 2203.03466). Write the rules you chose here as
  a table: role, init, LR, weight decay.
- Prefer init through config (the head's `partial_init_fn.scale`, where std =
  scale/√fan_in) plus per-group LR over a forward multiplier, so `src/core/model.py` stays
  untouched; line 1 edits it. Under Adam, a forward multiplier α equals init × α and
  LR × α, up to ε.
- No attention change. In the H family dhead = 64 at every H (width grows by adding heads),
  so muP's 1/d attention scale is constant.
- Scheduler: once group LRs differ, cosine's absolute `eta_min` is right only for group 0.
  Either make cosine per-group with the single-group path unchanged, or reject cosine
  under muP. WSD already works.
- `train/lr`: decide what it logs once groups differ (e.g. the base LR). The line is in
  `src/core/trainer.py`, which line 3 also edits, so keep that hunk small.
- Weight decay: torch AdamW decays by lr·wd, so LR × 1/m also weakens decay at width.
  Decide whether to keep lr·wd fixed, and record it.
- `origin/ctx_scl-simpleP` holds simpleP: muP's LR rule only (`configs/_parameterization/`,
  `build_simpleP_param_groups` in its `main.py`). Task 2's `adamw_simplep` is the same
  rule, except that ctx_scl also decays gains and embedding.

Checks:
- Coordinate check on CPU: h = 2, 4, 8, 16 at fixed depth (pin `common.n_blocks`, since
  the H family scales depth with H). Run a few steps and log activation and update RMS per
  role: under SP they grow with width, under muP they stay flat. Add a script under
  `scripts/`, and the resulting table or plot here.
- Stage 2: `configs/jm/feature/h12_mup.yaml`. Pick a base width below h12 (a small proxy). If
  h12 is the base, every multiplier is 1 and the run differs from SP only in the init and
  decay rules.
- Width sweep, on top of stage 2: an LR sweep at three or more widths, fixed depth, SP vs
  muP, with short WSD runs. The muP optimum should stay put across widths. Write the
  configs under `configs/jm/feature/`; Janek launches them.

As landed (simpleP; why full muP was dropped: `documents/notes/simplep_over_mup.md`):
- Decision (Janek, 2026-11-25): one option, `adamw_simplep`, muP's LR rule only. Full muP
  (`adamw_mup`: also decay × m and head init × m^-1/2) did no better in the width sweep
  and needed the optimizer to set the model's head init. Its code is in `455421d2`.
- `adamw_simplep` is `adamw_groups` (its defaults list) plus `base_dmodel: 256`; the
  builder divides the hidden and head LR by m = `common.dmodel / optimizer.base_dmodel`.
  At h4 (the base) the option equals `adamw_groups`. LR is relative to
  `trainer.learning_rate`:

  | Role | Init | LR | Weight decay |
  | --- | --- | --- | --- |
  | `vector` | unchanged | × 1 | 0 |
  | `embedding` | unchanged | × 1 | 0 |
  | `head` | unchanged | × 1/m | `trainer.weight_decay` |
  | `hidden` | unchanged | × 1/m | `trainer.weight_decay` |

  So lr·wd on hidden and head falls as 1/m. MoE's experts and router are `hidden`, so
  each expert gets LR ÷ m like the dense FFN (tested).
- Scheduler: cosine is rejected off base width by task 1's one-LR assert; the cosine code
  is untouched, since line 7 owns it. WSD scales each group's own LR, including line 7's
  `final_lr_fraction`. `jm/feature/tiny_local_simplep` is `tiny_local` with a WSD override
  (`tiny_local_wsd`'s 10/50 steps).
- `train/lr` logs group 0, `vector`, which keeps the base LR.
- `tiny_local_simplep` sets `optimizer.base_dmodel: 128` so m = 2 at h4: at the default
  base every h4 multiplier would be 1.
- `jm/feature/h12_simplep` (m = 3) is one run: 50_000 steps, since simpleP changes no
  FLOPs, and `learning_rate: 8.415` = 10 − log2 3. The LR is the base model's and h12's
  matrices get it / 3, so they train at 2^-10 with the baseline's lr·wd; it differs from
  `h12_param_groups` only in 3× LR on gains and embedding. About 3.7 GPU-h:
  h12_chinchilla's tuned run (W&B `kiu7rawb`) took 1.85 h on 2 H100, evals and
  checkpoints included.
- Stage 1, run at `46a5eb89`: with the switch off, `tiny_local` and
  `tiny_local_param_groups` give 202 `train/loss` lines, byte-identical to the pre-change
  baselines. `tiny_local_simplep` trains, loss 10.88 → 8.19 in 200 steps. Tests: group
  LRs and decay at h4/h8, MoE experts and router in `hidden`, save/resume with distinct
  group LRs. The suite's only failures are 11 tests in `test_h_balanced.py` and
  `test_mfu.py` that still expect dff = 4·dmodel (as on `pretraining-jm`). Merged with
  `pretraining-jm` after the MoE merge (`a3f4efdb`) and re-tested.
- Coordinate check, `scripts/mup_coord_check.py` defaults: 2 blocks, constant LR 2^-7,
  one random batch of 4×32, 4 steps. Cell: RMS of the change since step 0 in the role's
  module outputs, mean over modules; `vector` includes the QK-norm gains. Under SP hidden
  and head grow 4.8× and 2.7× from h2 to h16. simpleP keeps hidden flat; its head falls
  from h2 to h8 (1.84 → 1.07), then stays flat. Full muP (dropped) was flat too:

  | role | arm | h2 | h4 | h8 | h16 |
  | --- | --- | --- | --- | --- | --- |
  | hidden | SP | 0.62 | 0.87 | 1.25 | 2.99 |
  | hidden | simpleP | 0.93 | 0.87 | 0.83 | 0.78 |
  | hidden | muP (dropped) | 0.96 | 0.87 | 0.84 | 0.82 |
  | head | SP | 1.12 | 1.33 | 1.25 | 3.06 |
  | head | simpleP | 1.84 | 1.33 | 1.07 | 1.07 |
  | head | muP (dropped) | 1.60 | 1.33 | 1.30 | 1.25 |
  | vector | SP | 0.62 | 0.77 | 0.92 | 1.05 |
  | vector | simpleP | 0.76 | 0.77 | 0.77 | 0.74 |
  | vector | muP (dropped) | 0.79 | 0.77 | 0.76 | 0.76 |
  | embedding | all | 0.020 | 0.020 | 0.020 | 0.020 |

- Width sweep, one standalone config per arm in `configs/jm/feature/` (h12_chinchilla's
  groups in their own defaults list): `mup_width_sp` (`adamw_groups`, 40 runs) and
  `mup_width_simplep` (`adamw_simplep`, base h4, 41 runs). A root `^sweep` list pairs
  widths with their own LR exponents, and `common.h: ${sweep.h}` /
  `trainer.learning_rate: ${sweep.lr}` read the pair; W&B logs the resolved values. Both
  grids are h2–h12 × 2^-3..2^-11, plus h16 at 2^-7..2^-10 (SP) or 2^-4..2^-8 (simpleP):
  exactly the runs in W&B, which first ran as three configs per arm. Runs are 2000 steps
  at h12_chinchilla's batch (64 × 1024, 131M tokens), WSD 100/400, at `n_blocks: 4` on 1
  H100 with micro 64; they took 5.2 (h2) to 7.3 (h16) min each. No checkpoints or
  downstream eval; periodic eval and `final_ce` stay. The full-muP arm's configs were
  dropped; its 41 runs stay in W&B under the tag `mup`. Plot with
  `jupyter_notebooks/mup_lr_plots.ipynb` (`ff = "dense"` or `"moe"`; grids side by side,
  a star at each width's best LR, and a best-LR table).
- MoE width sweep (Janek, 2026-11-25; not run yet): `mup_width_moe_sp` and
  `mup_width_moe_simplep`, the dense configs plus the MoE feed-forward (8 experts, top-2,
  `configs/_ff_layer/moe.yaml`), tagged `moe`. Same widths, steps and batch. Assuming
  MoE's best LRs are near dense's, each width gets 5 LRs centred on its dense optimum:
  SP 2^-3..2^-7 at h2 sliding to 2^-7..2^-11 at h16, simpleP 2^-4..2^-8 at every width.
  25 runs per arm. Cost: MoE steps are ~1.5× dense, so about 6–8 GPU-h for both arms.
- Results (2026-11-25, 122 runs). `final_ce` at the best LR, with a neighbour where the
  gap is under the ~0.016 bf16 resolution:

  | width | SP best | simpleP best | muP (dropped) best |
  | --- | --- | --- | --- |
  | h2 | 2^-5 4.477 (2^-6 4.493) | 2^-6 4.504 | 2^-6 4.474 |
  | h4 | 2^-6 4.205 | 2^-6 4.211 | 2^-6 4.201 |
  | h8 | 2^-7 4.011 | 2^-6 3.998 | 2^-6 3.985 |
  | h12 | 2^-8 3.921 | 2^-6 3.892 | 2^-6 3.909 |
  | h16 | 2^-9 3.868 (2^-8 3.873) | 2^-6 3.838 (2^-7 3.839) | 2^-7 3.857 (2^-6 3.871) |

  SP's best LR falls about one power of 2 per width step (2^-5 → 2^-9); simpleP's and
  muP's stay at 2^-6, with a real interior optimum (2^-5 is worse). From h8 up simpleP's
  loss at its best LR is equal to or below muP's. Above 2^-5 runs get unstable.

### Task 3: Muon

Scope:
- Muon on `hidden`; AdamW on `embedding`, `head` and `vector`. Both live in one optimizer
  class, e.g. the `MuonWithAuxAdam` pattern in the reference implementation
  (github.com/KellerJordan/Muon).
- torch 2.7.1 has no `torch.optim.Muon`. By default, write your own in
  `src/core/optimizers.py`: Nesterov momentum, quintic Newton–Schulz in bf16, and an
  update scale. A dependency is worth it only if it brings a working FSDP2 path, and it
  needs Janek's OK first. If it gets the OK, follow the memory note on adding a dep to
  the v6 pixi lock.
- FSDP2: gradients and momentum are `Shard(0)` DTensors. Momentum is elementwise and can
  stay sharded, but Newton–Schulz needs whole matrices. Start simple: gather
  (`full_tensor()`), orthogonalize on every rank, keep the local slice. Measure that. If it
  is too slow, assign matrices to ranks round-robin and scatter the results (as in
  Moonlight's distributed Muon or Microsoft's Dion).
- Update scale: pick one and record it, because task 4's LR rules depend on it. The
  options are Keller Jordan's `max(1, rows/cols)^0.5` (its own LR scale, around 0.02) or
  Moonlight's RMS match, `0.2·√max(rows, cols)` ("Muon is Scalable for LLM Training",
  arXiv 2502.16982), which lets Muon groups reuse AdamW's LR and weight decay.
- 3D `hidden` params (MoE experts from line 3, if they landed as one tensor):
  orthogonalize each matrix along dim 0, or reject them at build time.
- Checkpoints: DCP's `get_state_dict` initializes empty optimizer state by stepping with
  zero grads, and it shards state that is per-param tensors. Test save and resume.
- If line 6 (FP8) has landed, check Muon with float8 linears.
- MFU counts model FLOPs only, so compare optimizers by s/step.

Checks:
- `configs/jm/feature/tiny_local_muon.yaml` trains on CPU.
- Unit tests:
  - Newton–Schulz output has singular values near 1.
  - With every parameter in AdamW groups, the class matches `torch.optim.AdamW`.
  - The sharded step equals the unsharded one, in a 2-rank gloo CPU test built on the
    FSDP2 harness in `tests/test_distributed.py`.
- Stage 2: `configs/jm/feature/h12_muon.yaml`. h12_chinchilla runs on 2 GPUs, so this run
  exercises FSDP2 Muon. Compare it with h12_chinchilla by loss and s/step. If your update
  scale doesn't reuse AdamW's LR, the inherited 2^-10 is in the wrong range: set one LR
  in this config and say why in a one-line comment.

### Task 4: muP for Muon

Scope:
- Width rules for the Muon groups. The AdamW groups keep task 2's rules. In the H family
  every matrix's aspect ratio is fixed (dff = 3·dmodel, kv = dmodel/2), so shape factors
  are constants across widths. With Keller's scale the update's spectral norm is O(1),
  which meets the spectral condition for feature learning (Yang, Simon and Bernstein, "A
  Spectral Condition for Feature Learning"), so hidden LR should be width-independent.
  With Moonlight's RMS match the spectral norm grows as √width, so hidden LR should scale
  as 1/√width. Derive the rules from task 3's choice, verify them, and write the table
  here.
- The config surface is the same: a Muon + muP option in `_trainer/_optimizer/`.

Checks: as in task 2, a CPU coordinate check at fixed depth. Then stage 2,
`configs/jm/feature/h12_muon_mup.yaml` with a base width below h12, and on top of it a width sweep
with Muon.

### Out of scope for line 2

Attention and residual code in `src/core/model.py` and `src/core/attention.py`, and the
packer in `src/core/datasets.py` (line 1). The FFN slot and `src/core/moe.py` (line 3).
The float8 conversion in `main.py` (line 6), which sits in the same function as the
optimizer code, in a different hunk. `src/core/trainer.py` beyond the `train/lr` line.
