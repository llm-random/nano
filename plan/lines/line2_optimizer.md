# Line 2: optimizer switches

## Human

New optimizer options, each off by default. Now: task 4/4, muP for Muon, not started. Muon's h12 run is in: lower loss than the AdamW baseline at the same steps; after the speed work each step is 3% slower.

- [x] Param groups: no weight decay on gains and embeddings
- [x] muP for AdamW, kept as simpleP (muP's LR rule only): best LR stays put across widths
- [x] Muon: Muon for matrices, AdamW for the rest
- [ ] muP for Muon: best LR stays put with Muon
- [ ] Later, own branch: weight decay that scales with width (papers in the simpleP note)

**Needs Janek:**
- The MoE width sweep is written and ready to launch: the same widths, no muP vs simpleP, 5 LRs per width around the dense best LRs. 50 short runs, about 6–8 GPU-hours.
- After the merge: launch the h12 simpleP run together with the h12 param-groups run, its fair baseline (both configs are ready): about 4 GPU-hours each. simpleP uses the tuned LR shifted so the big layers train at exactly the baseline's LR.
- The h12 param-groups run is also Muon's fair baseline. Until it runs, we can't tell how much of Muon's gain is Muon and how much is the param groups.
- Speed is done: Muon is now 3% slower per step than AdamW (was 16%) and MuonClip's cap costs under 1% (was 14%). At equal training time Muon should now come out ahead, roughly 0.01 in final loss. A full h12 run with the new code would confirm it (about 4 GPU-hours); results per step won't change, only time.
- MuonClip stays as an option, off by default. At h12 it was not better: Kimi's cap was never reached (largest attention score 76, cap 100), so the run was just Muon without QK-norm, with slightly worse loss. It is now 3% faster per step than Muon with QK-norm (it skips QK-norm), not enough to make up the loss. Keep QK-norm on for now.
- Optional, later: Muon at 2× and ½ the LR, to check that reusing AdamW's LR is right. 2 runs, about 7.5 GPU-hours; config on request.

**Last change:** second round of profiles. Muon's step is 3% slower than AdamW's (was 16%). MuonClip's cap is nearly free (was 14%): it reads the attention kernel's own output, which matched the exact largest score within 0.2.

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
| 2 | muP for AdamW | done | simpleP only (`documents/notes/simplep_over_mup.md`): option `configs/_trainer/_optimizer/adamw_simplep.yaml`, hidden and head LR ÷ m, m = `common.dmodel / optimizer.base_dmodel` (base 256, h4). One new key, `optimizer.base_dmodel`: null in `adamw`/`adamw_groups`, 256 in `adamw_simplep` (which is `adamw_groups` plus that key). When it is set, the builder multiplies the hidden and head groups' `lr` by `base_dmodel / dmodel`. Cosine and WSD both decay each group to its own final fraction (cosine made per-group at Janek's request, bit-identical with one LR). `validate()` rejects a `base_dmodel` ≤ 0. Rank 0 logs each param group once at build; `train/lr` logs the base LR; the model config and `trainer.py` are untouched. Full muP (`adamw_mup`) was built, swept and dropped (`455421d2`). Configs in `configs/jm/feature/`: `tiny_local_simplep`, `h12_simplep` (one LR, 2^-8.415), `mup_width_{sp,simplep}` (dense), `mup_width_moe_{sp,simplep}`. Script: `scripts/mup_coord_check.py`; plots: `jupyter_notebooks/mup_lr_plots.ipynb`. Cluster: dense width sweep done, 122 runs with the dropped muP arm; MoE sweep and `h12_simplep` not yet. |
| 3 | Muon | done | `src/core/optimizers.py`: `MuonWithAdamW` (subclasses `torch.optim.AdamW`) and `newton_schulz`. Option `configs/_trainer/_optimizer/muon.yaml` = `adamw_groups` + `muon: {momentum: 0.95, ns_steps: 5}`. One new key, `optimizer.muon` (typed `MuonSchema`), null in `adamw`/`adamw_groups`, as `base_dmodel`: no class key. Groups as `adamw_groups` plus `use_muon` (true on `hidden` only), `momentum`, `ns_steps`. Moonlight's update scale, so every group keeps `trainer.learning_rate` and its role's decay. The builder asserts `base_dmodel` is null with Muon. MoE router and experts: Muon. FSDP2: each rank gathers and orthogonalizes every matrix, batched by shape. The group log names each group's optimizer. Configs `jm/feature/{tiny_local_muon,h12_muon}`. MuonClip (Janek's request): attention switch `common.attention.qk_clip` (null off), K2's QK-Clip by an optimizer post-step hook, needs `qk_norm: false`; configs `jm/feature/{tiny_local_muon_clip,h12_muon_clip}`. Touches line 1's attention files (see "As landed"). Cluster (2026-11-27, results under "As landed"): `h12_muon` (`e703v6py`) final_ce 3.0913 against h12_chinchilla's 3.1063, at +15.6% s/step. `h12_muon_clip` (`ci4m5vs9`) 3.0995; the clip never fired (max logit 75.5). `h12_param_groups` not run yet. Profiles (2026-11-28), after a per-shape rewrite of the Muon step: Muon +5.6% s/step (was +15.6%); QK-Clip's logit tracking +14%. Since then Newton–Schulz is split across ranks and QK-Clip on GPU tracks flash attention's logsumexp. Round-2 profiles: Muon +3.1% s/step, QK-Clip +0.6%. |
| 4 | muP for Muon | todo | |

### Today's code (at `a3f4efdb`, with task 2)

- `main.py:209` calls `build_optimizer_and_scheduler` (`src/core/optimizers.py`) once.
  `optimizer.roles: null` (`adamw`, the default) is one AdamW group over
  `model.parameters()`; otherwise one group per non-empty role, each with its own `lr`
  (LR, or for hidden and head LR × `base_dmodel / dmodel` under simpleP) and
  `weight_decay`. `only_weights` loads the model only.
- `get_cosine_scheduler_with_warmup` decays each group to `final_lr_fraction` of its own
  LR through `PerGroupCosineAnnealingLR` (`src/core/schedulers.py`): torch's cosine with
  one `eta_min` per group, bit-identical to it when groups share one LR. `tiny_local`
  uses cosine (`_trainer/llama.yaml`); `_trainer/nano.yaml` and `h12_chinchilla` use WSD.
- `WSDScheduler` chains `LinearLR`/`ConstantLR`, which scale each group's own LR, so it
  handles several groups, including the decay to `final_lr_fraction` (line 7). Its
  `load_state_dict` replays steps.
- `train/lr` logs `scheduler.get_last_lr()[0]`, i.e. group 0 (`src/core/trainer.py:813`).
  With param groups, the builder logs each group once (rank 0): role, LR, decay and its
  parameter names.
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
  rule, except that ctx_scl also decays gains and embedding and keeps the MoE router at
  the base LR (here the router is `hidden`, LR ÷ m).

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
- Scheduler (Janek, 2026-11-25: make cosine take simpleP): cosine's `eta_min` is now per
  group, `final_lr_fraction` × each group's own LR (`PerGroupCosineAnnealingLR`, which
  copies torch 2.7.1's `get_lr` and closed form with `eta_min` indexed by group). With
  one LR it is bit-identical to the stock scheduler (tested), so `tiny_local` is
  unchanged. WSD already scaled each group's own LR, including line 7's
  `final_lr_fraction`. `jm/feature/tiny_local_simplep` is plain `tiny_local` (cosine)
  plus the option. `validate()` rejects a `base_dmodel` ≤ 0.
- `train/lr` logs group 0, `vector`, which keeps the base LR; the other groups' LRs
  follow from the config. The builder logs each param group once at startup (rank 0):
  role, LR, decay, tensor and parameter counts, and every parameter name with its shape.
- Caveats (note): the sweep doesn't settle the decay rule (full muP decayed 2–4× harder
  from h8 up, which may explain simpleP's small loss edge); a width rule for decay is a
  later item, on its own branch. simpleP is width-only by design.
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

Since task 2 (2026-11-26), read before the scope:
- Config: `optimizer.roles.<role>` holds only `weight_decay`; `optimizer.base_dmodel`
  turns on simpleP, which the builder applies to the hidden and head LR. Add Muon as
  an option that extends `adamw_groups` (`defaults: [adamw_groups, _self_]`, as
  `adamw_simplep.yaml` does) plus its own keys; when options stop sharing keys, add
  the class key described under Config surface.
- simpleP's hidden LR ÷ m is AdamW's rule. Muon's update scale changes it (task 4), so
  in task 3 make the builder assert `base_dmodel` is null for Muon, or apply simpleP
  to the AdamW groups only. Don't silently divide a Muon LR by m.
- Schedulers: cosine and WSD both scale each group relative to its own LR (cosine
  per group since task 2), so Muon groups can have their own LR under either.
- MoE has landed (line 3, #201): each expert is a `GatedFeedForward` with 2D weights,
  and the router (`num_experts × dmodel`) is `hidden`, so by role both go to Muon.
  Decide whether the router belongs in Muon or AdamW and record it. No 3D expert
  tensors on `pretraining-jm` yet; line 3's dropless follow-up may add them.
- Torch is still 2.7.1. If `pretraining-jm` moves to a newer torch (line 3's dropless
  follow-up targets 2.11), check whether it ships `torch.optim.Muon` and how it works
  with FSDP2 before writing your own.
- The builder logs each param group once at startup (rank 0); add which optimizer
  each group uses. Check new configs with `run_exp.expand_and_check(cfg)`, the
  launcher's own pre-submit check (never call `run_exp.py` itself).

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
  exercises FSDP2 Muon. Compare it by loss and s/step with `jm/h12_param_groups` (same
  groups and decay, AdamW everywhere: the like-for-like baseline) and h12_chinchilla.
  If your update scale doesn't reuse AdamW's LR, the inherited 2^-10 is in the wrong
  range: set one LR in this config and say why in a one-line comment.

As landed:
- Update scale: Moonlight's RMS match. A Muon step is
  `p ← p·(1 − lr·wd) − lr·0.2·√max(rows, cols)·NS(Nesterov momentum)`, with the group's
  own LR in the decay, before the update, as in Moonlight's `toy_train.py` and torch's
  AdamW. Its update RMS is about 0.2·lr, AdamW's size. So the Muon group takes
  `trainer.learning_rate` and `trainer.weight_decay` as they are. There is no new LR key,
  `h12_muon` inherits the tuned 2^-10, and both schedulers work unchanged. For task 4 this
  means the update's spectral norm grows as √width at the H family's fixed aspect ratios,
  so the expected rule is Muon LR × m^-1/2. Under Keller's scale it would be
  width-independent, but it needs its own LR (~0.02) and a second LR key.
- MoE router: Muon. It stays `hidden`, with no new role rule. It is a hidden matrix: its
  input is the residual stream, and its output feeds the routing softmax, not the network's
  output. Keller's rule is Muon for hidden matrices. Moonlight's example filter (every ≥2D
  parameter but embedding and head) would also put it in Muon, although that example
  model is dense. If MoE Muon runs show load-balance trouble, move the router to AdamW
  with a rule keyed on `MoE`.
- FSDP2, first version as planned: each rank all-gathers every whole Nesterov update
  (bf16 on GPU, ~85 MB per rank per step at h12) and orthogonalizes all of them, one
  batched call per shape (4 at h12). Then it keeps its own rows (`torch.chunk`, as
  `Shard(0)` splits). The 2-rank test shows this is exact. Estimated cost at h12: Newton–Schulz is
  1.54e12 FLOPs per rank per step, 5.5% of a GPU's model FLOPs per step, so expect a
  few percent on s/step. Measure it with `h12_muon` against `h12_param_groups`. Go
  round-robin (halves Newton–Schulz at 2 GPUs) only if the gap is large;
  `trainer.profile_timers=true` logs each step's `optim` ms per rank.
- simpleP: the builder asserts `optimizer.base_dmodel` is null with Muon. simpleP was not
  applied to the AdamW groups only; task 4 sets the Muon rule and the head's together.
- Implementation:
  - AdamW groups run torch 2.7.1's own `Adam.step` body per group (`_init_group` and the
    functional `adam`). They are bit-identical to `torch.optim.AdamW` (tested), and fused
    on CUDA.
  - Muon groups: Keller's lerp form of Nesterov momentum (β 0.95, state
    `momentum_buffer`, sharded like the param; Moonlight's `mul_/add_` form differs by a
    constant factor, which Newton–Schulz removes), then quintic Newton–Schulz (Keller's
    coefficients, 5 steps).
  - Newton–Schulz runs in bf16 on GPU but fp32 on CPU: on CPU, bf16 matmuls were ~40×
    slower, and `tiny_local_muon` would have taken ~25 min instead of ~1.5.
  - `__setstate__` skips Adam's fix-up for old checkpoints. That fix-up reads
    `state["step"]` for every param, so it would fail on Muon's momentum-only state in every
    `load_state_dict`.
  - `add_param_group` rejects Muon params that aren't 2D (no 3D experts on this branch)
    and DTensor placements other than `Shard(0)`.
  - torch is still 2.7.1, with no `torch.optim.Muon`. FP8 (line 6) hasn't landed: not
    checked.
- Stage 1 (CPU):
  - With the switch off, `tiny_local` and `tiny_local_param_groups` give 202 `train/loss`
    lines each, byte-identical to the pre-change baselines.
  - `tiny_local_muon` trains: loss 10.88 → 8.17 (mean of the last 100 steps; a smoke run,
    not a comparison).
  - 14 new tests in `tests/test_optimizers.py`:
    - Newton–Schulz singular values in (0.6, 1.2) and cosine > 0.98 to UVᵀ, in bf16 and
      fp32.
    - The option's groups; MoE experts and router in Muon; simpleP rejected.
    - All-AdamW groups equal `torch.optim.AdamW` bit for bit.
    - One Muon step equals Moonlight's update, with RMS ≈ 0.2·lr.
    - Save and resume, non-sharded and DCP (1 rank).
    - A 2-rank gloo run (`mp.spawn`): three sharded steps equal the unsharded ones bit
      for bit, including a 3-row matrix (2 + 1 rows) and a 1-row one (1 + 0). A 2-rank DCP
      save and a resume into a fresh model then step the same.
  - Suite: 255 passed, 11 failed, all 11 the known dff = 4·dmodel tests. The hygiene
    tests pass with the untracked configs included.
- Stage 2: `jm/feature/h12_muon`, h12_chinchilla plus `muon`: 50_000 steps, 2^-10
  inherited. It passes `run_exp.expand_and_check`, about 3.7 GPU-h. Compare it by loss
  and s/step with `h12_param_groups` (like-for-like) and h12_chinchilla (`kiu7rawb`).
- Optional LR sweep, not written: 2^-9 and 2^-11, after `h12_muon`, about 7.5 GPU-h.
- MuonClip, Kimi K2's QK-Clip on top of Muon (arXiv 2507.20534). Added at Janek's
  request (2026-11-26): "just add the code, don't coordinate with line 1, we will address
  conflicts if they occur".
  - Switch: `common.attention.qk_clip`, a threshold τ on each head's max logit, null for
    off. It touches line 1's files: `SWITCHES` and `_rope_attention` in
    `src/core/attention.py`, `RoPEAttention` in `src/core/model.py`, the null default in
    `configs/_model/h_balanced/base.yaml`, `AttentionSchema`, and `SWITCHES_OFF` in
    `tests/test_attention_registry.py`. It also adds one call in `main.py`. It works with
    any optimizer; MuonClip is the `muon` option plus `qk_clip`.
  - Tracking: with τ set, `RoPEAttention` keeps `max_logits` (one per query head), a
    non-persistent buffer, so checkpoint keys don't change. It holds the largest causal
    logit at SDPA's scale over training forwards since the last clip; eval forwards are not
    tracked. SDPA never materializes the logits, so this is one extra no_grad `QKᵀ` per
    layer. It survives whole-block compile: in a fullgraph compile on CPU, compiled equals
    eager.
  - Clip: `qk_clip_` in `src/core/optimizers.py` runs in an optimizer post-step hook,
    `add_qk_clip`. `main.py` registers it after the checkpoint load, so DCP's state-init
    step doesn't clip. Max logits are all-reduced (MAX) over ranks. Each query head gets
    γ = τ / max(s, τ). Under GQA, the kv head's W_k rows take √(smallest γ in its group)
    and W_q's rows the rest, so each head's logits scale by exactly its own γ; with one
    query head per kv head this is K2's √γ on each side. The row scaling is FSDP2-aware
    (`_own_rows`). Every 100 steps it logs `train/max_attention_logit`, the max over those
    steps, layers and heads before the clip. The first version logged every step, which
    forced a sync; see Speed below.
  - QK-norm must be off (`RoPEAttention` rejects both on). Its RMSNorm undoes any scaling of
    W_q and W_k, and its gains are shared across heads. K2 (MLA) has no QK-norm either.
  - Stage 1, both switches off: `tiny_local` and `tiny_local_param_groups` are
    byte-identical to the baselines again, and `tiny_local_muon` gives the same lines as
    before the QK-Clip change. A tracking-only run (τ 1e9) peaked at a max logit of 5.4
    in 200 steps, so `tiny_local_muon_clip` uses τ = 4. It trains, and the clip acts on 18
    of 200 steps (pre-clip max 4.4, against 5.4 unclipped). 7 new tests:
    - tracking equals a direct causal-logit max, and eval forwards are not tracked;
    - after a clip every head's max logit is min(s, τ), with a kv pair split (one head
      clipped, its partner not);
    - QK-norm is rejected;
    - off and on build the same init and checkpoint keys;
    - the hook clips and logs once per step, and is absent when off;
    - 2-rank gloo, with a different max logit per rank: the sharded clip equals the
      unsharded one bit for bit.

    Suite: 265 passed, the same 11 known failures. The hygiene tests pass with the
    untracked configs included.
  - Stage 2: `jm/feature/h12_muon_clip`, which is `h12_muon` with `qk_norm: false` and K2's
    τ = 100: MuonClip against Muon with QK-norm, 50_000 steps, LR 2^-10. The extra `QKᵀ`
    and masked max are not model FLOPs in `mfu.py`, so MFU reads low. Estimated cost at
    micro 32: ~1.6 GB of memory traffic per layer per micro-batch, roughly 5–10% on s/step.
    Caveat: if h12's max logit never reaches 100, the run measures Muon without QK-norm. A
    control (`qk_norm: false`, no clip) is available on request.
- h12 results (2026-11-27). W&B runs: `e703v6py` (Muon) and `ci4m5vs9` (MuonClip); the
  baseline is h12_chinchilla `kiu7rawb`. `h12_param_groups` hasn't run yet. All three ran
  on lem with 2 H100. The train-loss columns are 100-step means; s/step is the median of
  the per-step time from step 1000, with eval pauses excluded.

  | run | final_ce | train loss, 5k / 45k / 50k | s/step |
  | --- | --- | --- | --- |
  | h12_chinchilla (AdamW, one group) | 3.1063 | 3.7069 / 3.2903 / 3.1036 | 100.9 ms |
  | h12_muon | 3.0913 | 3.6344 / 3.2681 / 3.0881 | 116.7 ms (+15.6%) |
  | h12_muon_clip (no QK-norm) | 3.0995 | 3.6498 / 3.2789 / 3.0956 | 128.1 ms (+27.0%) |

  - Muon against AdamW:
    - At step 1000 Muon is behind (+0.032, still in warmup). It is ahead from step 2000 on.
      The gap peaks near 5k (−0.073) and is −0.022 by 45k; at the end, final_ce is −0.015.
    - Not like-for-like: some of the gain may come from the param groups (no decay on
      gains or embedding). `h12_param_groups` would separate the two.
  - Speed:
    - Five same-config AdamW runs span 100.2–102.9 ms, so Muon's +15.6% is real. It is
      well above the few-% estimate, so the FSDP2 path needs work. A profile
      (`trainer.profile_timers=true`) should come before choosing a fix: round-robin
      Newton–Schulz, one flat gather, or fewer DTensor calls.
    - At equal wall-clock Muon gets about 43k steps. From Muon's stable-phase slope
      (~0.0027 per 1k steps) it would finish near 3.108, no better than AdamW's 3.106.
      This is a rough estimate.
  - MuonClip:
    - The clip never fired. The pre-clip max logit peaked at 75.5 at step 20k (median
      32–42 after step 2k), below K2's 100. So the run is Muon without QK-norm.
    - It is behind Muon at every checkpoint from 1k to 50k (+0.008 to +0.025 in 100-step
      train loss), final_ce +0.008. It is also +9.8% per step, from the logit tracking net
      of the QK-norm it drops.
    - Keep QK-norm and leave `qk_clip` off.
  - Grad norm and downstream: grad norm is lower under Muon (median 0.205 and 0.212
    against 0.250 over 40–45k), with no spikes (AdamW had 1). Downstream evals: Muon and
    AdamW are within 1.5 stderr on every task. MuonClip's one larger gap is sciq acc_norm,
    0.635 against Muon's 0.681 (about 3 stderr).
- Speed (2026-11-27, Janek: keep the clip, profile, make it faster). Both h12 runs above used
  the first version.
  - Diagnosis: the first version gathered each of h12's 84 matrices on its own: a DTensor
    wrap and an all-gather per matrix, plus ~8 small kernels each. That is ~650 launches and
    84 collectives per step, mostly host time. On CPU with h12's 84 matrices (tiny sizes,
    `$CLAUDE_JOB_DIR/tmp/muon_host_bench.py`, not in the repo), the old step took 53 ms of
    host time on 1 rank.
  - Rewrite of `_muon_step`, per shape (4 shapes at h12): one stack, one all-gather
    (`_gather_rows`, padded to equal shards, so uneven ones work) and one batched
    Newton–Schulz. Momentum, decay and the update are foreach calls. Newton–Schulz now
    takes a stack and folds its scalings into `baddbmm`.
  - Effect: host time per step 53 → 7.8 ms on 1 rank, 205 → 19.8 ms on 2 gloo ranks (gloo's
    per-collective cost inflates both). After 60 steps the weights are within 1.5e-7 of
    the old code's (`baddbmm` rounding). The tests are unchanged in substance; the 2-rank
    sharded = unsharded test, uneven shards included, still passes bit for bit.
  - QK-Clip hook: it read the max logit with `.item()` after every optimizer step, which
    made the host wait for the step to finish. Now the max stays on the GPU, and
    `train/max_attention_logit` is logged every 100 steps as the max over those steps.
  - Profiler ranges for the kernel profile: `muon:momentum`, `muon:gather`,
    `muon:newton_schulz`, `muon:update` and `qk_clip`.
  - Profiling configs: standalone, h12_chinchilla's model, data and batch, 300 steps,
    kernel profile at steps 12–13, `timers_ms` every step, no eval. `h12_profile_adamw_groups`
    is the reference. `h12_profile_muon` measures Muon. `h12_profile_muon_no_qk_norm` and
    `h12_profile_muon_clip` split QK-norm's cost from QK-Clip's. All pass
    `run_exp.expand_and_check`; about 10 GPU-min each.
  - Left for after the profile: round-robin Newton–Schulz (halves it at 2 GPUs). Also
    cheaper QK-Clip tracking: the extra QKᵀ materializes 32·12·1024² logits per layer;
    options are flash attention's logsumexp (an upper bound within log(seq)), tracking
    every k steps, or a slice of the batch.
- Profile results (2026-11-28). W&B runs: `m6fgv8v0` (adamw_groups), `lo2yrbw7` (muon),
  `64d02urp` (muon_no_qk_norm), `m9nmxmkp` (muon_clip). The table gives medians over steps
  50–299; W&B's `train/step_time` agrees within 1 ms.

  | profile | step | forward | backward | optim |
  | --- | --- | --- | --- | --- |
  | adamw_groups | 101.1 ms | 31.9 | 64.4 | 2.8 |
  | muon | 106.8 ms (+5.6%) | 31.6 | 64.2 | 9.3 |
  | muon_no_qk_norm | 103.1 ms | 31.0 | 60.9 | 9.0 |
  | muon_clip | 117.4 ms (+13.9% on no_qk_norm) | 45.5 | 60.7 | 9.5 |

  - The rewrite recovered most of Muon's overhead: +15.8 ms per step in `h12_muon`, +5.7 ms
    now. AdamW's step is the same in both (100.9 and 101.1 ms).
  - The Muon step is now GPU-bound: kernels fill 8.6 of its 9.3 ms. Per step:
    Newton–Schulz 4.7 ms (1.54e12 FLOPs, ~330 TFLOP/s), gathers 1.5–2.1 ms, momentum and
    decay 0.7, update 0.5, and the AdamW groups ~0.6.
  - QK-norm costs 3.7 ms, matching the earlier ~3.5%.
  - QK-Clip costs +14.3 ms against `muon_no_qk_norm`, almost all of it max-logit
    tracking in the forward pass. Per step: the fused `masked_fill`+`amax` reduction over
    the logits 7.6 ms, the QKᵀ GEMM that writes them 5.9 ms, and copies feeding the bmm
    1.2 ms. The clip itself is 0.26 ms.
  - The kernel lines (`kprof_us`) also list the profiler ranges (`section:*`, `muon:*`) as
    GPU entries, so drop those before summing kernel time.
- Split and logsumexp (2026-11-28, Janek: split Newton–Schulz; option (a) for the clip).
  - Split. Per shape, rank s owns the s-th of world equal blocks of the matrices. The
    count is padded with zero matrices, which Newton–Schulz maps to zero. `_to_owners`
    sends every rank's rows to the owners (an async `all_to_all_single`), each owner runs
    Newton–Schulz on its block, and `_from_owners` sends each rank its rows back.
  - All the first exchanges go out before any Newton–Schulz, so later shapes' transfers
    overlap earlier shapes' math. Traffic equals the old all-gather, and Newton–Schulz per
    rank is 1/world of it.
  - Ranges: `muon:to_owners`, `muon:newton_schulz`, `muon:from_owners`, `muon:update`
    (`muon:gather` is gone).
  - Same weights bit for bit as the previous version (`$CLAUDE_JOB_DIR/tmp/muon_split_bench.py`),
    and the 2-rank sharded = unsharded test still passes bit for bit. Its 1-row and 3-row
    matrices exercise both the zero-matrix and the row padding.
  - CPU, 2 gloo ranks, h12's 84 matrices at width 192: 400 → 192 ms per step (2.08×).
    Host overhead at width 8: 5.6 → 4.4 ms on 2 ranks, +0.3 ms on 1 rank.
  - Logsumexp. With `qk_clip` set and in training on CUDA in bf16/fp16, `RoPEAttention`
    calls `aten._scaled_dot_product_flash_attention` directly: the kernel SDPA picks here,
    so the attention output is unchanged. It takes the head max of its logsumexp.
  - Per query row the logsumexp lies between the row's max logit and that plus
    log(row length), at most +6.9 at seq 1024. So the clip may act slightly early, never
    late, and `train/max_attention_logit` reads up to that much high. At h12 (true max
    75.5) it still stays below 100.
  - CPU and fp32 keep the exact logits, so the CPU tests still check exact values. The GPU
    path hasn't run yet, under compile and FSDP2 included; the profile run is its first
    test.
  - Configs: the four `h12_profile_*` configs are unchanged and profile the current code.
    Rerun them; their W&B names repeat, so take the newest run per name.
- Round-2 profiles (2026-11-28). W&B runs: `15dxg20b` (adamw_groups), `dpy48nvh` (muon),
  `t5ftheh2` (muon_no_qk_norm), `rqtkbff5` (muon_clip). Medians over steps 50–299, with
  W&B `train/step_time` in brackets:

  | profile | step | forward | optim | round 1 step |
  | --- | --- | --- | --- | --- |
  | adamw_groups | 100.4 ms [101.2] | 31.6 | 2.8 | 101.1 |
  | muon | 103.7 ms [104.3], +3.1% | 31.5 | 6.3 | 106.8 |
  | muon_no_qk_norm | 99.8 ms [100.8] | 31.0 | 6.0 | 103.1 |
  | muon_clip | 100.8 ms [101.4], +0.6% on no_qk_norm | 31.0 | 6.8 | 117.4 |

  - Muon overhead: +15.8 ms in `h12_muon`, +5.7 in round 1, +3.1 ms now. Newton–Schulz
    per rank fell 4.7 → 2.6 ms. The exchanges are `to_owners` 1.2–1.8 ms and `from_owners`
    1.0–2.4 ms; their NCCL time partly overlaps Newton–Schulz, so the ranges add up to
    more than the step's wall time.
  - At equal wall-clock Muon now gets ~48.5k steps. From its stable-phase slope that
    costs ~0.004, so it would finish near 3.095 against AdamW's 3.106 (rough, and against
    the one-group baseline).
  - The logsumexp route works on GPU under whole-block compile and FSDP2. The clip run's
    forward equals no_qk_norm's, so tracking is free; the clip itself is 0.26 ms. MuonClip
    is now 2.8% faster per step than Muon with QK-norm, because it drops QK-norm. Per
    step, though, its loss was 0.008 worse, which the extra steps at equal wall-clock
    (~0.004) don't make up.
  - The logsumexp bound is tight in practice. The clip runs of rounds 1 (exact) and 2
    (logsumexp), same config and seed, logged window maxima 18.38 / 26.62 / 28.62 and
    18.34 / 26.81 / 28.63. That is within 0.2, against a worst case of 6.9; GPU runs
    aren't bit-identical, hence the one lower value.
- Landing (2026-11-28): rebased onto `pretraining-jm` at `266886d0`, which brought line 1's
  NoPE (`no_rope_layers`), looped depth (`common.loop`) and the dff = 3·dmodel test fix.
  The conflicts were both sides adding attention switches; both were kept
  (`base.yaml`, `AttentionSchema`, `SWITCHES`, `SWITCHES_OFF`/`SWITCHES_ON` with `qk_clip`
  on = 100). `RoPEAttention` auto-merged: `rope` and `qk_clip` sit side by side, and the
  clip stays exact under NoPE. After the rebase:
  - the suite gives 296 passed, 0 failed, and the hygiene tests pass;
  - `tiny_local` and `tiny_local_param_groups` are byte-identical to the original
    baselines;
  - `tiny_local_muon` and `tiny_local_muon_clip` are byte-identical to before the rebase.

### Task 4: muP for Muon

Since task 3 (2026-11-26), read before the scope:
- Muon groups use Moonlight's scale, `lr·0.2·√max(rows, cols)`, with
  `trainer.learning_rate` and the role's decay (lr·wd). The update's spectral norm grows
  as √width at fixed aspect ratio, so start from Muon LR × m^-1/2.
- Replace the builder's assert (Muon with `base_dmodel` set) with the rule, and decide
  the AdamW head's together with it (÷ m under simpleP).
- The router (`num_experts × dmodel`) is in Muon. Its max dimension is dmodel, so it
  scales like the other matrices.

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

Since task 2: the AdamW groups' rule is simpleP's (LR ÷ m on head; decay unscaled).
Reuse `scripts/mup_coord_check.py` (add the Muon option), the paired-grid sweep configs
`configs/jm/feature/mup_width_*.yaml`, and `jupyter_notebooks/mup_lr_plots.ipynb`.

Checks: as in task 2, a CPU coordinate check at fixed depth. Then stage 2,
`configs/jm/feature/h12_muon_mup.yaml` with a base width below h12, and on top of it a width sweep
with Muon.

### Out of scope for line 2

Attention and residual code in `src/core/model.py` and `src/core/attention.py`, and the
packer in `src/core/datasets.py` (line 1). The FFN slot and `src/core/moe.py` (line 3).
The float8 conversion in `main.py` (line 6), which sits in the same function as the
optimizer code, in a different hunk. `src/core/trainer.py` beyond the `train/lr` line.
