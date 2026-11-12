# Line 2: optimizer switches

## Human

New optimizer options, each off by default. Now: task 1/4, param groups, not started.

- [ ] Param groups: no weight decay on gains and embeddings
- [ ] muP for AdamW: best LR stays put across widths
- [ ] Muon: Muon for matrices, AdamW for the rest
- [ ] muP for Muon: best LR stays put with Muon

**Needs Janek:** nothing.

**Last change:** plan written; no task has landed yet.

## AI

Plan for line 2 of the pretraining switches (`plan/today.md`): param groups → muP for
AdamW → Muon → muP for Muon. Each task builds on the one before. The per-switch survey
is `pretraining_switches.md`.

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
| 1 | Param groups | todo | |
| 2 | muP for AdamW | todo | |
| 3 | Muon | todo | |
| 4 | muP for Muon | todo | |

### Today's code (at `ba679a70`)

- `main.py:208` builds `torch.optim.AdamW(model.parameters(), ...)`: one group, so
  `trainer.weight_decay` (0.1) also decays RMSNorm gains and the embedding.
- `main.py:286` builds the same AdamW and scheduler again on the `only_weights` path.
  `_misc/default.yaml` sets `only_weights: true` and `load.path: null`, so a run on those
  defaults takes this path: the optimizer is built twice and the second one trains.
- `get_cosine_scheduler_with_warmup` asserts one group (`src/core/schedulers.py:81`), and
  its `eta_min` is absolute, taken from group 0. `tiny_local` uses cosine
  (`_trainer/llama.yaml`); `_trainer/nano.yaml` and the h12/h16 sweeps use WSD.
- `WSDScheduler` chains `LinearLR`/`ConstantLR`, which scale each group's own LR, so it
  already handles several groups. Its `load_state_dict` replays steps.
- `train/lr` logs `scheduler.get_last_lr()[0]`, i.e. group 0 (`src/core/trainer.py:749`).
- Gradient clipping is global over `model.parameters()` and doesn't see groups.
- The optimizer is built after `compile` (in place, names unchanged) and `fully_shard`
  (DTensor params with global `ndim`; module classes are subclassed, so `isinstance`
  still holds).

### Shared decisions

#### Parameter roles

Every parameter gets exactly one role, from its shape and owning module, never from its
name. The first matching rule wins:

| Role | Rule | In today's model |
| --- | --- | --- |
| `vector` | `p.ndim < 2` | RMSNorm gains: 2 per block, plus the head norm |
| `embedding` | owned by an `nn.Embedding` | `embedding.weight` |
| `head` | inside a `TransformerHead` | `head.linear.weight` |
| `hidden` | everything else | q/k/v/o_proj, ff_pre_act, gate, ff_post_act |

The h4 `tiny_local` model has 39 parameters: 28 hidden, 9 vector, 1 embedding, 1 head.

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
  - on the CLI, `_trainer/_optimizer@optimizer=<option>`, only for configs that don't
    reach `_trainer` through `..`
- Schema typing in `src/definitions.py` follows the scheduler precedent. Task 1 decides
  how and records it.

#### Acceptance check (every task)

Two stages, in this order. Copy `data/` and `data_eval/` from the main checkout before
running anything.

**Stage 1, local:**
1. **Off is unchanged.** Before editing, record the losses of
   `pixi run python main.py --config-path configs --config-name tiny_local infrastructure.metric_logger.type=stdout`
   on CPU. After your change, the same command gives identical `train/loss` lines. Use
   CPU, because GPU runs aren't bit-reproducible.
2. **On trains.** `configs/tiny_local_<switch>.yaml` holds the overrides, and `main.py`
   trains with it on CPU.
3. **Tests.** `pixi run python -m pytest -m "not gpu and not slow"` passes. New unit tests
   go in `tests/test_optimizers.py`.
4. Your task's own local checks, listed in its section.

**Stage 2, cluster config** (only after stage 1 passes): write
`configs/jm/h12_<switch>.yaml` on top of `h12_chinchilla`, following the pattern of
`h16_chinchilla_wsd.yaml` (`git show ba679a70:configs/jm/h16_chinchilla_wsd.yaml` if it
has since been deleted):
- The defaults list is `h12_chinchilla`, then the
  `- override ../_trainer/_optimizer@optimizer: <option>` entry, then `_self_`.
- The body sets only `infrastructure.metric_logger.name` and tags.
- Everything else is inherited, including the `^learning_rate` sweep, so each point
  compares directly with the h12_chinchilla baseline.

Janek launches it with
`run_exp.py --config-path configs/jm --config-name h12_<switch>`; agents don't submit
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
- `origin/ctx_scl-simpleP` holds a related parametrization (`configs/_parameterization/`).
  It is not muP.

Checks:
- Coordinate check on CPU: h = 2, 4, 8, 16 at fixed depth (pin `common.n_blocks`, since
  the H family scales depth with H). Run a few steps and log activation and update RMS per
  role: under SP they grow with width, under muP they stay flat. Add a script under
  `scripts/`, and the resulting table or plot here.
- Stage 2: `configs/jm/h12_mup.yaml`. Pick a base width below h12 (a small proxy). If
  h12 is the base, every multiplier is 1 and the run differs from SP only in the init and
  decay rules.
- Width sweep, on top of stage 2: an LR sweep at three or more widths, fixed depth, SP vs
  muP, with short WSD runs. The muP optimum should stay put across widths. Write the
  configs under `configs/jm/`; Janek launches them.

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
- `configs/tiny_local_muon.yaml` trains on CPU.
- Unit tests:
  - Newton–Schulz output has singular values near 1.
  - With every parameter in AdamW groups, the class matches `torch.optim.AdamW`.
  - The sharded step equals the unsharded one, in a 2-rank gloo CPU test built on the
    FSDP2 harness in `tests/test_distributed.py`.
- Stage 2: `configs/jm/h12_muon.yaml`. h12_chinchilla runs on 2 GPUs, so this run
  exercises FSDP2 Muon. Compare it with h12_chinchilla by loss and s/step at each LR
  point. If your update scale doesn't reuse AdamW's LR, the inherited `^learning_rate`
  grid is in the wrong range: override it in this config and say so.

### Task 4: muP for Muon

Scope:
- Width rules for the Muon groups. The AdamW groups keep task 2's rules. In the H family
  every matrix's aspect ratio is fixed (dff = 4·dmodel, kv = dmodel/2), so shape factors
  are constants across widths. With Keller's scale the update's spectral norm is O(1),
  which meets the spectral condition for feature learning (Yang, Simon and Bernstein, "A
  Spectral Condition for Feature Learning"), so hidden LR should be width-independent.
  With Moonlight's RMS match the spectral norm grows as √width, so hidden LR should scale
  as 1/√width. Derive the rules from task 3's choice, verify them, and write the table
  here.
- The config surface is the same: a Muon + muP option in `_trainer/_optimizer/`.

Checks: as in task 2, a CPU coordinate check at fixed depth. Then stage 2,
`configs/jm/h12_muon_mup.yaml` with a base width below h12, and on top of it a width sweep
with Muon.

### Out of scope for line 2

Attention and residual code in `src/core/model.py` and `src/core/attention.py`, and the
packer in `src/core/datasets.py` (line 1). The FFN slot and `src/core/moe.py` (line 3).
The float8 conversion in `main.py` (line 6), which sits in the same function as the
optimizer code, in a different hunk. `src/core/trainer.py` beyond the `train/lr` line.
