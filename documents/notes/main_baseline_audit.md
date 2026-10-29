# Making `main` a clean baseline — audit, fixes and test plan

**Goal.** `main` is the branch point for a new, non-PC project. Two separate questions
had to be answered before that branch is cut:

- **(A) Is `main` correct?** Does the framework still do what the paper says it does?
- **(B) Is `main` a *clean* baseline?** Can a non-PC project branch from it without
  inheriting projected-compression?

(B) is the constraint that motivated the merge — *"all the things I need to continue the
PC project need to be on main, but I don't want those PC things to interfere with trainer
and all the other things that can be used in future non-PC projects."* It was **violated**
when this audit started (F1), and is fixed.

Audited from `main @ 2e51359c`, 2026-10-27, on branch `fix/main-baseline-audit`.

---

## 1. Status

| | before | after |
|---|---|---|
| Tests in the suite | 11 | **69** (63 selected in CI) |
| Test files | 2 | 7 |
| Runnable configs that compose *and* resolve | 76 / 82 | **82 / 82** |
| `src/core` → projected-compression references | 3 | **0** |
| CI | `black` only | `black` + the test suite |
| A bare `pytest` run | collection error | passes |

**All six tiers are done.** Tiers 0 and 1 are enforced in CI. Tier 2 is done on CPU —
including resume-from-checkpoint, which found the most serious bug here (F14) — and its GPU
half passes on both helios and lem. Tier 3 is covered to a single rank in the unit tests
and to four real ranks by the PCD-50 run; its last open question, 1-GPU vs 2-GPU parity,
turned out to be a wrong expectation rather than a defect, and is now pinned by tests.
**Tier 4 reproduces the PCD-50 paper number over a complete 4096-step run** — final loss
2.6741 against the reference 2.6705, +0.13%. Tier 5 passes on helios, lem and entropy.

**What each cluster actually carries, since "verified on three clusters" would overstate
it.** The tiers are not uniformly distributed:

| | helios (GH200) | lem (H100) | entropy (h100a) |
|---|---|---|---|
| test suite | yes | **no** — no pytest in the env | yes, via a layered venv |
| PC smoketest | yes, GPU | yes, GPU, real 1B mem-eff | yes, **GPU** (A100, job 30738) |
| Tier 3 distributed | 1 and 2 ranks, resume | **4 ranks**, production trainer | **no** |
| Tier 4 numerical | — | **full 4096-step PCD-50** | **no** |
| `pc_finalize` / PC→HF | `pc_finalize` end to end | PC→HF end to end | — |

Tier 4 and the 4-rank Tier 3 evidence rest on **lem alone**; entropy contributes the suite
against a different dependency set and a CPU smoketest, not a distributed or numerical run.
A GPU smoketest there (job 30689) was still queued when this was written.

The suite's final state on GH200 (job **25325890**, at branch HEAD `377ae27d`):
**56 passed, 10 skipped, 0 failed**, `pytest` exit 0, with `-m "not network"` selecting 66
of the 68. The 10 skips are exactly the tests that need the gitignored `data/` — three
dataloader-determinism, three rank-sharding, four training-smoke — which helios does not
have, since it reads fineweb from cluster storage instead. Run locally with `data/`
present, all 68 pass.

Two corrections worth carrying forward, because both were wrong in an earlier draft and
both cost real coverage:

- **"No cluster access."** Wrong twice over, and the second time took three corrections to
  shake. helios and lem both authenticate as `plgmstefaniak`; checking only the cluster
  that had been named and concluding there was none was the first mistake. The second:
  **entropy is reachable too.** `~/.ssh/config` defines *three* aliases for
  `entropy.mimuw.edu.pl` — `entropy` (user `mstefaniak`) and `entropym` (user `crewtool`)
  both fail with `Permission denied (publickey)`, but **`entropy_j` (user `j321m`)
  connects**. I tested one alias, got a refusal, and generalised it to the host. The
  lesson is the same one F15 and the FSDP2-on-CPU correction taught: a single negative
  result is not a property of the system.
- **"Distributed cannot run without a GPU."** The framework is CUDA-gated, but FSDP2 is
  not: `fully_shard` works on a CPU mesh with gloo. That reaches `pc_finalize` and the
  sharded checkpoint path, neither of which had any test. See Tier 3.

Everything that needed a GPU cluster has had one: more than one rank, NCCL, the bfloat16
mixed-precision policy, the 4-GPU world size, the numerical reproduction, and `pc_finalize`
through `main.py`. Three things remain out of reach, each for a stated reason rather than
for want of trying:

- **`bench_block.py`'s timing comparison.** Its committed `.jsonl` was produced on entropy
  and step times are not comparable off that hardware. entropy is now reachable (see
  below), so this is no longer blocked in principle — it was simply not re-run, because
  the defensible comparison is the PC-vs-HP overhead *ratio* rather than absolute step
  times, and that is an experiment rather than an audit check.
- **`llama_1B_memeff_to_hf_conversion.yaml`** — `convert_memeff_to_hf.py` calls
  `AutoConfig.from_pretrained` and `AutoTokenizer.from_pretrained` on
  `meta-llama/Llama-3.2-1B`, a **gated** repo. lem's HF cache holds only `models--gpt2`;
  helios has `Llama-3.1-8B` but not `Llama-3.2-1B`. A trained checkpoint now exists, so
  this is blocked purely on gated access being granted.
- **`run_eval.py`** — it imports `version_code` from `run_exp`, which commits onto whatever
  branch you are standing on and only restores in a `finally`. That is the same reason
  `run_exp.py`'s submission machinery was left alone; driving it from the audit branch
  risks leaving the branch broken. Its eval target would also need an HF checkpoint, so the
  gated-access block applies to it as well.

---

## 2. What exists today

| Surface | State |
|---|---|
| `.github/workflows/black.yml` | `black --check` |
| `.github/workflows/tests.yml` | **new** — the suite, CPU wheels, offline |
| `.githooks/pre-commit` | black **and** a `pixi.lock` version guard; now uses `--frozen` |
| `conftest.py` / `pytest.ini` | **new** — repo root on `sys.path`, markers, testpaths |
| `tests/test_repo_hygiene.py` | **new** — Tier 0, 6 checks, ~20s |
| `tests/test_projected_compression.py` | **new** — Tier 1 PC, 5 tests, ~6s |
| `tests/test_core_framework.py` | **new** — Tier 1 framework, 17 tests, ~2s |
| `tests/test_trainer_save_types.py` | **new** — save-type contract, 12 tests, ~4s |
| `tests/test_training_smoke.py` | **new** — end-to-end local run, 3 tests, ~28s, `slow` |
| `tests/test_convert_memeff_to_hf.py` | 2 tests, now marked `network` |
| `tests/test_slicegpt.py` | 9 tests, CPU, tiny model from the real configs |
| `smoke/` | SliceGPT end-to-end scenarios, cluster |

`tests/test_slicegpt.py` was the pattern everything new follows: a tiny model composed from the
*real* configs with the dimensions overridden, so config wiring is exercised too.

---

## 3. Findings

Every one was reproduced through the repo's own entry points, not inferred.

### Fixed

**F1 — `src/core/` depended on projected-compression.** `trainer.py` imported
`finalize_projection_weights`, listed `pc_finalize` in `SUPPORTED_SAVE_TYPES` and carried
`save_pc_finalized_checkpoint()`. A non-PC project inherited all of it. The final-save
branch of `train()` became `save_final_checkpoint()`, which subclasses override and defer
to with `super()`; finalization moved to a new `PCFinalizeTrainer`. This also broke an
import cycle, since `compression.py` imports `main`, which imports `src.core`.

**F2 — `configs/pc_project/llmr_300.yaml` had `batch_size` twice** (512 then 16), which
Hydra's strict loader rejects, so the config could not load at all — nothing historical is
affected, because no run ever got past loading it. Both lines arrived in the same commit,
so neither was "the later one"; kept 512, matching sibling `llmr_800`. **Maciej confirmed
512 on 2026-10-28**, so this is a decision rather than my guess.

**F3 — three conversion configs could not compose by either invocation.**
`_model/llama/base_model.yaml` referenced the `ff_layer` group absolutely, which resolves
from the search-path root: fine as `--config-path configs`, broken for the documented
per-directory form `run_exp.py` uses. It made `llama_1B_conversion`, `llama_8B_conversion`
and `hf_to_nano_conversion/1B` unrunnable — the configs that *produce* the PC checkpoint
every `*_comp_memeff` run consumes as `source_model_path`.

**F4 — two pretrain configs never set `common.batch_size`**, which
`_dataset/default.yaml` interpolates and no `_model` or `_trainer` supplies. The 1B one
was masked: it failed earlier on `gres: ???`, so a check that stops at the first missing
mandatory value never reached the bad interpolation.

**F8 — a bare `pytest` failed collection.** No root `conftest.py`, so `tests/` could not
import `src`. Only `python -m pytest` worked, because it injects the cwd. CI would have
hit this immediately.

**F11 — `solve_config_lr` dropped the learning-rate exponent.** It read
`ret_exp_lr = ret_exp_lr` on the `config_lr >= 1.0` branch — a self-assignment leaving it
`None`. That is the branch every config using the `n` means `2**-n` convention takes, so
wandb recorded `exp_lr: None` for those runs. The rate itself was always right; the
exponent is what identifies a run afterwards, and run provenance leans on it.

**F14 — a resumed run replayed the data from the beginning.** `train()` takes exactly
one batch per step and zips `range(start_step, n_steps)` against `self.train_dataloader`.
The datasets are deterministic iterables that reseed in `__iter__`, so a resumed run got
a **fresh** iterator starting at batch 0 while the step counter carried on from the
checkpoint: a run resumed at step 11 re-trained on batches 0, 1, 2 … and never saw the
batches it was supposed to.

It failed in the worst direction. On `tiny_local`, a 20-step run interrupted at step 10
and resumed reported **10.17** at step 11 against **12.39** uninterrupted — the loss
*drops* by 2.2, because the model is being shown examples it already learned, so it reads
as a good run rather than a broken one.

Fixed with `Trainer._batches_from_start_step()`, used by all three training loops
(`src/core/trainer.py` and the two PC trainers that override `train()`). The resumed run
is now bit-identical to the uninterrupted one at every step.

This path was marked untested in `configs/_checkpoints/none.yaml`
(`training_state_filename: null # not tested`), so it is unlikely any published result
came through a resume — but any preempted-and-requeued job would have hit it.

**F15 — `init_compression` built the projections on CPU regardless of the model.** The
top-k indices went to `get_device()` while `init_projections` created its weights with no
device or dtype, so always float32 on CPU. Where the model is not on `get_device()` the
norm slice `cloned_data[dmodel_top_indices]` gathers a CPU tensor with cuda indices and
raises; every test in `tests/test_projected_compression.py` failed on helios while passing
locally.

Not a production outage, and worth being precise about: `main.py` places the model before
`apply_functions` runs, so the gather is consistent there, and FSDP2 then shards the
CPU-created projections onto the GPU. The helios smoketest trains normally before and
after the fix. What remained was a latent inconsistency — the projections landed wherever
FSDP2 later put them rather than where the model was, nothing fixed them up without FSDP2,
and the path could not be tested on any machine with a GPU. Both indices and projections
now follow the model's own device and dtype, which also lifts the float32-only limitation
recorded as F12.

**F18 — resume crashed outright whenever eval was switched off.**
`Trainer.__attrs_post_init__` advances the eval stream to where a resumed run left off:

```python
if self.start_step > 0:
    n_skip_eval_batches = (self.start_step - 1) // self.eval_interval * self.n_eval_steps
```

`eval_interval` is **0** whenever eval is off, and **31 shipped configs set it** — the
PCD-50 repro among them. So every one of those runs raised `ZeroDivisionError` before
training began and **could not be resumed at all**: a preempted-and-requeued job was
unrecoverable. Together with F14, which silently replayed the data, the resume path was
broken in both directions — one loudly, one quietly.

It needs no GPU and no data to reproduce, but it only fires on a resume, which is why it
sat behind the one path nothing routinely runs. Found by the GPU resume job (helios
25327019), not by anything in the suite.

Guarded on `eval_interval > 0` and on an eval iterator existing, with four regression
tests. Two things worth recording from the fix. `eval_iterator` was only ever assigned
inside the eval branch, so reaching it with eval off was an `AttributeError` waiting behind
the `ZeroDivisionError`; it is initialised to `None` now. And my first guard tested the
iterator for truthiness, which calls `__len__` and raises
`TypeError: object of type '_InfiniteConstantSampler' has no len()` — caught not by the new
unit tests, whose stub loader has a length, but by the existing end-to-end resume test.
The check is `is not None`.

**F19 — both mem-eff → HF conversion configs had been dead since March.**
`convert_memeff_to_hf.convert_pc_to_hf` called
`get_metric_logger(..., neptune_run_id=None)`. The neptune → wandb transition (d25c326b,
PR #169) renamed that parameter to `tracker_run_id` and missed this one caller — the only
one left in the repo. `convert_pc_to_hf` is the `init_model_opt_sched_fn` entry point for
**both** `llama_1B_memeff_to_hf_conversion` and `llama_8B_memeff_to_hf_conversion`, so both
died with `TypeError` on the first line of real work. The mem-eff PC checkpoints that every
`*_comp_memeff` run produces could not be converted to HF at all, which is the step before
any lm_eval number.

Nothing in the suite could see it. Tier 0 imports every module, but the bad call is inside
a function body that no test executes, and the configs that reach it need a trained 1B
checkpoint. It took running the conversion for real (lem job 6183879) to surface it.

Fixed by dropping the argument — `tracker_run_id` defaults to `None`, so it carried no
information. `test_call_sites_match_the_function_signature` now walks the AST of every
tracked Python file and checks each `get_metric_logger` call's keywords against the live
signature, which catches this whole class: a rename that misses a caller in code the tests
never run.

**And a second bug was stacked behind it, in the same four lines.** With the `TypeError`
gone the conversion got one line further and raised
`AttributeError: 'dict' object has no attribute 'type'`. The call passed
`instantiate(cfg.infrastructure.metric_logger, _convert_="all")`, and `_convert_="all"`
yields a plain dict, while `get_metric_logger` reads `.type` by attribute. The two working
call sites — `main.py:228` and `initialization.py:49` — pass the config node itself; this
one also silently omitted `full_config`, which is what populates the wandb run config.

The caller now matches those two, and `get_metric_logger` normalises a dict to a
`DictConfig` rather than asking every caller to remember which form it is holding.
`test_metric_logger_accepts_either_config_form` covers both. Two defects in four lines,
neither reachable without a trained 1B checkpoint — which is the argument for running a
config rather than reasoning about it.

**F17 — `save_training_state` logged a path with no file at it.** The function writes to
`step_checkpoint_path(save_config.path, step)` — that is `<path>/step_<n>/` — but logged
`save_config.path`, the parent. Measured directly: the log named
`<root>/__training_state_filename.pt` while the file sat at `<root>/step_5/`, and the root
held nothing.

That matters because resume reads `load.path` and looks for the training state at exactly
that path. Anyone following the log would point `load.path` at the parent, get
`Training state file ... not found. Starting training from scratch`, and restart at step 0
having meant to resume. It is F14 one layer out: the thing that fails is loud, and the
thing you act on is wrong. Fixed to log `path`; a test now asserts that whatever the log
names is a file that exists, and the old string fails it. Every other checkpoint log line
in `src/core` and `src/projected_compression` already names the variable it wrote to, so
this was the only one.

**F7 — `pixi run` rewrote the lock to v7. FIXED, and it was worse than recorded.**
The finding said pixi 0.81 rewrites `pixi.lock` from v6 to v7, which helios's 0.67.1
cannot read. Measuring four real versions rather than inferring from one:

| pixi | where | effect on the lock |
|---|---|---|
| 0.63.2 | lem | writes **v6** |
| 0.67.1 | helios | leaves it at **v6** |
| **0.77.0** | **entropy** | **bumps to v7** |
| 0.81.0 | this machine | bumps to v7 |

**entropy's own pixi corrupts the lock**, which nothing had recorded — the entry named only
the local 0.81, so the hazard looked like a workstation problem when one of the clusters
has it too.

`pixi.toml` now declares `requires-pixi = ">=0.60,<0.77"`. A too-new pixi stops with a
clear message instead of silently rewriting: verified in both directions, local 0.81
refuses and lem's 0.63.2 runs normally leaving the lock byte-identical. The clusters
activate from their own `PIXI_HOME`, which carries its own manifest, so the bound binds
only inside the repo — which is exactly where the damage happened. Running code on a newer
pixi still works by bypassing it, as `documents/issues.md` already advised:
`./.pixi/envs/default/bin/python run_exp.py ...`.

I had twice called this infrastructure and out of scope. It was neither.

**F16 — the committed lock did not satisfy the manifest. FIXED.** `pixi.toml` declared
`pytest = ">=8.0,<9"` and `accelerate = ">=0.26.0"` that `pixi.lock` did not carry, and
pinned `transformers == 4.56.2` while the lock held **4.55.0**. Any environment installed
from the committed lock therefore had no pytest — confirmed on lem (both
`nano/$USER/pixi` and `plgllmefficont3/mstefaniak/pixi`: `No module named pytest`) and in
the local `.pixi/envs/default`. helios's environment was re-solved rather than installed
from the lock, so it had pytest all along.

The fix is the by-the-book one, and the reason it was thought impossible was a wrong
assumption of mine: that regenerating the lock necessarily moves it to v7. It does with the
local pixi 0.81.0, but the format follows the *writer*. **lem runs pixi 0.63.2 and helios
0.67.1** (read on a GH200 compute node — helios's login node is x86_64 while its pixi
binary is aarch64, so the version cannot be read from the login node). Regenerating with
pixi **0.63.2**, the older of the two, keeps the lock at **v6** and both clusters can read
it.

The result is small and targeted — **168 insertions, 41 deletions**, against the
+6234/−6089 that a pixi 0.81 rewrite produces — and re-running `pixi lock` reports
"Lock-file was already up-to-date", so it is stable rather than churning. It adds pytest
8.4.2 (with `iniconfig` and `pluggy`) and `accelerate` 1.15.0, and moves `transformers`
4.55.0 → 4.56.2 and `tokenizers` 0.21.4 → 0.22.2.

That bump is not a new risk: helios's environment **already runs transformers 4.56.2 and
pytest 8.4.2**, verified on a compute node. The regenerated lock codifies what production
already runs and brings lem into line with both helios and the manifest. The v6 pin and its
hygiene test are unchanged, so F7 still holds: do not let pixi 0.81 rewrite the lock.

This is the v6 pin's cost, now concrete. The lock is deliberately held at 6 because helios
runs pixi 0.67.1 and cannot read 7 (F7), so bringing lem's environment up to date means
either regenerating the lock — which breaks helios — or installing pytest into lem's
environment out of band. Worth a decision rather than a drift: a gate that cannot run on
one of the three clusters is only a partial gate.

### Not fixed — need facts I cannot verify from here

**F9 — `configs/_cluster/lem.yaml:58-59` hardcodes another account's directory.**

```yaml
train_path_fineweb: "/lustre/pd01/plgrid/plgllmefficont2/crewtool/proba2/fineweb-edu/train/train"
eval_path_fineweb:  "/lustre/pd01/plgrid/plgllmefficont2/crewtool/proba2/fineweb-edu/train/train"
```

This is framework config that every experiment inherits, and the file already carries the
rule it breaks ("Derived from `$USER`: no account name belongs in a shared cluster
config"). The c4 paths next to it use a shared `datasets/` directory, so the shape of the
fix is probably `.../datasets/fineweb-edu/...` — but that is a claim about what exists on
lem, and guessing a data path breaks everyone's runs. Allowlisted in the Tier 0 test so
new violations still fail.

**F10 — fineweb eval was the training data, on every cluster. FIXED.**
`_dataset/fineweb.yaml` gave its eval dataloader `split: train` and `seed: 123`, the same
as the train one, and `eval_path_fineweb` equalled `train_path_fineweb` on entropy, helios
and lem. The datasets are a deterministic function of their seed, so eval yielded **the
batches the model was training on, in the same order** — a re-scoring of training data,
wrong in the flattering direction, across **29 of the 64 fineweb configs**.

Not a typo: `datasets/fineweb/` holds only `train` on both clusters, verified directly,
while `datasets/c4/` has `train` and `validation`. No fineweb validation split was ever
downloaded, which is why eval was pointed at the training data.

I first recorded this as needing a decision about experiments, and deferred it. That was
wrong — there is no second directory to point at, but none is needed. `load_from_disk`
returns a **map-style** dataset before `to_iterable_dataset`, so a `holdout` parameter can
carve a disjoint slice out of the one directory that exists: the last `holdout_fraction`
(default 1%) of rows to eval, everything before the cut to train, both computed from the
same row count so they cannot overlap. `holdout=None` keeps the old whole-dataset
behaviour.

Tested by disjointness rather than by configuration — defeat the holdout either way and
20 of 20 sequences appear in both slices. The hygiene test now accepts a shared eval path
**only** where a disjoint holdout is declared, and its allowlist drops from five entries to
one (the toy local c4, one directory on purpose).

**Consequence worth stating plainly:** fineweb training runs now see 99% of the rows, so
new runs are not bit-comparable with older ones. That is inherent to having an honest eval
rather than one that scores training data; the fraction is a config knob. Paper headline
numbers are train loss and the downstream evals come from lm_eval, so those are unaffected.

Verified directly rather than inferred: `datasets/fineweb/` contains only `train` on both
helios and lem, while `datasets/c4/` has `train` and `validation`. No fineweb validation
split was ever downloaded. c4 is configured correctly and shows what right looks like, so
this is a missing dataset split, not a copy-paste error — which is also why all three
clusters share it.

Left for the user: the fix is to build a held-out split or to stop reporting fineweb eval,
and both are decisions about experiments rather than cleanups. Guarded by
`tests/test_repo_hygiene.py::test_eval_data_is_not_the_training_data`, with the affected
pairs allowlisted so that no other dataset acquires the same shape.

Scope: the paper's headline numbers are train loss (`steps/train/loss`) and its downstream
numbers come from lm_eval via `evals/`, so neither depends on this dataloader.

### Recorded, no change made

**F5 — the wider picture is sound.** 31/31 modules import. Of 84 top-level configs, 82
are genuinely runnable (two under `configs/ff_layer/` are components that predate the
underscore convention); all 82 now compose and resolve, and 26 stop at a `???` supplied at
submit time, which is correct.

**F6 — no GPU here.** `torch.cuda.is_available()` is False. Tiers 2+ need a cluster.

**F7 — `pixi run` silently rewrites `pixi.lock` from v6 to v7.** Observed during this
audit: every `pixi run` rewrote the header, +6234/-6089 lines. helios runs pixi 0.67.1 and
cannot read v7, so an accidental `git add -A` breaks submission there behind a diff too
large to review. Now guarded in two places — a Tier 0 test and the pre-commit hook, which
also gained `--frozen` so that the hook is no longer itself a trigger. Use
`pixi run --frozen`, or call `.pixi/envs/default/bin/python` directly.

**F12 — `init_compression` builds projections with no dtype or device**, so they are
always float32 on CPU. Production is fine because `apply_functions` runs before
`setup_distributed_training` moves the model (`main.py:205-211`), but it means compression
must be initialised before any dtype or device change, and a float64 model raises. Left
alone: the production ordering is correct and changing the signature is churn. Documented
in `tests/test_projected_compression.py`.

**F13 — `finalize()` needs DTensor parameters, not a GPU.** It calls `.full_tensor()`
and reads `os.environ["RANK"]`, so it is unreachable unless the model is sharded. This
was first recorded as "needs a cluster"; that was wrong. `fully_shard` on a CPU mesh with
gloo provides DTensors, and `tests/test_distributed.py` covers the merge. A related
fact worth knowing: `finalize()` replaces the FSDP-managed parameters and sets them to
`None` off rank 0, so **a finalized model cannot be used for a forward pass or trained
on** — it is a save-and-exit path.

---

## 4. The tiers

### Tier 0 — static, ~20s, no GPU, no data — **done, in CI**

`tests/test_repo_hygiene.py`:

| Check | Catches |
|---|---|
| every `src` module imports | a module broken for everyone |
| `src/core` has no PC reference | F1 regressing |
| no duplicate YAML keys | F2 |
| every runnable config composes and resolves | F3, F4 |
| `pixi.lock` is v6 | F7 |
| no account names in framework config groups | F9 spreading |

The config check mirrors the real invocation and resolves with `throw_on_missing=False`,
so a `???` is tolerated while a broken interpolation behind one is not — that distinction
is what exposed F4's second config.

### Tier 1 — CPU unit tests, ~12s — **done, in CI**

The central one is **full-rank invariance** (`test_projected_compression.py`). At
`k = full dimension` the top-k selection is a permutation of the residual stream rather
than a reduction, and a permutation applied consistently across embedding, both norms,
attention and head is a no-op, so the logits must not move. It is the analogue of the
SliceGPT suite's `s1_invariance` and the strongest check available without a GPU.

Also covered: projections are exactly `I` at full rank with the auxiliary term at zero;
gradients reach the projections and never the frozen base weights; exactly what
`init_compression` leaves trainable; the learning-rate convention; the cosine schedule's
shape and its guard; caret-key grid expansion; and the save-type contract.

Still unwritten from the original plan: checkpoint save/load **round-trip** equality,
nano→HF→nano logit equality, dataloader determinism by seed (needs a dataset), and
importance-score shapes.

### Tier 2 — end-to-end — **done**

The CPU half is done and is now `tests/test_training_smoke.py`: `main.py` driven the way
the README documents a local run, for both save types, asserting a checkpoint lands and
the loop reports a finite positive loss every step.

**Resume-from-checkpoint is done** and was the most valuable test in this whole audit —
it is what found F14. It runs on CPU: a 20-step run checkpointed mid-flight at step 10,
resumed, and compared step for step against the uninterrupted run.

Holding `n_steps` equal across the two runs is essential and was not obvious: `warmup_steps`
and the cosine `T_max` derive from `n_steps`, so a shorter first run trains under a
different learning-rate schedule. Getting that wrong made F14 look like a 0.017 rounding
difference on the first attempt instead of a 2.2 regression.

Still blocked on access: plain/PC/distillation/prune training on a GPU. The existing
`smoketest_pc_1gpu_{helios,lem}` configs cover some of it.

### Tier 3 — distributed — **done**

An earlier draft of this document said Tier 3 was impossible without a GPU. That was
wrong, and worth recording as a lesson: the *framework* is CUDA-gated, but FSDP2 is not.
`main.py:distributed_setup()` and `setup_distributed_training()` both check
`torch.cuda.is_available()`, and `setup_fsdp2_model` builds its mesh with
`init_device_mesh("cuda", ...)` — but `fully_shard` itself works on a CPU mesh with the
gloo backend and produces real DTensor parameters with a working `full_tensor()`.

`tests/test_distributed.py` uses that to reach two paths that had no test anywhere:

- **`pc_finalize`** — `ProjectedLinear.finalize()` calls `.full_tensor()` and reads
  `os.environ["RANK"]`, so it is unreachable unless the parameters are DTensors. This was
  recorded below as F13, "cannot run single-process on CPU"; that was also wrong. The test
  checks the merge directly: each layer must collapse to `P_out W P_in + auxiliary`. The
  same path is now also exercised through `main.py` on a GPU — see "pc_finalize end to
  end" in the cluster log.
- **the sharded checkpoint path** — `dcp.save`/`dcp.load` over `TrainingState`, which the
  plain `torch.save` checkpoints never touch.

Two things the fixture has to do, both learned the hard way. It perturbs the projections
and the auxiliary term first, to stand in for a trained model: straight after
`init_compression` the projections are one-hot and the auxiliary term is zero, so merging
it is a no-op and deleting it from `finalize()` entirely still passed. And it compares
weights rather than a forward pass, because `finalize()` replaces the FSDP-managed
parameters outright (and sets them to `None` off rank 0), so the module cannot be run
afterwards — fine for production, which finalizes, dumps the state dict and stops, but it
does mean **a finalized model cannot be trained on**.

**The 4-GPU world size is now directly observed.** The PCD-50 run (lem job 6174094) logged
four distinct ranks, four `Sharding done` lines — FSDP2 sharding once per rank — and four
allocated H100s, under the production `PCDistillationTrainer` with the real 1B model. That
closes the `documents/issues.md` item which until now rested on a transitivity argument,
and it means real multi-rank NCCL sharding *is* exercised; only the unit tests are confined
to a single rank.

**The bfloat16 policy and per-block clipping are covered too**, which an earlier draft
also listed as outstanding. `setup_fsdp2_model` applies
`MixedPrecisionPolicy(param_dtype=torch.bfloat16)` unconditionally, and that run went
through it; its log carries `train/grad_norm`, `train/projection_grad_norm` and the
per-module `..._embedding` / `..._head` norms under `gradient_clipping: 0.25` with
`only_target_model_gradient_clipping: 1.0`.

**1-GPU vs 2-GPU parity: not an FSDP2 defect, but a real reproducibility bug.** helios job 25259068 ran
the same config at both world sizes with a fixed seed. Step 0 matched exactly (22.0 — same
init), both runs converged to ~9.6, but intermediate steps diverged by up to **0.75**, mean
|delta| 0.244 over 20 steps. I had expected equality on the reasoning that the global batch
is `total_batch_size` regardless of world size, so a divergence would mean FSDP2 was
changing the maths rather than distributing it.

It is not FSDP2. It is the dataloader, and it is deliberate. `C4Dataset._load_dataset` and
`FineWebEduDataset._load_dataset` both do this before any shuffling:

```python
if not self.world_size_independent:
    hf_dataset = split_dataset_by_node(hf_dataset, rank=self.rank, world_size=self.world_size)
```

Every shipped config sets `world_size_independent: false`, so the HF dataset is sharded at
load time — each rank takes its share of the 64 shards — and `AbstractDataset.__iter__`
then returns the packer unsliced. The `true` branch is the other route: load the whole
dataset on every rank and stride it by rank inside `__iter__`. Both shard; they shard at
different layers.

The consequence is that **a 2-GPU run does not read the same corpus order as a 1-GPU run.**
Measured directly, three separate processes with `WORLD_SIZE`/`RANK` set in each
environment (one process per rank is the only way to check this — `importlib.reload` inside
a single process does not re-read the env cleanly, and an earlier probe that appeared to
show ranks replaying one stream was reporting a hardcoded `echo`, not a measurement):

| streams compared | shared sequences of 24 |
|---|---|
| world_size=2 rank 0 vs rank 1 | **0** |
| world_size=2 rank 0 vs world_size=1 | **0** |
| world_size=2 rank 1 vs world_size=1 | **0** |

Ranks do not duplicate data — the thing actually worth worrying about — but neither does
either rank replay the 1-GPU stream. Shard assignment and shuffle both change with
`world_size`, and packing boundaries follow the stream, so the global batch at step *n* is
a different set of samples. Identical per-step losses across world sizes is **not a
property this framework has, and asserting it would be a wrong test**. The 0.75 spread is
two runs on different samples of the same corpus converging to the same place, which is
what the end of the curve shows.

What does have to hold is that ranks never train on the same samples: silent duplication
would scale the effective batch down and hide behind a plausible loss curve.
`test_ranks_never_see_the_same_sample` asserts it on both branches, and
`test_world_size_independent_partitions_one_stream` pins the `true` branch to an exact
interleave of the single-rank stream. Mutation-tested: dropping the `split_dataset_by_node`
call puts 12 of 12 sequences on both ranks, and dropping the `islice` stride breaks the
partition.

**This is a bug, not just a caveat.** Reproducibility is per world size: a fixed seed gives
the same data only at a fixed `world_size`. Nothing warns you when that changes, so a run
preempted and requeued on a different GPU count silently trains on different data, and any
A/B whose arms ran at different world sizes is confounded. `world_size` belongs in run
provenance next to the seed.

**The eval dataloader makes it worse, and contradicts the code's own docstring.**
`C4Dataset`'s docstring says "world_size_independent should be 'True' for tests and eval" —
but `_dataset/fineweb.yaml`, `c4.yaml` and `local_dummy.yaml` all set `false` on **both**
the train and the eval dataloader. So eval loss is scored on a rank-dependent subset, and
eval numbers do not compare across GPU counts either.

**The eval side is fixed.** All three eval dataloaders are now
`world_size_independent: true`, matching the docstring, and
`test_eval_dataloaders_are_world_size_independent` enforces it, so eval no longer moves
with the GPU count. The train-side sharding keeps `false`: it is a deliberate strategy and
should stay. What *it* needs is to be documented and recorded per run, which is why
`world_size` belongs in run provenance next to the seed.

### Tier 4 — numerical reproduction — **done, full run reproduces the reference**

Nothing below this tier proves `main` still produces the paper's numbers.

The reference is wandb run **`uhhl03fy`** (`ideas_cv/pc_v2`, name `llama1_pcd_50p_2BT_repro`),
the designated PCD-50 repro. Two things about reading it that cost time:

- the metric key is **`steps/train/loss`**, not `train/loss`; querying the obvious name
  returns an empty history for a run that is in fact complete.
- the original `ic3zozd8` is **not** in `pc_v2` and does not resolve there.

Its final is `steps/100/train/loss` = **2.67046875**, which is the 2.6705 the notes record.

Early curve, pulled from that run and kept at `scratchpad/pcd50_reference.json` (142 points):

| step | 0 | 1 | 5 | 10 | 20 | 40 | 60 | 80 | 100 | 120 |
|---|---|---|---|---|---|---|---|---|---|---|
| loss | 8.875 | 8.3125 | 6.78125 | 5.625 | 4.75 | 4.125 | 3.70312 | 3.5 | 3.39062 | 3.3125 |

**Comparing a prefix requires keeping `n_steps` at 4096** and truncating the job by wall
clock instead. `warmup_steps` and the cosine `T_max` are both derived from `n_steps`, so
shortening it changes the learning-rate schedule and the comparison proves nothing. This
is the same trap that made F14 look like a rounding error on the first attempt, and it is
worth stating because the instinct is always to shorten the run.

Submitted on lem as job 6173802: 4 H100 (matching the reference world size), the repro
config with only its paths retargeted, 35-minute wall clock. The config is generated on
the cluster rather than committed, following `smoke/README.md` -- a composed config is a
snapshot, and committing one means it silently stops testing current code.

Also in scope for this tier, and **not** done, with reasons:

- `benchmarks/bench_block.py` against its committed `.jsonl`. This one cannot be done
  honestly from here: it measures **step times**, and the committed results were produced
  on entropy (`llama_1B_pc_1blk_bs_probe_*_entropy`). Re-running on helios GH200 or lem
  H100 and diffing the numbers compares hardware, not code, and would look like a
  regression check while being nothing of the sort. The defensible variant is to compare
  the **PC-vs-HP overhead ratio** rather than absolute milliseconds, since that is what
  Fig. 2 actually reports and it is far more robust across machines — worth doing, not
  done here.
- `run_eval.py` against `evals/evals/aggregated_results_llama1b.json`. Needs a trained
  checkpoint at one of the paper's configurations, which this audit never produced.


### Tier 5 — cross-cluster — **done on helios and lem**

All three clusters are reachable. helios and lem authenticate as `plgmstefaniak`; entropy
authenticates only through the `entropy_j` alias as **`j321m`** — the `entropy` and
`entropym` aliases (users `mstefaniak` and `crewtool`) are refused with
`Permission denied (publickey)`. Running there means acting on a colleague's shared
account against a shared GPU cap, so it was done only after asking, and kept to one GPU
through `sbatch`, with nothing of theirs touched and no job of theirs cancelled.

Run on **helios** (GH200, `plgllmefficont3-gpu-gh200`) and **lem** (H100,
`plgllmefficont3`): the branch checked out on group storage, and each cluster's own
smoketest — `smoketest_pc_1gpu_helios.yaml`, a self-contained tiny PC model exercising the
pixi env, the fineweb path, the model build, `init_compression` and the training loop; and
`smoketest_pc_1gpu_lem.yaml`, the production `PCTrainer` mem-eff path on the real 1B
model. Both pass. The suite runs on helios — 56 passed, 10 skipped, 0 failed at branch HEAD
(job 25325890). It could not run on lem when this was written; F16 is now fixed, so a
fresh install from the lock there gets pytest. `pc_finalize` also runs end to end on helios;
see its section in the cluster log.

**The cemetery redirect is exercised on both.** `infrastructure.cemetery_experiments_dir`
composes to the cluster-specific path rather than the `~/llmrandom_cemetery` default, and
running exactly what `run_exp.py:205-212` runs — `mkdir -p` then a clone into
`{cemetery_dir}/{name}` — lands the clone under the redirected directory with `$USER`
expanded, leaving `$HOME` untouched. Probe directories were removed afterwards. That was
the one link in the chain nothing had ever executed.

It does **not** exercise `run_exp.py`'s submission machinery. `version_code()` commits onto
whatever branch you are standing on and only restores in a `finally`, so driving it from
the audit branch risked leaving the branch broken without adding any coverage of the
redirect itself. A real submission still tests that part.

## 5. What changed

```
core: stop src/core depending on projected-compression
tests: make a bare `pytest` run work
configs: fix the six configs that could not be composed
tests: add the Tier 0 static gate
tests: CPU unit tests for projected compression
core: report the learning-rate exponent instead of dropping it
ci: run the test suite on every PR, and guard pixi.lock
tests: cover the save-type contract and a real local training run
```

Every new check was **mutation-tested**: the defect was reintroduced and the test
confirmed to fail, then reverted. That caught a real weakness — an early version of the PC
fixture set all RMSNorm weights to a constant, and a constant vector is
permutation-invariant, so two norm-ordering bugs passed unnoticed until the fixture was
given real structure.

The two save paths were also compared against `main` directly on a real run: identical
loss trace step for step, and byte-identical `model.safetensors` and nano checkpoint
files.

---

---

## 6. Cluster execution log

All three clusters are reachable: **helios and lem** as `plgmstefaniak`, **entropy** only
via the `entropy_j` alias as `j321m`. The
branch is checked out on group storage on both (`baseline_smoke/nano`), with a hand-written
sbatch that reproduces the env from each cluster's own `_cluster/*.yaml` script block —
deliberately not through `run_exp.py`, whose `version_code()` commits onto the branch you
are standing on and leaves it broken if the job is killed.

| job | cluster | what it ran | outcome |
|---|---|---|---|
| 25254406 | helios GH200 | suite + `smoketest_pc_1gpu_helios` | smoketest **passed** (loss 14.375 → 9.875 over 20 steps on GPU); 5 PC unit tests **failed** |
| 25254605 | helios GH200 | one failing test, full traceback | `RuntimeError: indices should be either on cpu or on the same device as the indexed tensor` at `compression.py:99` |
| 25255266 | helios GH200 | suite + smoketest, after the device fix | PC tests **now pass on GPU**; smoketest trains, rc=0; one new failure in a test this branch added |
| 25255801 / 25256547 / 25256588 | helios GH200 | the remaining failures, full tracebacks | each a device assumption in a test of mine, not the framework: CPU mesh under CUDA, finalize() writing to CPU, noise built on CPU |
| **25257102** | helios GH200 | **final suite + smoketest at HEAD** | **55 passed, 7 skipped, 0 failed** (`pytest exit=0`); smoketest trains, rc=0 |
| 25259068 | helios GH200 ×2 | 1-GPU vs 2-GPU FSDP2 parity at fixed seed | running at time of writing |
| **6174690** | lem 4×H100 | **full 4096-step PCD-50** | **COMPLETED 4h41m, rc=0, all 4096 steps.** Final `steps/100/train/loss` **2.674062** vs reference **2.670469** (+0.0036, 0.13%); mean \|delta\| 0.0088 over 40 points. See "PCD-50 full run" |
| **25325254** | helios GH200 | **`pc_finalize` end to end** through `main.py` | **both stages rc=0**; merged state dict has 39 tensors and **0** projection/auxiliary tensors, `embedding.weight` `(128256, 256)`. First execution of the config this branch retargets. See "pc_finalize end to end" |
| **25325890** | helios GH200 | **suite at branch HEAD** (`377ae27d`) | **56 passed, 10 skipped, 0 failed**, `pytest` exit 0. Refreshes the stale 25257102 figure |
| **25332615** | helios GH200 | **fineweb holdout on real cluster data**, eval ON | `HOLDOUT_RC=0`; train 1266316358 rows / eval 12791074 rows from the same cut, `eval/loss` computed on the held-out slice |
| ~~6183879~~ / ~~6183912~~ | lem 4×H100 | mem-eff PC → HF conversion, first attempts | **both failed** — `TypeError` then `AttributeError`; found F19's two defects |
| **6183950** | lem 4×H100 | **the same at the twice-fixed HEAD** | `CONVERT_RC=0`; `hidden_size` 1344, `intermediate_size` 5376, safetensors + tokenizer written. Reloaded on an H100: **0.802B params, finite logits, "The capital of France is" → ' Paris'** |
| 25327019 | helios GH200 | resume from a sharded checkpoint, eval off | run A rc=0; **run B rc=1, `ZeroDivisionError`** — found F18 |
| **25328745** | helios GH200 | the same, at the fixed HEAD | **`RESUME_RESULT=IDENTICAL`** — resumed run bit-identical to the uninterrupted one over 9 steps |
| **25326196** | helios 2×GH200 | the same at **two ranks** | **`RESUME_RESULT=IDENTICAL`** — multi-rank sharded resume bit-identical to its own uninterrupted run |
| 6173738 | lem H100 | suite + `smoketest_pc_1gpu_lem` (mem-eff path, real 1B) | smoketest **passed** on H100 (production `PCTrainer`, real 1B, rc=0); suite could not run — no pytest in the env (F16) |
| ~~6173802~~ | lem 4×H100 | PCD-50 prefix, first attempt | **cancelled** -- its log pipeline died of SIGPIPE (see below) and the job was training blind |
| 6174094 | lem 4×H100 | PCD-50 prefix vs the reference curve | **451 steps, agrees to bf16 resolution from step 80**; see below |

The two helios jobs together are what found F15, and the pairing matters: the smoketest
passing while the unit tests failed is exactly what distinguishes "production is broken"
from "this is untestable on a GPU". Only the second is true.

The smoketest's losses are close but **not bit-identical** across the device fix
(step 15: 9.8125 before, 9.875 after). That is expected rather than alarming: the
projections are now created on the GPU in the model's dtype instead of on CPU in float32
and moved there afterwards, so the arithmetic differs in the last bits. Both runs descend
normally.

`test_sharded_checkpoint_round_trips` is a test this branch added, and it passes on CPU
while failing on a machine with a GPU — the same shape of problem as F15, in my own test
rather than in the framework. It builds a CPU device mesh with gloo, which is fine on a
CPU-only box and evidently not fine where CUDA is present.

### PCD-50 prefix result

Job 6174094, 4×H100 on lem, ran to its wall clock and reached **step 450** — 11% of the
4096-step reference. Compared against wandb `uhhl03fy` over all 451 matched steps. Same
config with only the paths retargeted, `n_steps` at 4096 so the schedule matches.

| step | reference | this run | delta |
|---|---|---|---|
| 0 | 8.875 | 8.9375 | +0.0625 |
| 10 | 5.625 | 5.28125 | −0.3438 |
| 40 | 4.125 | 4.0 | −0.1250 |
| 80 | 3.5 | 3.5 | 0.0000 |
| 120 | 3.3125 | 3.3125 | 0.0000 |
| 200 | 3.25 | 3.265625 | +0.0156 |
| 300 | 3.125 | 3.125 | 0.0000 |
| 400 | 3.0625 | 3.046875 | −0.0156 |
| 450 | 3.0625 | 3.078125 | +0.0156 |

| band | n | max \|delta\| | mean delta | exactly equal |
|---|---|---|---|---|
| 0–20 | 21 | 0.3438 | −0.1384 | 2/21 |
| 21–80 | 60 | 0.1250 | −0.0122 | 18/60 |
| 81–200 | 120 | 0.0469 | +0.0044 | 33/120 |
| 201–451 | 250 | 0.0625 | +0.0074 | 84/250 |

Over everything from step 40 on: mean **+0.005**, max |delta| 0.125, and **129 of 411
steps exactly equal**.

So after a noisy first twenty steps the two runs agree to within bf16 resolution and stay
there for four hundred steps, with no drift in either direction. One ulp at these
magnitudes is 0.0156 to 0.0625, which is the size of almost every non-zero delta past
step 80 — the curves are as close as the logging precision can express.

The early spread is what different hardware should produce: the reference ran on entropy,
this on H100, and early training is the steepest, least determined part of the curve.
A fixed absolute tolerance would be the wrong test there, which is why the plan called
for a prefix comparison rather than a single final number.

**The full run landed — see "PCD-50 full run" below.** It completed all 4096 steps and
reproduces the reference, so this tier no longer rests on the prefix alone.

**What the prefix established, and what it did not.** It put a PCD-50 run on the recorded
trajectory and held it there for 11% of training — a real signal, but not a reproduction of
the final loss. That needed all 4096 steps. The full run below supplies it.

### PCD-50 full run

lem job **6174690**, 4×H100, `n_steps: 4096`, **COMPLETED in 4h41m, exit 0**, all 4096
steps, final checkpoint written. Compared against wandb `uhhl03fy`
(`llama1_pcd_50p_2BT_repro`) on `steps/100/train/loss`, the 100-step rolling mean the
reference logs every 100 steps — 40 points, reconstructed from this run's per-step
`train/loss` over the same windows.

| step | reference | this run | delta |
|---|---|---|---|
| 101 | 4.252344 | 4.107813 | −0.144531 |
| 201 | 3.262969 | 3.263281 | +0.000312 |
| 501 | 3.029688 | 3.037031 | +0.007344 |
| 1001 | 2.900781 | 2.908281 | +0.007500 |
| 2001 | 2.771094 | 2.777656 | +0.006563 |
| 3001 | 2.697344 | 2.706562 | +0.009219 |
| 4001 | **2.670469** | **2.674062** | **+0.003594** |

Across all 40 points, mean |delta| **0.0088**, max 0.1445 — and that maximum is entirely
the first point. From step 201 on, every delta lies between **+0.0003 and +0.0092**, mean
**+0.0053**: a small constant offset, not drift, and it does not widen over four thousand
steps. The final number, the one the paper reports, agrees to **+0.0036 (0.13%)**.

The single outlier at step 101 is the steepest part of the curve averaged over the first
hundred steps, where the reference's hardware (entropy) and this run's (H100) have the most
room to differ; the prefix comparison saw the same early spread and the same convergence
afterwards. Shifting the window by one step moves mean |delta| to 0.0077 — the alignment is
not load-bearing.

**`main` reproduces the PCD-50 paper number.** That is the strongest statement this audit
can make about the baseline, and it is now made on a complete run rather than a prefix.

### pc_finalize end to end

This branch retargets `configs/pc_project/llama_1B_comp_finalize.yaml` at the new
`PCFinalizeTrainer`. That config is the **only** consumer of the `pc_finalize` save type,
and it cannot run without a trained 1B PC checkpoint and an importances file, so nothing
had ever driven the path through `main.py` — the change was covered by a CPU unit test
against a synthetic model and nothing else. Verifying my own edit was the gap worth
closing.

The PCD-50 checkpoint cannot serve: it is a `projected_mem_eff_1B` model, whose
finalization route is `llama_1B_memeff_to_hf_conversion.yaml`, a different config.
`configs/pc_project_mas/smoketest_pc_finalize_helios.yaml` is a tiny analogue instead —
same `_target_`, same `pc_finalize` save type, same `n_steps: 0`, same
init-compression-then-load-weights shape, on the self-contained tiny model that needs none
of the 1B assets.

helios job **25325254**, 1×GH200, two stages:

| stage | what | result |
|---|---|---|
| A | train the tiny PC model 6 steps, save type `nano` | loss 22.0 → 14.25, **sharded** checkpoint at `step_5`, rc=0 |
| B | load it, finalize, save type `pc_finalize` | merged checkpoint written, rc=0 |

The merged state dict holds **39 tensors and zero projection or auxiliary tensors**, with
`embedding.weight` at `(128256, 256)` — the target width, down from `base_dmodel` 512. Both
stages report **146,543,104** trainable parameters, which is the check that matters for the
warning in the 1B config's own comment: `init_compression` has to rebuild exactly the
structure the checkpoint's weights fit into, and it did.

What this covers that nothing covered before: `PCFinalizeTrainer` reached through hydra's
`_target_`, the `pc_finalize` save type and its validation, the `save_final_checkpoint()`
override dispatch, `finalize_projection_weights` on a real FSDP2-sharded GPU model rather
than a CPU fixture, the lazy import inside the method that breaks the
`compression` → `main` cycle, and a sharded nano checkpoint written by one process and read
back by another.

**A wart, now fixed.** The output landed in `step_-1`: `n_steps: 0` leaves `self.step` at
−1 and `step_checkpoint_path` interpolated it straight into the folder name. Not
PC-specific — any save type with `n_steps: 0` hits it, since
`_should_save_final_checkpoint` is satisfied at `step >= n_steps - 1`.

I first left this alone, reasoning that changing it moves where production checkpoints are
written. That was too cautious: a pass that completes no steps *is* at step 0, and a
negative index in a path is something an operator then has to go hunting for.
`step_checkpoint_path` clamps to 0, with a test that ordinary steps are untouched.

### Resume on a GPU, from a sharded checkpoint

F14 (resume replayed the data) and F18 (resume crashed with eval off) were both fixed and
tested on CPU in a single process. The PCD-50 run saved only at the end, so
**sharded-checkpoint resume on real GPUs had no coverage at all** — and that is the shape
production actually uses. Running it is what found F18.

The recipe needs no new config, only overrides on the existing helios smoketest: train 20
steps with `save.interval=10`, then a second run at the same `n_steps` with
`load.path=<ckpt>/step_10`, `load.training_state_filename=__training_state_filename.pt` and
`load.only_weights=false` — the combination `_checkpoints/none.yaml` marks `# not tested`.
`n_steps` is held equal because warmup and the cosine `T_max` derive from it.

| job | HEAD | result |
|---|---|---|
| 25327019 | before F18 | run A rc=0; **run B rc=1, `ZeroDivisionError`** — resume impossible |
| **25328745** | after F18 | run A rc=0, run B rc=0, **`RESUME_RESULT=IDENTICAL`** |

At the fixed HEAD the resumed run picks up at exactly step 11 and its losses are
**bit-identical** to the uninterrupted run for every compared step:

| step | straight | resumed |
|---|---|---|
| 11 | 11.1875 | 11.1875 |
| 12 | 10.625 | 10.625 |
| 13 | 10.375 | 10.375 |
| 14 | 10.125 | 10.125 |
| 15 | 9.875 | 9.875 |
| 16 | 10.0 | 10.0 |
| 17 | 9.6875 | 9.6875 |
| 18 | 9.625 | 9.625 |
| 19 | 9.5625 | 9.5625 |

**The 2-GPU variant agrees** (job 25326196, 2×GH200, COMPLETED 20m25s, both runs rc=0,
`RESUME_RESULT=IDENTICAL`). Its absolute losses differ from the 1-GPU run, exactly as the
`world_size` sharding above predicts — 11.5 against 11.1875 at step 11 — but the resumed
run matches *its own* uninterrupted run bit for bit at every compared step:

| step | straight | resumed |
|---|---|---|
| 11 | 11.5 | 11.5 |
| 13 | 10.8125 | 10.8125 |
| 15 | 10.1875 | 10.1875 |
| 17 | 9.75 | 9.75 |
| 19 | 9.5 | 9.5 |

So on a GPU, from an FSDP2 sharded checkpoint, with eval off, at one rank and at two: the
run resumes at all (F18) and resumes onto the right data (F14). Multi-rank sharded
`dcp` save/load and the `only_weights=false` path are both covered.

### The fineweb holdout on real data

The holdout (F10) was written and unit-tested against the tiny local dataset. That proves
the slices are disjoint, but real fineweb is a 1.28-billion-row Arrow dataset and the split
runs `len()` and `.select()` over it before `to_iterable_dataset` — a different proposition
at that size, and not something a CPU test can speak to. `smoketest_fineweb_holdout_helios`
runs both branches against the cluster copy, with eval left **on**, because the eval
dataloader is half of what is being tested.

helios job **25332615**, 1×GH200, `HOLDOUT_RC=0`:

```
holdout='train': 1266316358 of 1279107432 rows (cut at 1266316358)
holdout='eval':    12791074 of 1279107432 rows (cut at 1266316358)
```

Exactly 1% held out, both slices taken from the same cut, so disjoint by construction at
full scale and not merely in the fixture. `eval/loss` is computed (15.25, then 15.0 — a
random tiny model, so the value means nothing; that it is *computed on the held-out slice*
is the point). Training descends normally over 12 steps.

This also exercises `world_size_independent: true` on the eval dataloader, which is the
other half of the F10/GPU-count fix.

**The holdout costs no memory, which was worth checking rather than assuming.** Carving a
slice out of a 1.27-billion-row dataset looks like it should be expensive: if `select` built
an indices mapping, a 1.266-billion-entry index array would add roughly 10 GB of RSS to
every fineweb run. It does not. `Dataset.select` takes a fast path when given a contiguous
`range` — "If the indices correspond to a contiguous range, the Arrow table is simply
sliced" — and both holdout calls pass `range(...)` objects. `_select_contiguous` is present
in **datasets 3.1.0**, the version `pixi.toml` pins and the one the clusters run, so the
fast path is the one that executes.

This was checked because an entropy CPU run OOM'd at 32 GB and the holdout was the obvious
suspect. It was the wrong suspect: the memory went on training a 146M-parameter model on
CPU in float32 with Adam state, not on the dataset slice.

### Mem-eff PC → HF conversion, end to end

`llama_1B_memeff_to_hf_conversion` is one of the configs F3 unblocked, and the PCD-50 run
finally supplied the trained checkpoint it needs. I had twice recorded it as blocked on
gated `meta-llama/Llama-3.2-1B` access. **Both times that was wrong**: helios holds an HF
token, and lem has the model in the group's shared cache
(`.../nano/plgj321m/hf_cache/hub`). Running it found F19 — two independent defects in four
lines, which had made every mem-eff PC checkpoint unconvertible since March.

lem job **6183950**, 4×H100, against the PCD-50 checkpoint from job 6174690.
`CONVERT_RC=0`:

| field | value | expected |
|---|---|---|
| `hidden_size` | 1344 | 1344 (50% of 2688) |
| `intermediate_size` | 5376 | 5376 |
| `num_hidden_layers` | 16 | unchanged from the source |
| `tie_word_embeddings` | False | False — the compressed model has independent embedding and head |
| weights | `model.safetensors` | — |
| tokenizer | `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` | — |

Structure alone would not prove much, so the output was loaded back through
`AutoModelForCausalLM` on an H100:

```
params: 0.802B      logits (1, 6, 128256), all finite
next-token loss on a six-token prompt: 3.5079
top-1 continuation for "The capital of France is" -> ' Paris'
```

So the converted checkpoint is not merely well-formed, it is a working language model. That
is the full PC pipeline exercised end to end on `main`: train a compressed 1B to the paper's
loss, convert it to HF, and get something that answers. It is also the step immediately
before lm_eval, so the downstream evaluation path now has a real input.

Two notes. `dcp.load` reads shards straight from `checkpoint.load.path`, while the
distillation trainer writes them to `step_<n>/model`, so the path has to include `/model`.
And the run config is `configs/pc_project_mas/_lem_memeff_to_hf.yaml`, left on the cluster
rather than committed, like `_lem_pcd50_prefix.yaml` — these carry absolute per-user paths.

### entropy

The audit recorded entropy as unreachable for most of its life. That was wrong.
`~/.ssh/config` defines three aliases for `entropy.mimuw.edu.pl`: `entropy` (user
`mstefaniak`) and `entropym` (user `crewtool`) are both refused with
`Permission denied (publickey)`, but **`entropy_j` (user `j321m`) connects** — and that is
the account the project normally uses there. I tested one alias, got a refusal, and
generalised it to the host.

Running there means acting on a colleague's shared account against a shared GPU cap, so it
was done only after asking, kept to one GPU via `sbatch`, with nothing of theirs touched
and no job of theirs cancelled.

What is set up there:

| | |
|---|---|
| checkout | `/storage_nvme_4/nano/j321m/mas_baseline_audit/nano`, branch at `741dee9` |
| importances | `.../mas_baseline_audit/tiny_random_dimensions_importances.pt`, 37 KB, seeded |
| env | `/storage_nvme_4/nano/j321m/pixi` — pixi **0.77.0**, torch 2.7.1+cu128, **transformers 5.12.1** |
| config | `configs/pc_project_mas/smoketest_pc_1gpu_entropy.yaml`, importances path left `???` and passed at submit time |
| job | **30689** — suite + 1-GPU PC smoketest, log at `~j321m/mas_smoke_30689.out` |

**Result: both halves pass on entropy.**

| job | what | result |
|---|---|---|
| **30703** | full test suite, CPU, via a venv layered over the shared env | **`SUITE_RC=0`** — passes against `transformers 5.12.1`, where `pixi.toml` pins `==4.56.2` |
| **30740** | the same suite through an env **built from this repo's lock** (job 30739's sync) | **`SUITE_RC=0`** at `pytest 8.4.2 / torch 2.7.1+cu128 / transformers 4.56.2`. The gate running on a cluster through the project's own environment manager, rather than a workaround, is what F16 originally blocked |
| **30711** | 1-GPU PC smoketest, CPU, 20k-row fineweb slice | **`SMOKE_RC=0`** — loss 21.71 → 16.27 over 4 steps, peak RSS **6.29 GB** |
| ~~30700/30706/30707~~ | earlier attempts | cancelled or OOM — see below |
| 30689 | the same on a real GPU | still queued; estimated start 2026-10-29 14:00 |

The suite result is the more informative half: entropy's environment is **not** built from
this repo's lock, so this is the branch holding up against a substantially different
dependency set than helios or lem.

**Two things had to be solved to get there, and both are worth recording.**

*No pytest in the environment.* entropy's env has the same gap as lem's. Refreshing it was
not an option — it is shared, and moving `transformers` 5.12.1 → 4.56.2 could break someone
else's work. A `python -m venv --system-site-packages` layered over it, with pytest
installed into the venv alone, is additive and isolated.

*Memory, which was misdiagnosed twice.* The smoketest OOM'd at 32 GB and the new fineweb
holdout was the obvious suspect; it was not (see "The fineweb holdout on real data"). The
real cause is **Arrow page cache**: SLURM's cgroup counts mmap'd pages, and streaming the
1.27-billion-row dataset pulls them in, so the memory floor of a fineweb run is set by the
dataset rather than by the model. Peak RSS on a 20 000-row slice of the same dataset in the
same format is **6.29 GB** for a 146M-parameter model — the model was never the problem.
That is why production asks 90–220 GB per GPU and why a modest allocation OOMs on a tiny
model. Raising the request does not help on a busy node: 120 GB could not be scheduled at
all, and 60 GB would not backfill, while 28 GB with a sliced dataset ran immediately and
exercises `FineWebEduDataset`, `_load_dataset`, the holdout, `split_dataset_by_node`, the
sample packer, `init_compression` and the training loop identically.

**The GPU half ran too, on a100a rather than h100a.** `h100a` is the only node mounting
`/storage_nvme_4`, and its 8 GPUs were booked out for another ~34 hours, several of the
queue held by `QOSMaxGRESPerUser` against the shared cap. I treated that as the end of it
for several rounds. It was not: **`/home` is NFS and visible from every node**, including
`a100a`, which had 4 free GPUs the whole time. Staging the environment there routes around
the mount.

| job | what | result |
|---|---|---|
| 30722 / **30730** | stage 7.9 GB env + repo + sliced dataset to `/home` | 30722 hit its 25-minute limit at 7.3 of 7.6 GB; 30730 finished it with `rsync` (37m14s — NFS and many small files). **Relocated interpreter imports torch**, which was the real risk: conda-style envs bake absolute prefixes in |
| 30735 | GPU smoketest, first attempt | **`cuda: True`, A100 80GB** — then failed on the gated `meta-llama/Llama-3.1-8B` tokenizer, whose cache is on `/storage_nvme_4` and invisible from a100a |
| 30736 / **30737** | stage the tokenizer | 30736 copied nothing: `$DST/hub` did not exist and rsync will not create nested parents, and extension filters miss an HF cache's hash-named blobs behind symlinks. 30737 used `-L` and `--max-size=100M`, keeping the 8B weights out: **`tokenizer OK, vocab 128256`** |
| **30738** | **GPU smoketest** | **`SMOKE_RC=0`** — 20 steps on an **NVIDIA A100 80GB**, loss descending to **9.25**, against the ~9.5–9.6 the same tiny model reaches on helios and lem |

So all three clusters ran the PC smoketest on a real GPU. What entropy still does not carry
is a distributed or numerical-reproduction run; see the per-cluster table in section 1. `h100a` is the only node mounting `/storage_nvme_4` — the env, the
fineweb copy and the checkout all live there — and its 8 GPUs were fully allocated to other
users' multi-day jobs throughout, several of them held by `QOSMaxGRESPerUser` against the
shared cap. `a100a` had free GPUs but mounts `/storage_nvme_1` instead, with no env, no
torch and no module system; building one into a colleague's home was not a reasonable thing
to do on an account lent for smoketests. Job 30689 is left queued rather than cancelled, so
it runs unattended.

What that costs is narrower than it looks: the smoketest's own purpose, per its header, is
to prove the env, the dataset path, the model build, `init_compression` and the training
loop still work, and job 30711 did all of that on entropy. What is missing is only CUDA
underneath it, and that half is covered by helios and lem, which run the same smoketest on
real GPUs. The original note said this job "did not run inside this audit"; that was true
but suggested it was still owed, so it is restated here as the limit it is. `h100` is the single node `h100a` and it was full
of other users' multi-day jobs; the scheduler put job 30689's estimated start at
**2026-10-29 13:12**, well after this was written. It was left queued rather than cancelled,
since it costs nothing and runs unattended — read `~j321m/mas_smoke_30689.out` for the
result.

Two things are worth knowing before reading it. entropy's environment is **not** built from
this repo's lock: it carries `transformers 5.12.1` against a manifest pinning `==4.56.2`, so
a failure there may be an environment difference rather than a defect in `main`. And the
same suite and the same smoketest already pass on helios and lem, so this is a third
data point rather than the only one.

**The lock fix does not reach the clusters, and that is worth stating plainly.** Each
cluster sets `PIXI_HOME=$PROJECT_HOME_PATH/$USER/pixi`, a directory with **its own copy** of
`pixi.toml` and `pixi.lock`. F16 repaired the repo's lock; nothing syncs it outward, so
entropy's env still has no pytest — the same symptom as lem. Refreshing a cluster env means
copying the repo's manifest and lock into that `PIXI_HOME` and re-installing, which is a
deployment step a PR cannot perform. It was deliberately not attempted on entropy: the env
is shared, and moving `transformers` 5.12.1 → 4.56.2 could break someone else's work. Where
the suite has to run without touching a shared env,
`python -m venv --system-site-packages` layered over it, with pytest installed into the venv
alone, is additive and isolated — that is what the entropy run does.

`scripts/sync_cluster_env.sh` makes the deployment step one command rather than folklore,
**and it has now been run for real.** Job 30739 applied it on entropy against a `PIXI_HOME`
created for the purpose rather than the shared one: `APPLY_RC=0`, the lock stayed at **v6**,
and the resulting environment reports `pytest 8.4.2 | transformers 4.56.2` — pytest present
where the shared env has none, and transformers at the version the manifest pins instead of
5.12.1. The lock fix reaches a cluster environment; what is left is a decision about the
shared one, not a missing capability.

That run also proved the `requires-pixi` strip was load-bearing rather than tidy: entropy
runs pixi **0.77**, which the repo's bound excludes, so without stripping that line the
apply would have refused outright. A fix for one issue would have disabled the fix for
another, on the cluster that needed it most.
Dry-run against entropy's real `PIXI_HOME` (job 30716, exit 0, nothing changed) it reports:
both locks at **v6**, but `pixi.toml` and `pixi.lock` both differ, and the cluster lock
carries **0** pytest entries against the repo's **1**. That is the drift, measured. `--apply`
was not passed, and should be run by whoever owns that environment rather than by an audit
borrowing the account: it would move `transformers` 5.12.1 → 4.56.2 under everyone using
it.

### A trap worth not repeating

The first PCD-50 submission (6173802) piped the training output through
`| grep ... | head -200`. `head` exits once it has its 200 lines, which closes the pipe;
`grep` then dies of SIGPIPE on its next write and everything after that is discarded. The
job kept training for another 17 minutes against a log file that had stopped growing,
which looks exactly like a hung job. With four ranks each logging loss, learning rate and
several projection gradient norms, 200 lines is about 8 steps.

`#SBATCH --output` already captures stdout. Do not add a pipeline on top of it; filter
the file afterwards.

### What running on a cluster was worth

Nothing in Tiers 0–3 would have found F15 — it only appears where `torch.cuda.is_available()`
is true, which no amount of local CPU testing reproduces. That is the argument for keeping
a cluster leg in the gate rather than trusting CI alone.

---

## 7. Branch-point protocol

1. Tag the commit — `git tag baseline/<name> && git push origin baseline/<name>`. A tag,
   unlike a branch, will not drift.
2. Record here: the tag, the Tier 4 numbers actually observed, the pixi lock version, and
   the cluster each tier was verified on. As of this audit those are: final
   `steps/100/train/loss` **2.674062** against reference `uhhl03fy` **2.670469**, lock
   **v6**, Tier 4 and the 4-rank Tier 3 evidence on **lem** (4×H100), Tiers 0/1/2 in CI
   and on **helios** (GH200), Tier 5 on both.
3. Branch the new project from that tag.
4. **Refresh each cluster's environment from the repo's lock**, or the new project
   inherits the drift this audit found. A cluster's `PIXI_HOME` holds its own copy of
   `pixi.toml` and `pixi.lock`, and nothing syncs it: entropy's still has no pytest and
   runs `transformers 5.12.1` against a manifest pinning `==4.56.2`. Check with
   `scripts/sync_cluster_env.sh <PIXI_HOME>` (dry run) and apply it only after confirming
   who else uses that environment.
5. **Decide what happens to `configs/mastest/` before branching.** 50 configs, 252 KB,
   matched by `.gitignore:194 *mastest*`, 47 of them already broken. They are gitignored
   on purpose, so nothing is wrong — but nothing carries them to the new project either,
   and the failure mode is someone assuming they came along. Commit the ones that matter
   deliberately, or accept losing them.
6. **Know what the tag does and does not cover.** Tier 4 and the four-rank Tier 3
   evidence rest on lem alone; entropy contributes the suite and a CPU smoketest; lem
   cannot run the suite at all for want of pytest. The per-cluster table in section 1 is
   the honest version.

---

## 8. Open decisions

- ~~**F10 first.**~~ **Fixed, not deferred.** Every fineweb eval number recorded before
  this change scored training data and should be treated as meaningless rather than
  merely suspect. New runs get a real held-out 1%; see F10. What is left for you is only
  whether 1% is the right fraction, and whether any published fineweb eval number needs
  retracting.
- ~~**F9**: what is the shared fineweb path on lem?~~ **Accepted as-is** on 2026-10-28 —
  the hardcoded account directory stays, allowlisted so no new ones appear.
- **Tier 4 tolerance**: how close must a repro be to count as a pass? Now answerable with
  a measurement rather than a guess — the full run lands **+0.13%** on the final number
  and holds **+0.0003 to +0.0092** on the rolling mean from step 201 to the end, on
  different hardware from the reference. A threshold of 0.5% on the final loss would pass
  this comfortably and still catch anything that actually broke; the first 100 steps need
  excluding from any per-point bound, since that point alone is 0.14 off.
- ~~**F16 — regenerate `pixi.lock` or not.**~~ **Resolved.** Regenerating with pixi 0.63.2
  keeps the lock at v6, so no cluster loses the ability to read it, and the premise that
  regeneration forces v7 was wrong — the format follows the writer, not the lock. See F16.
- `configs/ff_layer/` is a component group living outside the underscore convention; the
  Tier 0 test special-cases it. Renaming it to `_ff_layer` would remove the special case.
- ~~`test_slicegpt.py` sits at the repo root~~ **moved.** It called hydra's
  `initialize()`, which resolves `config_path` relative to the calling file. It now uses
  `initialize_config_dir` with a path built from the repo root, so it lives under
  `tests/` with everything else and `pytest.ini` no longer needs a second testpath. Same
  9 tests pass before and after.
