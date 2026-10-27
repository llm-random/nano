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
| Tests in the suite | 11 | **63** (57 run in CI) |
| Test files | 2 | 7 |
| Runnable configs that compose *and* resolve | 76 / 82 | **82 / 82** |
| `src/core` → projected-compression references | 3 | **0** |
| CI | `black` only | `black` + the test suite |
| A bare `pytest` run | collection error | passes |

Tiers 0 and 1 are complete and enforced in CI. Tier 2 is done on CPU — including
resume-from-checkpoint, which found the most serious bug in this audit (F14). Tier 3 is
partly done on CPU, and Tier 5 partly done on helios.

Two corrections worth carrying forward, because both were wrong in an earlier draft and
both cost real coverage:

- **"No cluster access."** entropy is genuinely unreachable (`Permission denied
  (publickey)`; the key is offered and the server rejects it, so it needs authorising
  server-side). But **helios and lem both authenticate.** Checking only the cluster that
  had been named, and concluding there was none, was a mistake.
- **"Distributed cannot run without a GPU."** The framework is CUDA-gated, but FSDP2 is
  not: `fully_shard` works on a CPU mesh with gloo. That reaches `pc_finalize` and the
  sharded checkpoint path, neither of which had any test. See Tier 3.

What genuinely still needs a GPU cluster: more than one rank, NCCL, the bfloat16
mixed-precision policy, the 4-GPU world size, and **the numerical reproduction (Tier 4)**
— nothing here proves `main` still produces the paper's numbers.

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
| `test_slicegpt.py` | 9 tests, CPU, tiny model from the real configs |
| `smoke/` | SliceGPT end-to-end scenarios, cluster |

`test_slicegpt.py` was the pattern everything new follows: a tiny model composed from the
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
Hydra's strict loader rejects, so the config could not load at all. Both lines arrived in
the same commit, so neither was "the later one"; kept 512, matching sibling `llmr_800`.

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

**F16 — the test suite cannot run on lem.** Both pixi environments there
(`nano/$USER/pixi` and the older `plgllmefficont3/mstefaniak/pixi`) are at lock version 6
and **neither contains pytest**: the committed lock predates the `pytest` dependency that
`pixi.toml` declares. `python -m pytest` on lem gives `No module named pytest`, so only
the smoketest could run there. helios's environment does have it.

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

**F10 — on the same two lines, `eval_path_fineweb` points at the *train* directory.**
Identical to `train_path_fineweb`, ending `/train/train`, while c4 alongside it correctly
splits `/train` and `/validation`. If that is read as written, fineweb eval runs on
training data and the eval loss is meaningless — which would quietly flatter every
fineweb run. More serious than F9 and needs the same lem access to settle.

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
gloo provides DTensors, and `tests/test_distributed_cpu.py` covers the merge. A related
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

### Tier 2 — end-to-end — **partly done**

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

### Tier 3 — distributed — **partly done on CPU**

An earlier draft of this document said Tier 3 was impossible without a GPU. That was
wrong, and worth recording as a lesson: the *framework* is CUDA-gated, but FSDP2 is not.
`main.py:distributed_setup()` and `setup_distributed_training()` both check
`torch.cuda.is_available()`, and `setup_fsdp2_model` builds its mesh with
`init_device_mesh("cuda", ...)` — but `fully_shard` itself works on a CPU mesh with the
gloo backend and produces real DTensor parameters with a working `full_tensor()`.

`tests/test_distributed_cpu.py` uses that to reach two paths that had no test anywhere:

- **`pc_finalize`** — `ProjectedLinear.finalize()` calls `.full_tensor()` and reads
  `os.environ["RANK"]`, so it is unreachable unless the parameters are DTensors. This was
  recorded below as F13, "cannot run single-process on CPU"; that was also wrong. The test
  checks the merge directly: each layer must collapse to `P_out W P_in + auxiliary`.
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

Still needs a cluster: more than one rank, NCCL, the bfloat16 mixed-precision policy
production uses, the 4-GPU world size (still argued only by transitivity), FSDP2 2-GPU vs
1-GPU parity, and per-block gradient clipping.

### Tier 5 — cross-cluster — **partly done on helios**

entropy is unreachable (`Permission denied (publickey)`; the key is offered and the server
rejects it, so it needs authorising server-side). **helios and lem both authenticate** —
checking only the cluster that had been named, and reporting "no cluster access", was a
mistake.

Run on helios (GH200, `plgllmefficont3-gpu-gh200`): the branch checked out on group
storage, the test suite under the cluster's own pixi environment, and
`smoketest_pc_1gpu_helios.yaml` — a self-contained tiny PC model that exercises the pixi
env, the fineweb path, the PC model build, `init_compression` and the training loop on a
real GPU. See the execution log at the end.

Still open: lem, and the cemetery redirect end to end.

### Tier 4 — numerical reproduction — **prefix run submitted**

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

Also in scope for this tier and **not** done: re-running `run_eval.py` against the
committed `evals/evals/aggregated_results_llama1b.json`, and `benchmarks/bench_block.py`
against its committed `.jsonl`.

### Tier 5 — cross-cluster — **partly done on helios**

entropy is unreachable (`Permission denied (publickey)`; the key is offered and the server
rejects it, so it needs authorising server-side). **helios and lem both authenticate** —
checking only the cluster that had been named, and reporting "no cluster access", was a
mistake.

Run on helios (GH200, `plgllmefficont3-gpu-gh200`): the branch checked out on group
storage, the test suite under the cluster's own pixi environment, and
`smoketest_pc_1gpu_helios.yaml` — a self-contained tiny PC model that exercises the pixi
env, the fineweb path, the PC model build, `init_compression` and the training loop on a
real GPU. See the execution log at the end.

Still open: lem, and the cemetery redirect end to end.

### Tier 4 — numerical reproduction, the real acceptance gate

Nothing below this tier proves `main` still produces the paper's numbers.

| Run | Expected final loss |
|---|---|
| PCD-50 repro baseline | **2.6705** |
| no-P / W_b-only 90% | **3.2214** |
| HPD-90 | 3.2448 |
| PCD-90 | 3.1312 |

PCD-50 is the designated baseline and the natural pick. Prefer comparing the first ~100
steps of the loss curve at fixed seed over a single final number — a short-horizon prefix
must match far more tightly, and a final loss hides drift.

---

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

## 8. Cluster execution log

entropy is unreachable; **helios and lem both authenticate** as `plgmstefaniak`. The
branch is checked out on group storage on both (`baseline_smoke/nano`), with a hand-written
sbatch that reproduces the env from each cluster's own `_cluster/*.yaml` script block —
deliberately not through `run_exp.py`, whose `version_code()` commits onto the branch you
are standing on and leaves it broken if the job is killed.

| job | cluster | what it ran | outcome |
|---|---|---|---|
| 25254406 | helios GH200 | suite + `smoketest_pc_1gpu_helios` | smoketest **passed** (loss 14.375 → 9.875 over 20 steps on GPU); 5 PC unit tests **failed** |
| 25254605 | helios GH200 | one failing test, full traceback | `RuntimeError: indices should be either on cpu or on the same device as the indexed tensor` at `compression.py:99` |
| 25255266 | helios GH200 | suite + smoketest, after the device fix | PC tests **now pass on GPU**; smoketest trains (14.375 → 9.5625, rc=0); one new failure, `test_sharded_checkpoint_round_trips` |
| 25255801 | helios GH200 | that failure, full traceback | see below |
| 6173738 | lem H100 | suite + `smoketest_pc_1gpu_lem` (mem-eff path, real 1B) | smoketest **passed** on H100 (production `PCTrainer`, real 1B, rc=0); suite could not run — no pytest in the env (F16) |
| 6173802 | lem 4×H100 | PCD-50 prefix vs the reference curve | see below |

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

### What running on a cluster was worth

Nothing in Tiers 0–3 would have found F15 — it only appears where `torch.cuda.is_available()`
is true, which no amount of local CPU testing reproduces. That is the argument for keeping
a cluster leg in the gate rather than trusting CI alone.

---

## 6. Branch-point protocol

1. Tag the commit — `git tag baseline/<name> && git push origin baseline/<name>`. A tag,
   unlike a branch, will not drift.
2. Record here: the tag, the Tier 4 numbers actually observed, the pixi lock version, and
   the cluster each tier was verified on.
3. Branch the new project from that tag.

---

## 7. Open decisions

- **F10 first.** If `eval_path_fineweb` really is the train directory, every fineweb eval
  number on lem is suspect. Needs someone who can look at the filesystem.
- **F9**: what is the shared fineweb path on lem?
- **Tier 4 tolerance**: how close must the repro be to count as a pass?
- `configs/ff_layer/` is a component group living outside the underscore convention; the
  Tier 0 test special-cases it. Renaming it to `_ff_layer` would remove the special case.
- `test_slicegpt.py` sits at the repo root because it calls hydra's `initialize()`, which
  resolves `config_path` relative to the calling file. Switching it to
  `initialize_config_dir` with an absolute path would let it move under `tests/`.
