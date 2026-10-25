# Task: merge `mas/nips` into `main` and delete the branch

Branch: `mas/nips` · Written: 2026-10-23 · Updated: 2026-10-25
Status: **all fixes done and all tiers verified. Nothing is merged yet.**

9 commits sit on `mas/nips-merge-prep` (worktree `/home/maciej/code/nano-merge`), on top of `mas/nips`
at `bced7b9d`. Tier 0, 1, 2 and 3 all pass — see *Execution log*. What remains is packaging, plus
**uncommitted work from a parallel session now sitting in both worktrees** (next section).


## What is left before we can open merge requests

### The working trees are not clean — read this first

Two sessions have been writing into these worktrees. Nothing below is committed anywhere, so a careless
`git commit -a` in either worktree would mix unrelated work into the merge.

**`/home/maciej/code/nano` — `mas/nips` @ `bced7b9d`**

| Path | What it is |
|---|---|
| `jupyter_notebooks/{mfu,pc_weights,polts}.ipynb` (D), `notebooks/no_p_90_loss_compare.ipynb` (D) + the matching untracked files under `documents/scripts/` | A **relocation**, not a deletion — all four are byte-identical at their new paths, so git records renames and the 31 MB `pc_weights.ipynb` blob is not duplicated |
| `documents/README.md` (M) | one line, part of the same relocation |
| `documents/notes/merge_to_main.md` (??) | this file |
| `documents/notes/claude-code-machine-setup.md` (??) | unrelated note |
| `documents/issues.md` (??), `documents/notes/cluster_fixes_lem_helios.md` (??) | **new, from the parallel session** |

**`/home/maciej/code/nano-merge` — `mas/nips-merge-prep` @ `1ae384d9`**

| Path | Group | What it is |
|---|---|---|
| `configs/_model/{llama_3_8_pc,llama_tiny_compressor,llmrandom/base_pc_model,smollm/base_pc_model}.yaml` | **A — repo bug** | `ProjectedEmbedding` still passed the pre-#147 nested `embedding:` arg; every non-memeff PC model built from these dies at instantiation. Broken on `origin/main` too |
| `configs/_cluster/lem.yaml`, `configs/_cluster/helios.yaml`, `configs/_misc/default.yaml` | **B — cluster env** | `cluster_cemetery_dir` redirect (both clusters' `$HOME` are over quota/inodes), `USER_HOME_PATH` made overridable, per-user `HF_HUB_CACHE` |
| `configs/pc_project_mas/smoketest_pc_1gpu_{lem,helios}.yaml` | **C — new** | 1-GPU PC smoketests; both already ran green (lem `6155713`, helios `25023297`) |

Group B's lem change **settles an open item above**: `$USER` cannot be substituted, because the lem login
is `plgmstefaniak` while the directory is `mstefaniak`. Declining to guess that was correct.

### Proposed split: three merge requests

Keeping these separate matters because only the first is verified end to end, and because B changes a
default that affects everyone.

| MR | Content | Verified? | Blocked by |
|---|---|---|---|
| **1 — main-readiness** | `mas/nips` (20 commits) + the 9 prep commits + the notebook relocation | **yes, fully** | the squash decision |
| **2 — PC config API drift** | group A, the 4 `_model/*` files | compose-checkable in minutes | nothing |
| **3 — cluster env** | groups B and C | needs a `run_exp.py` run per cluster | cluster time |

### Steps

**MR 1 — ready the moment the squash call is made**

1. Decide: squash the 9 junk commits (`dev`, `results`, `configs backup`, `revert configs`,
   `claude i love you`) or fast-forward as-is. **This must happen before anything below**, or the rewrite
   lands under freshly committed work and has to be redone.
2. In `mas/nips`: commit the notebook relocation — the 4 renames + the `documents/README.md` line. Commit
   `merge_to_main.md` and `claude-code-machine-setup.md` with it; leave `issues.md` and
   `cluster_fixes_lem_helios.md` for MR 2/3, since they describe that work.
3. Squash, if chosen.
4. Fast-forward the 9 prep commits from `mas/nips-merge-prep` onto `mas/nips`.
5. Re-run Tier 0 against the final history (`black --check`, `pytest`, imports, config compose) — the
   checks must be known to hold for exactly what is about to land, not for the tree they first ran on.
6. `git fetch origin main` and confirm it is still `d12c8f02` (static since 2026-04-14, but a six-month-idle
   branch is exactly when someone else pushes).
7. Open the MR. Afterwards, confirm every path from *Step 5 — merge the PC artifacts* is present on `main`,
   then delete `mas/nips` local and remote.

**MR 2 — group A, independent of MR 1**

1. Branch off `main` (or off `mas/nips` once MR 1 lands) and move just the 4 `_model/*` files onto it.
   They are currently stacked in the prep worktree on top of the 9 commits, so they must be lifted out
   deliberately — `git stash push -- <paths>` or `git diff -- <paths> > patch`.
2. Verify by composing the affected configs before and after: they should go from
   `TypeError: ProjectedEmbedding.__init__() got an unexpected keyword argument 'embedding'` to clean.
   `documents/issues.md` has the detail.
3. Open the MR.

**MR 3 — groups B and C**

1. Same lift-out onto its own branch.
2. Compose every config to prove the `${oc.select:...}` default in `_misc/default.yaml` resolves
   everywhere — it changes a value every experiment reads, so a silent failure here is expensive.
3. Re-run a `run_exp.py` submission on lem and on helios to prove the cemetery redirect works. Per
   `cluster_fixes_lem_helios.md`, **none of items 1-3 there has been executed**; the smoketests passed
   but the cemetery fix itself is unproven.
4. Open the MR.

### Decisions still outstanding

- **Squash or fast-forward** — blocks MR 1.
- **`pixi.lock`**: bump v6 → v7 as part of MR 1, or leave it? Commit 4 adds `pytest` to `pixi.toml`
  without touching the lock. The cluster independently prints the v6 warning on every run. A v7 lock
  with `pytest` already solved is saved in the session scratchpad.
- **`evals/`**: all 95 result files on `main`, or aggregated JSON + charts only? 2.9 MB either way.
- **`configs/mastest/`**: gitignored, so it will **not** reach `main`. If any of it matters for continuing
  PC it must be committed deliberately — deleting the branch does not lose it, but nothing carries it over.

## Goal and constraint

Everything needed to continue the PC project must end up on `main`, because `mas/nips` is deleted
afterwards — nothing can be left behind on the branch.

**Constraint:** `main` is the starting point for future *non-PC* projects. PC-specific logic must not sit
in the generic code paths (`src/core/`), where someone starting a new project would inherit it.

These two pull in opposite directions only for *code*. Data and paper artifacts live in their own
directories and cannot interfere, so they merge as-is.

## Starting state (verified 2026-10-23)

- `origin/main` is at `d12c8f02` *"Jk moe (#172)"*, 2026-04-14. `main` has not moved in six months.
- `mas/nips` is **20 commits ahead, 0 behind** `origin/main` → fast-forward, no conflicts.
- The local `main` ref is **stale**, one commit behind `origin/main` (`d25c326b`, 2026-03-25). Compare
  against `origin/main`; `git diff main..HEAD` wrongly attributes `Jk moe` (which adds `src/core/moe.py`)
  to this branch.
- 190 files, ~92k insertions, but **~88 % is data**: `evals/` 80,771 lines (2.9 MB on disk),
  `documents/` 3,385, `configs/` 2,906, `notebooks/` 2,080, `benchmarks/` 1,424, `src/` **1,058**.

## Findings

### 1. The PC code added to `src/core/trainer.py` is dead code

`Trainer.save_hf_checkpoint()` (30 lines, `src/core/trainer.py:394`) and the
`original_llama_path` field (`:56`) are unreachable on every real code path:

- `PCTrainer` (`src/projected_compression/trainer.py:128`) and `PCDistillationTrainer`
  (`trainer_distillation.py:133`) **both override `save_checkpoint` *and* `save_hf_checkpoint`**, with no
  `super()` call.
- **Every** config that sets `hf_only`/`nano_and_hf` targets `src.projected_compression.trainer`:
  `llama_1B_comp_memeff`, `llama_8B_comp_memeff`, `llama_1B_comp_memeff_distil`,
  `llama_8B_comp_memeff_distil`, `llama_8B_comp_memeff_test`. Nothing routes those save types through the
  bare core `Trainer`.
- The two implementations are not equivalent, confirming separate lineages: the PC one gathers from
  `self.model.target_model` and computes the projected embedding from `source_model`/`projections`
  (PC_memeff); the core one calls `finalize_projection_weights(self.model)` inside a
  `try/except AttributeError` (old-style comp / prune). **The core copy is the stale one.**

### 2. Deleting it does not cost PC any capability — checked explicitly

The core copy's comment claims it covers prune models. It does not; that intent was never wired up:

| Model family | `save.type` | Code path | Status after the change |
|---|---|---|---|
| HP / HPD (prune) | `huggingface` | `src/core/conversion_to_hf.py::save_to_llama_3_hf` | **untouched** |
| PC / PCD (memeff) | `hf_only` | `PCTrainer.save_hf_checkpoint` | **untouched** |
| old comp (`pc_finalize`) | `pc_finalize` | `Trainer.save_pc_finalized_checkpoint` | **untouched** (see 3) |
| — | `hf_only` via core `Trainer` | `Trainer.save_hf_checkpoint` | **deleted (unreachable)** |

HP and HPD models *were* evaluated in HF format — `evals/evals/evals_llama1b/` contains `llama1b_hp_*`
and `llama1b_hpd_*` results — and they reach HF through `save.type: huggingface`
(`configs/pc_project_mas/llama_1B_prune.yaml:46,49` and `llama_1B_prune_distillation.yaml:49,53`), which
this plan does not touch. So **no future PC workflow loses a route to HF**.

### 3. Correction to the earlier plan: leave the pre-existing `pc_finalize` coupling alone

An earlier draft proposed also moving the three *pre-existing* PC references out of `src/core/trainer.py`
(the `finalize_projection_weights` import, the `save.type == "pc_finalize"` dispatch, and
`save_pc_finalized_checkpoint`). **That is not safe.** Both configs using `pc_finalize` —
`configs/pc_project/llama_1B_comp_finalize.yaml` and `configs/mastest/old/llama_1B_comp_finalize.yaml` —
run on the **core `Trainer`**, not a PC subclass, and they are old non-memeff configs. Repointing them at
`PCTrainer` would break them, since `PCTrainer` assumes the memeff model structure
(`self.model.target_model`).

These three references predate this branch and are already on `main`. Leave them; record them as known
residual coupling. Removing them is a separate task with its own testing, not part of a merge.

### 4. The branch breaks the repo's only CI gate

`.github/workflows/black.yml` runs `black --check` on every push to `main` and every PR.
`origin/main` passes it completely: **36/36 tracked `.py` files clean**, `src/core/moe.py` included.
So the gate is green today and every failure below is introduced by this branch — 17 files:

- 8 from the PC work: `src/core/metric_loggers.py`, `src/core/trainer.py`,
  `src/projected_compression/{convert_memeff_to_hf,initialization,mem_eff,trainer,trainer_distillation}.py`,
  `tests/test_convert_memeff_to_hf.py`
- 7 from the 2026-10-20 commits: `benchmarks/*.py` (3), `documents/scripts/h_balanced_params.py`,
  `influences_figs/plotly_regen/*.py` (3)
- 2 others, **also new on this branch** (an earlier draft called these pre-existing; they do not
  exist on `origin/main` at all): `evaluate_hf_models.py`, `run_eval.py`

Formatting only, no logic.

### 5. The one test cannot be run from the project env

`tests/test_convert_memeff_to_hf.py` imports `pytest`, which is **not in `pixi.toml`**. It passes
(2 tests) in a throwaway env with the deps added, so the test is sound — it is just unrunnable via
`pixi run`.

### 6. Otherwise the code is clean

No personal paths (`/home/maciej`, `j321m`, `/storage_nvme`), no `breakpoint`, no `FIXME`, 2
commented-out lines, 3 `print`s (all in `run_eval.py`, a script), 2 honest `TODO: add scheduler saving`
notes in `mem_eff.py`.

One smell worth fixing while in the file — in the method being deleted anyway, so it resolves itself:
`try: finalize_projection_weights(self.model) except AttributeError: pass` would swallow a genuine
`AttributeError` from inside that function and turn a crash into a silently wrong checkpoint.

### 7. A one-character typo on this branch breaks every `lem` config (found during Tier 0)

`configs/_cluster/lem.yaml` line 1 read `" infrastructure:"` with a leading space. That pushes the whole
mapping one level deep, so `cluster_switch:` at column 1 becomes a second root node and PyYAML rejects the
file outright (*expected '<document start>'*). All **9** configs defaulting to the `lem` cluster were
impossible to compose:

```
pc_project_mas/{llama_1B,llama_8B}_comp_memeff
pc_project_mas/{llama_1B,llama_8B}_comp_memeff_distil
pc_project_mas/{llama_1B,llama_8B}_prune
pc_project_mas/{llama_1B,llama_8B}_prune_distillation
pc_project_mas/eval_hf_models
```

Introduced on this branch in `b1034ffc` *"configs backup"* — one of the commits Step 6 proposes squashing.
`origin/main` parses fine. Fixed in commit 5 below (whitespace only).

**Separate, not fixed — needs your call:** the same commit also replaced the user-generic
`export PIXI_HOME=$PROJECT_HOME_PATH/$USER/pixi` with a hardcoded
`/lustre/pd03/plgrid/plgllmefficont3/mstefaniak`, and changed `account` from `plgllmefficont2` to
`plgllmefficont3`. That personalises a *shared* cluster config which is about to land on `main`. Either
restore `$USER` or accept that `lem.yaml` on `main` is Maciej-specific.

### 8. `_build_save_path` logs a checkpoint path that does not exist (new on this branch)

`main.py::_build_save_path` (added on this branch) is logged to wandb as
`full_save_checkpoints_path` — the provenance metadata used to map paper cells back to runs
([[project-paper-run-provenance]]). For `save.type: huggingface` it appended a `step_N` level that is never
created: `train()` writes that type via `save_to_llama_3_hf()` straight to `get_full_checkpoint_path()`.

So **all 8 `huggingface`-type configs — the HP/HPD prune baselines among them — recorded a path one level
below the actual model.** Confirmed on a real run: logged `checkpoint/step_4`, model written to
`checkpoint/`. The other three types were already right, because `step_checkpoint_path()` wraps
`get_full_checkpoint_path()` and so keeps the SLURM job/array-task segment that `_build_save_path()` adds:

| `save.type` | actual location |
|---|---|
| `nano` | `{base}/step_N` |
| `hf_only` | `{base}/step_N/hf` |
| `pc_finalize` | `{base}/step_N` |
| `huggingface` | `{base}` ← was reported as `{base}/step_N` |

### 9. An unsupported save type was discovered only after training finished

Two failure modes in the same method:

1. `save.type=hf_only` on the core `Trainer` trained to completion and *then* raised — on a multi-hour job
   that loses the whole run. (Before this merge it did not even raise cleanly: `original_llama_path` was
   `None`, so it called `from_pretrained(None)` and died with a misleading *network* error.)
2. `save_checkpoint()` silently wrote **no model** for `huggingface` and `pc_finalize`. Those are written
   once by `train()` at the end, so a config with `save.interval > 0` got training-state files and no
   model at every interval, with nothing in the log to say so. No config in the repo hits this today —
   which is precisely why it was worth closing before someone starts a non-PC project from `main`.

`pc_finalize` additionally **cannot** be saved mid-training: `finalize_projection_weights()` merges the
projections into the weights in place and would corrupt the run. So it raises rather than dispatching.

Fixed with a `SUPPORTED_SAVE_TYPES` class attribute that the PC trainers widen, checked in
`__attrs_post_init__` and again in `save_checkpoint`. A typo'd save type now also fails at startup with a
`ValueError` naming the supported values, instead of training and writing nothing.

### 10. `lem.yaml` shadowed `PROJECT_HOME_PATH`; `slicegpt.md` contradicted itself

`lem.yaml` exported `PROJECT_HOME_PATH` twice with different values, so `HF_HOME` (set between the two)
resolved against the shared project dir while `HF_DATASETS_CACHE`, `PIXI_HOME` and the `XDG_*` paths
resolved against the personal one. The split is intended — shared model cache, personal pixi env — but
getting it by shadowing is a trap. Renamed the per-user path to `USER_HOME_PATH`; verified
behaviour-preserving by sourcing the script list before and after, all 10 variables byte-identical.

`documents/notes/slicegpt.md` said *"planning (nothing implemented yet)"* in its header while the body
documented a finished port whose one-shot 1B and 8B results are already in paper Table 1. Header corrected.

### 11. A too-short cosine schedule dies with a bare ZeroDivisionError (pre-existing)

`get_cosine_scheduler_with_warmup` sets `T_max = n_steps - warmup_steps - 1`. At
`n_steps == warmup_steps + 1` that is 0 and the first `scheduler.step()` raises `ZeroDivisionError` from
inside torch, with nothing in the traceback naming the config. Hit with `n_steps=2, warmup_steps=1` while
shrinking a config for a smoke test; no shipped config is affected (real runs are 4096/40). Now an assert
with an explanatory message, skipped for `n_steps == 0`, which is legitimate here — the conversion and
importances configs build a scheduler they never step. An unconditional version of that assert would have
**broken 12 working configs**; verified that of the 45 configs using this scheduler with concrete values,
all 45 pass. Commit 9, `1ae384d9`.

## Plan

### Commit structure and ordering

All fixing happens on `mas/nips` **before** the merge — `main` is never touched until Step 7. But the
commits are **not** all created at the end: they interleave with the fixes, in this order.

| # | Commit | Files | Must come |
|---|---|---|---|
| 1 | remove dead PC code from core | `src/core/trainer.py` | before 2 |
| 2 | `black` formatting only, no logic | 16 files | after 1 |
| 3 | add `pytest` to `pixi.toml` | `pixi.toml` | any time |

This table is the *original* 3-commit plan. Execution added 5 more for bugs found along the way (findings 7-10); the actual list is in the *Execution log*.

**Why 1 must precede 2:** both would otherwise touch `src/core/trainer.py`, tangling a logic change and a
formatting sweep in one diff that cannot be split by path.

Verified 2026-10-23: **all three** of black's complaints in `src/core/trainer.py` are *inside* the code
Step 1 deletes — the `get_model_state_dict, StateDictOptions` import, the local
`load_pc_state_dict_to_llama` import, and the `load_pc_state_dict_to_llama(...)` call. So after Step 1 the
file is already black-clean and drops out of the formatting commit entirely. The two commits then touch
**disjoint** file sets: 1 file vs 16. Doing it in the other order would mean reformatting 30 lines that the
next commit deletes.

Simulating the deletion also confirms the plan's verification target: the only `projected_compression`
reference left in `src/core/trainer.py` is the pre-existing `finalize_projection_weights` import from
finding 3, which this merge deliberately leaves alone.

Keep commit 2 free of any logic change so a reviewer can skip it, and add its SHA to a new
`.git-blame-ignore-revs` (the repo has none) so `git blame` stays useful across a 16-file sweep.

### Step 1 — make `src/core/` PC-free of the *new* coupling
1. Delete `Trainer.save_hf_checkpoint()` (`src/core/trainer.py:394-423`).
2. Delete the `original_llama_path` field (`:56`).
3. Delete the dispatch `if save_type in ("hf_only", "nano_and_hf"): self.save_hf_checkpoint(...)` (`:383-384`).
4. Drop the now-unused `get_model_state_dict, StateDictOptions` import (`:10`).
5. In core's `save_checkpoint`, make an unsupported save type **fail loudly** rather than silently
   no-op: raise `NotImplementedError` naming the PC trainers for `hf_only`/`nano_and_hf`. A future
   non-PC project then gets a clear message instead of a missing checkpoint.

Keep in `src/core/` — all generic, all valuable to any project:
- **the FSDP2 `clip_grad_norm_` fix** (`src/core/trainer.py:301`). Real bug: plain `clip_grad_norm_` on
  DTensor sharded grads computes only the local shard norm with no cross-rank reduction, underestimating
  the global norm by ~√world_size. Any FSDP2 project benefits.
- the SLURM-provenance logging in `src/core/metric_loggers.py` (job id, array task id, log path → wandb).
- the `main.py` checkpoint-path fix (`_build_save_path`).
- the `save_checkpoint` nano-path restructure.

### Step 2 — pass the CI gate
Run `black` on **16** files — the 17 from finding 4 minus `src/core/trainer.py`, which Step 1 already made
clean (see *Commit structure*). No carve-out is needed: `src/core/moe.py` is already black-clean both
here and on `origin/main`, so running black repo-wide touches exactly those 16 files and nothing of the
other team's ([[project-repo-scope]]).

### Step 3 — make the test runnable
Add `pytest` to `pixi.toml`, then confirm `pixi run python -m pytest tests/ -q` passes.

### Step 4 — smoke-test the changes

The change is a deletion justified by a static argument ("unreachable"). That argument has to be
confirmed at runtime, not only by reading. Tiered, cheapest first; **stop and fix at the first tier that
fails** rather than carrying on.

Note on environment: this machine has **no GPU** (`torch.cuda.is_available()` → False), so tiers 2–3 need
entropy. `main.py:118` has an explicit CPU fallback that skips distributed setup, which is what makes
tier 1 possible locally.

**Tier 0 — static, local, seconds.** No cluster, no GPU.
1. `black --check --extend-exclude='\.ipynb$' .` (the exact CI command) clean on all 46 tracked files.
2. `pixi run python -m pytest tests/ -q` → 2 passed (needs Step 3 done first).
3. Import every changed module, so a bad formatting edit or a stale import is caught immediately:
   `python -c "import src.core.trainer, src.projected_compression.trainer, src.projected_compression.trainer_distillation, src.projected_compression.initialization, src.projected_compression.mem_eff"`.
4. `grep -rn "projected_compression" src/core/` → only the pre-existing `finalize_projection_weights`
   import from finding 3.
5. Hydra-compose every PC config family and assert `OmegaConf.missing_keys(cfg)` is empty, registering the
   repo's custom resolvers first: `comp_memeff`, `comp_memeff_distil`, `prune`, `prune_distillation`,
   `pcd_*_noP`, `comp_finalize`, plus `tiny_local`. Catches a config that lost a key it needed.

**Tier 1 — CPU end-to-end, local, minutes.** Exercises the generic path a future non-PC project uses.
6. Run `tiny_local.yaml` (tiny model, dummy dataset, `save.type: huggingface`) with `n_steps` shrunk to
   ~5 and the stdout logger. It must train, save, and exit 0. This is the single most relevant test for
   the stated constraint: it proves the **core `Trainer` still works with no PC involvement at all**.
7. Because `dist.is_initialized()` is False on CPU, this covers only the non-distributed branch of the new
   `clip_grad_norm_` code. It still proves `clip_grads_with_norm_` is wired correctly and returns a norm.

**Tier 2 — single GPU on entropy, ~minutes.** One CPU-only `srun` first to check paths
([[reference-cluster-entropy]]), then 1 GPU; keep to 1–2 GPUs so the shared quota is not hogged.
8. A PC memeff config with `n_steps` tiny and `save.type: hf_only`, raising gradient accumulation to hold
   the per-GPU micro-batch. Assert the HF checkpoint directory **is actually written** — this is the direct
   test that deleting core's copy changed nothing, because the write must come from
   `PCTrainer.save_hf_checkpoint`.
9. A `prune` config with `save.type: huggingface`, confirming the HP/HPD → HF route (finding 2) still works.
10. A `comp_finalize` config with `save.type: pc_finalize`, confirming the path deliberately left in core
    (finding 3) is untouched.

**Tier 3 — multi-GPU on entropy, short.** The only tier that can exercise the cross-rank reduction.
11. Run one PC config on the **same GPU count as a known-good previous run** — required anyway, since the
    dataloader is world-size dependent ([[project-paper-run-provenance]]).
12. **Equivalence check, the real test of "the deleted code was inert":** same config, same seed, same GPU
    count, ~20 steps, before vs after the change. The loss trace and gradient norms should be
    **bit-identical**. Any divergence means the deletion was not inert and the premise of Step 1 is wrong.
13. Confirm the reported gradient norm is the *global* norm, not the local shard norm — the bug the FSDP2
    fix addresses. Sanity check: it should not be smaller than the old value by roughly √world_size.

**Record the results** in this note with run ids and dates, so the merge has evidence behind it rather
than an assertion.

### Step 5 — merge the PC artifacts as-is
`evals/`, `benchmarks/`, `configs/pc_project_mas/`, `documents/`, `influences_figs/`, `notebooks/` all go
to `main` in their own directories. They are inert files on PC-scoped paths and cannot affect a non-PC
project's code. `evals/` is 2.9 MB — large in line count, unremarkable in bytes.

### Step 6 — tidy the history (decide before merging)
15 of the 20 commits are pre-existing history; 9 are named `dev`, `results`, `configs backup`,
`revert configs` or `claude i love you`. A fast-forward puts those on `main` verbatim. Options:
**(a)** fast-forward and accept it; **(b)** squash the 9 into one or two described commits.
Recommend (b) — `main` is about to become the long-lived base for other projects, and this is the last
cheap moment to do it.

**If squashing, do it before commits 1–3**, not after, or the rewrite lands on top of work just
committed and has to be redone. Resulting commit count on `main`:
fast-forward 20 + 3 = **23**; squashed ~11–13 + 3 = **~14–16**.

### Step 7 — verify, then delete the branch
1. Re-run **Tier 0** of Step 4 on the final squashed/rebased history, so the checks are known to hold for
   exactly what is about to land — not just for the tree they were first run against.
2. Confirm the Step 4 tier results are recorded in this note, Tier 3's equivalence check included.
3. Confirm `origin/main` still has not moved (`git fetch origin main`; it has been static since
   2026-04-14, but a six-month-idle branch is exactly when someone else pushes).
4. Merge, confirm `main` contains every path from Step 5, then delete `mas/nips` — local and remote.

## Open items for Maciej

*Superseded by **What is left before we can open merge requests** at the top — kept for the reasoning.*

- **Step 6**: squash the `dev` commits, or fast-forward as-is? **Still the one blocking decision.**
- **`evals/`**: keep all 95 result files on `main` for reproducibility, or keep the aggregated JSON and
  charts only? Inert either way — this is a preference, not a blocker.
- ~~**`documents/notes/slicegpt.md`** says "planning (nothing implemented yet)"~~ — **fixed** in commit 8.
- **`pixi.lock`** — resolved as far as the working tree goes: it is **clean** now, no longer modified.
  But the committed lock is `version: 6` and pixi 0.81 reports it out-of-date and rewrites it to
  `version: 7` on any solve (~12k-line reformat, and it drops a `ca-certificates` entry). Commit 4 adds
  `pytest` to `pixi.toml` **without** touching the lock, so manifest and lock are formally out of step —
  though they already were, independently of this merge. **Your call:** bump the lock to v7 as part of
  this merge (forces everyone onto a newer pixi), or leave it and let `pixi install` re-solve. The
  generated v7 lock with pytest in it is saved at
  `/tmp/claude-1000/-home-maciej-code-nano/5cc471df-b3f1-49c3-b05f-7854868356ff/scratchpad/pixi.lock.v7-with-pytest`
  if you want it.
- ~~**`configs/_cluster/lem.yaml` personalisation**: restore `$USER`?~~ — **answered 2026-10-24 by the
  parallel session: `$USER` cannot be used.** The lem login is `plgmstefaniak` while the directory is
  `mstefaniak`, and sibling dirs mix both conventions. It is now overridable via an
  `if [ -z "$USER_HOME_PATH" ]` guard instead (uncommitted, group B).
- **`configs/mastest/`** (new): 252 KB, 50 configs, matched by `.gitignore:194 *mastest*`, so it is
  **local-only and will not reach `main`**. Finding 3 cited `configs/mastest/old/llama_1B_comp_finalize.yaml`
  as one of the two `pc_finalize` configs — only the `pc_project/` one actually merges. 47 of those 50
  configs are already broken in your working tree anyway (`MissingConfigException`). If any of it matters
  for continuing PC, it has to be committed deliberately, because deleting the branch will not lose it but
  nothing will carry it to `main` either.
- ~~**Validate `save.type` at startup**~~ — **done** in commit 6 (finding 9).
- ~~**`lem.yaml` hardcodes `mstefaniak`**~~ — resolved as above. The caution was justified: the lem
  username and the directory name genuinely differ, so substituting `$USER` would have broken it.
- ~~**`to_memory_proposal.md`** / **`to_memory_compiled.md`** untracked~~ — **gone** from the repo root
  as of 2026-10-23; nothing to do.

## Execution log — 2026-10-23

Executed in a separate worktree so `mas/nips` was never touched:

```
git worktree add /home/maciej/code/nano-merge -b mas/nips-merge-prep mas/nips
```

### Commits created (on `mas/nips-merge-prep`, 9 on top of `bced7b9d`)

| # | SHA | Commit | Files |
|---|---|---|---|
| 1 | `da7fecaf` | core: remove dead PC-specific HF checkpointing from Trainer | `src/core/trainer.py` |
| 2 | `1b793713` | style: black 25.1.0 formatting sweep, no logic changes | 16 |
| 3 | `148d0e8d` | chore: add `.git-blame-ignore-revs` for the black sweep | 1 (new) |
| 4 | `08f6c8d3` | pixi: add pytest so `tests/` can be run from the project env | `pixi.toml` |
| 5 | `82aad823` | configs: fix stray leading space that broke every lem config | `configs/_cluster/lem.yaml` |
| 6 | `1b62bc12` | core: validate checkpoint save.type at startup, not after training | 3 |
| 7 | `8409b06c` | main: report the real checkpoint path for save.type huggingface | `main.py` |
| 8 | `45ac3915` | configs+docs: stop shadowing PROJECT_HOME_PATH; correct slicegpt status | 2 |
| 9 | `1ae384d9` | core: assert instead of ZeroDivisionError on a too-short cosine schedule | `src/core/schedulers.py` |

Commit 1 removed 34 lines and added 11 (net -23). Exactly as predicted, it left `src/core/trainer.py`
black-clean, so that file dropped out of commit 2 and the two commits touch **disjoint** file sets (1 vs 16).

### Tier 0 — all pass

| # | Check | Result |
|---|---|---|
| 1 | `black --check --extend-exclude='\.ipynb$' .` | **46/46 clean** (was 16 failing) |
| 2 | `python -m pytest tests/ -q` | **2 passed** |
| 3 | import every changed module + `main` | OK |
| 4 | `grep projected_compression src/core/` | only the pre-existing `finalize_projection_weights` import; the one other hit is the PC trainer names inside the new error *message* |
| 5 | compose every config | 139/139 YAML files parse; **only** difference vs `mas/nips` is the 9 configs the lem fix repaired |

**Finding 1 and 2 are now verified empirically, not just by reading.** Composing all 71 composable configs
and pairing `trainer._target_` with `checkpoint.save.type`:

| Trainer class | `save.type` | count |
|---|---|---|
| `Trainer` | `nano` | 29 |
| `PCTrainer` | `nano` | 16 |
| `Trainer` | `huggingface` | 8 |
| `PCDistillationTrainer` | `nano` | 7 |
| `TrainerDistillation` | `nano` | 3 |
| `PCTrainer` | **`hf_only`** | **3** |
| `TrainerDistillation` | `huggingface` | 2 |
| `PCDistillationTrainer` | **`hf_only`** | **2** |
| `Trainer` | `pc_finalize` | 1 |

`hf_only` occurs 5 times and **always on a PC trainer**; `nano_and_hf` occurs **nowhere**; the plain core
`Trainer` only ever uses `nano`, `huggingface` or `pc_finalize`. The deleted method was unreachable from
every config in the repo, and the new `NotImplementedError` cannot fire for any of them.

Also confirmed: `pc_project/llama_1B_comp_finalize` is `Trainer` + `pc_finalize`, i.e. finding 3's reason
for leaving that path in core holds.

### Tier 1 — all pass

Ran `tiny_local` (core `Trainer`, no PC involvement, `save.type: huggingface`) for 5 steps on CPU:

- **exit 0**; trained, logged `train/loss` and `train/grad_norm` for all 5 steps, saved.
- `train/grad_norm` is populated (3.55 → 4.16), so the new `clip_gradient` returns a real norm on the
  non-distributed branch. The cross-rank reduction still needs Tier 3.
- Wrote a checkpoint that **loads back** as `LlamaForCausalLM`, 16,663,104 params (matching the trained
  model), and runs a forward pass. This is the HP/HPD → HF route of finding 2, proven working.
- **Equivalence check (the plan's Tier 3 item 12, done at CPU level):** same config, same seed, 5 steps,
  before (`mas/nips`) vs after (`mas/nips-merge-prep`) — loss and grad-norm traces **bit-identical**, and
  the saved `config.json`, `model.safetensors` and `generation_config.json` are **byte-identical by
  sha256**. On the non-distributed path the deletion and the reformat are numerically inert.
- **Positive test of the new failure:** `trainer.checkpoint.save.type=hf_only` on the core `Trainer` now
  raises the intended `NotImplementedError`. Under the old code the same run reached the dead
  `save_hf_checkpoint`, where `original_llama_path` was `None`, so it called `from_pretrained(None)` and
  died with a misleading *network* error (`We couldn't connect to huggingface.co`) **after** training. So
  the deleted code was not merely unreachable — on the only path that reached it, it was broken.

Reproduction detail worth keeping: these configs compose only via
`--config-path configs/<dir> --config-name <bare name>` (as recorded in `documents/notes/no_p.md:166`).
Composing them as `--config-name <dir>/<name>` resolves the `@_here_` package differently and fails with
`InterpolationKeyError: infrastructure.metric_logger.name`.

### Tier 2 / Tier 3 on entropy — setup and scope (2026-10-23)

Both trees were put on entropy with `git archive` rather than rsync, so each holds exactly the tracked
files of its commit and the two differ **only** by the 8 prep commits — verified on the cluster:
`diff -rq` reports exactly 22 differing paths (the 16 black files, `main.py`, `src/core/trainer.py`,
`pixi.toml`, `configs/_cluster/lem.yaml`, `documents/notes/slicegpt.md`, and the new
`.git-blame-ignore-revs`). Trees live at `~/tier_tests/{b,a}` on entropy; jobs run through
`~/tier_tests/job.sh`, which replicates the `_cluster/entropy.yaml` env block and the
`srun torchrun` call from `grid_generator/sbatch_builder.py`.

Two things found while setting up:

- **`/storage_nvme_4` is h100a-local.** a100a mounts only `/storage_nvme_1`, and the fineweb dataset
  (6.0 TB), `llama1b_in_nano.pt`, the importances *and* the 16 GB pixi env all live on `/storage_nvme_4`.
  So **PC jobs can only run on the h100 node** unless ~22 GB is first staged into shared `/home`. Worth
  knowing before planning any a100 run. ([[reference-cluster-entropy]])
- The `j321m` account is shared and in active use by other work (an mrunner `eval-lcb-olympiccoder` job
  holding 1 GPU + 90 GB), so free GPUs and free host RAM both move minute to minute. The QOS ceiling
  (`gres/gpu=8` on h100 and a100) is never the binding constraint; **host RAM is** — entropy defaults to
  `mem_per_gpu: 90G`, so two GPUs need 180 GB that is often not there.

**Item 10 (`pc_finalize`) deliberately not run, and why it costs nothing.** Comparing the two trees by
AST, `Trainer.save_pc_finalized_checkpoint`, `Trainer.train` and `Trainer.clip_gradient` are **identical**.
The only `pc_finalize` mentions in the diff are the new `SUPPORTED_SAVE_TYPES` tuple that *accepts* it, a
comment, and the new periodic-save guard. So the final-save route for `pc_finalize` is byte-identical
before and after, and a run could only characterise pre-existing behaviour. It would also need a legacy
non-memeff comp checkpoint — `configs/pc_project/llama_1B_comp_finalize.yaml` has `??? # CHANGE` for
`checkpoint.load.path` and the importances — and no such checkpoint exists on entropy (only memeff
`pc_v2_pcd_*` ones). The acceptance of `pc_finalize` by the new startup validation is covered by the
local matrix test instead.

The same AST check shows `clip_gradient` is unchanged by the prep commits, so Tier 3 item 13 measures the
branch's pre-existing FSDP2 fix, not my edits — still worth doing before it lands on `main`, which is what
the plan intends.

### Tier 2 on entropy — 2026-10-23, partly passed

Jobs run from `~/tier_tests/{a,b}` via `~/tier_tests/job.sh` (not `run_exp.py` — see below).

| # | Test | Job | Result |
|---|---|---|---|
| 8 | PC memeff + `save.type: hf_only` | 29456 | **PASS** |
| 9 | generic `huggingface` save route (substituted, see below) | 29461 | **PASS** |
| 9' | the 1B `prune` config itself | 29447/29448/29459 | **blocked — OOM at the 28 GB grant ceiling** |
| 10 | `pc_finalize` | — | not run, justified above |

**Item 8 — the direct test that deleting core's `save_hf_checkpoint` cost nothing.** 6 steps, loss
11.375 → 8.125, exit 0. The HF checkpoint was written to `out/t2_hfonly/step_5/hf/` (737 MB:
`model.safetensors`, both tokenizer files, both configs) — and it must have come from
`PCDistillationTrainer.save_hf_checkpoint`, because core's copy is gone in this tree. Loading it back:

```
LlamaForCausalLM: 188,791,232 params, layers=16 hidden=448 dff=1728 heads=32/8
forward OK, logits (1, 8, 128256); tokenizer OK, vocab 128000
```

188.8M params at dmodel=448 / dff=1728 is **exactly the paper's HP 90 % row**, so the saved model is
architecturally right, not merely a file that exists. Needed `model.cast_bfloat16=true` to fit
(`MaxRSS` 28.0 GB against a 28 GB grant), so its loss values are not comparable to the 4-GPU runs — this
is a save-path test, not a numerical one.

**Item 9 — substituted, and why.** The 1B `prune` config OOMs at the ceiling and has no memory lever
(`cast_bfloat16` is a PC-memeff-only option for computing the compressed matrix, not a general model
flag). What rescues the coverage: **`src/projected_compression/pruning.py` and
`src/core/conversion_to_hf.py` (which holds `save_to_llama_3_hf`) are unchanged by the entire branch**
versus `origin/main`, so the HP/HPD → HF route's own code is not at risk from this merge. `tiny_local`
(`Trainer` + `save.type: huggingface`) drives the identical save route at 16 MB and answers the one
question the merge created. **The prune-specific half remains unverified on entropy.**

**Item 9 also confirmed finding 8 on the cluster.** `tiny_local` with `n_steps=6` wrote
`out/t2_tiny_hf/{config.json,model.safetensors}` with **no `step_*` directory**. Before the fix,
`_build_save_path` would have logged `{base}/step_5` to wandb while the model sat at `{base}`. The bug was
real on entropy and the fix reports the true location. Item 8 confirms the other branch of the same
function from the opposite side: `hf_only` with `n_steps=6` did write `step_5/hf`, which is what
`_build_save_path` reports for that type and which the fix deliberately left alone.

### Tier 2 completed on entropy — 2026-10-24

**Item 9' (the real 1B `prune` config -> HF): PASS**, job 29744, 6 steps, loss 11.375 -> 8.5, exit 0.
Wrote `out/t2_prune_hf/{config.json,generation_config.json,model.safetensors}`, 361 MB, which loads as:

```
LlamaForCausalLM: 188,791,232 params, layers=16 hidden=448 dff=1728 heads=32/8 - forward OK
```

**Identical to the PC checkpoint from item 8** (same param count, same dims) — which is exactly what the
paper's param-matched HP-vs-PC comparison requires, and a good independent check that both save routes
produce the same architecture at 90 %. It also **reconfirms finding 8 from a third angle**: the model
landed at `{base}` with no `step_N` level, which is what the fixed `_build_save_path` now reports.

Two incidental notes:
- `save_to_llama_3_hf` writes **no tokenizer** (the load above fails only on `AutoTokenizer`), unlike
  `PCTrainer.save_hf_checkpoint` which saves one. Pre-existing and unchanged by this merge, but it means
  HP/HPD HF checkpoints need the tokenizer supplied separately for lm_eval.
- Getting here needed the framework's **own grid expansion**: `llama_1B_prune.yaml` uses `^n_steps`, and
  running `main.py` directly leaves that key in `cfg.trainer`, so `instantiate` raises
  `TypeError: Trainer.__init__() got an unexpected keyword argument '^n_steps'`. Hydra cannot delete it
  either — `~trainer.^n_steps` fails in the override lexer (`LexerNoViableAltException`). The fix was to
  call `grid_generator.generate_configs.create_grid_config` locally, dump the expanded config and point
  `main.py` at that, which is what `run_exp.py` does internally. **This is the one place where bypassing
  the launcher genuinely cost something.**

So Tier 2 is **complete**: items 8, 9 and 9' pass, item 10 is a justified skip.

### Tier 3 — item 13 verified; items 11 and 12 still blocked

**Item 13 (the gradient norm is global, not per-shard): PASS.** Entropy cannot give 2 GPUs before
Oct 25 (below), and NCCL cannot place two ranks on one GPU — `distributed_setup` hardcodes
`backend="nccl"`. But item 13's claim is arithmetic about `Trainer.clip_gradient`, which is in **core**
and model-agnostic, so it was tested directly: `torchrun` with gloo on CPU, parameters whose `.grad` are
real sharded `DTensor`s on a CPU device mesh, and the **real `Trainer.clip_gradient` method** called on a
stub holding `model`/`gradient_clipping`/`device` (the same technique used for `_validate_save_type`).

| world_size | returned by `clip_gradient` | true global norm | shard-only norm (the bug) | ratio |
|---|---|---|---|---|
| 1 | 11.045871 | 11.045871 | 11.045871 | 1.0000 |
| 2 | 11.045871 | 11.045871 | 8.114388 | 1.3613 |
| 4 | 11.045871 | 11.045871 | 5.102412 | 2.1648 |

The returned norm is **invariant across world sizes and equals the true global norm**, which is exactly
what correctness requires here — the full gradients are identical in all three runs. The shard-only norm,
which is what plain `clip_grad_norm_` would have reported, decays as ~1/sqrt(world_size). So the
cross-rank reduction is demonstrably happening and the underestimate the fix targets is real and would
have been ~2x at the paper's 4-GPU world size. Script:
`scratchpad/t3_clipnorm.py`. (The script prints FAIL at world_size 1 because its assertion requires
shard/global divergence, which cannot exist with one shard; `returned == global` still holds there.)

**What this does not cover:** real FSDP2 on GPU with the NCCL backend, the PC memeff model's own
projection-gradient path (`pass_gradient_to_projections`, which computes norms via
`mem_eff.get_global_grad_norm`, a different function), and items 11 and 12.

**Item 12 (A/B equivalence): PASS — 2026-10-25.** Jobs 29485 (`before` = `mas/nips`) and 29486
(`after` = prep branch), 2 GPUs each, `n_steps=20`, identical config and seed.

The plan's phrasing — *"the loss trace and gradient norms should be bit-identical"* — turned out to be the
**wrong acceptance criterion for this hardware**: the raw traces were not identical (step 0 grad norm
89.39971 vs 89.39948, growing to ~1 % by step 10). So a control run was added, **29743**, re-running the
*same* tree `b` a second time, to measure the run-to-run nondeterminism floor before drawing any
conclusion. Per metric family, over 20 steps:

| metric | same tree (b vs b2) | cross tree (b vs a) |
|---|---|---|
| `train/loss` | 20/20 identical | **20/20 identical** |
| `train/distill_loss` | 20/20 identical | **20/20 identical** |
| `train/total_loss` | 20/20 identical | **20/20 identical** |
| `train/lr` | 20/20 identical | **20/20 identical** |
| `train/grad_norm` | 1.87e-2 | 2.08e-2 |
| `train/total_grad_norm` | 1.52e-2 | 3.78e-2 |
| `train/projection_grad_norm` | 5.69e-2 | **2.38e-2** |
| `train/projection_grad_norm_head` | 2.18e-2 | 3.89e-2 |
| `train/projection_grad_norm_embedding` | 2.28e+0 | **5.92e-1** |
| `projection_grad_norm_block_*` (320 pts) | 1.95e+0 | **5.04e-1** |

**Every loss metric is bit-identical across the two trees**, and every gradient norm differs by the same
order of magnitude whether the tree changes or the identical code is simply run again — for the noisiest
metrics the *cross-tree* difference is 2-4x **smaller** than the same-tree difference. Overall max
relative delta: 2.283 same-tree vs 0.592 cross-tree, a ratio of 0.26.

**Conclusion: the 8 commits are numerically inert.** The runs are not reproducible run-to-run on this
hardware (nondeterministic reduction order), so "bit-identical" was never achievable for the norms; the
control is what makes the comparison meaningful. One caveat kept honest: the losses are logged at coarse
precision (values like 11.4375, 9.625), so their bit-identity is weaker evidence than it looks — the
load-bearing result is the cross-tree-vs-same-tree ratio.

This is worth remembering for any future A/B in this repo: **always run the same config twice first**, or
a nondeterministic norm will be mistaken for a regression.

**Item 11: cancelled by Maciej's decision, 2026-10-25** (job 29487, never ran). Accepted as covered,
and the reasoning holds up: item 12 shows the 9 commits are numerically inert versus `mas/nips`, and
`mas/nips` demonstrably runs at 4 GPUs — that is how the paper's runs were produced. So 4-GPU behaviour
follows by transitivity from "unchanged code" plus "the unchanged code works at 4 GPUs". The 2-GPU runs
had already shown multi-rank FSDP2 working end to end with the changes in place, which was the part that
actually needed observing. The only thing not directly observed is the production world size itself.

**Original queueing note, 2026-10-23 17:30.** They need >= 2 real GPUs, and every partition
fails on at least one count: h100 and a100 each had exactly **1** GPU free; `common` has 24 idle titan_v
but the torch build is `2.7.1+cu128` with arch list
`['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120','compute_120']` — **sm_70 (Volta) is not in it**, so
those GPUs would fail with "no kernel image available"; `common`'s only compatible GPUs (rtx_2080_ti,
sm_75) are 16/16 allocated; and NCCL cannot co-locate two ranks on one GPU (`distributed_setup` hardcodes
`backend="nccl"`). Rather than wait, the jobs are **queued** so SLURM starts them as the node drains:

| Job | Name | What it settles | Shape | SLURM start |
|---|---|---|---|---|
| 29484 | `t2_prune90` | **item 9'** — the real 1B `prune` config with `save.type: huggingface`, at 90 GB so it does not OOM | 1 GPU x 90 GB | Oct 25 ~09:54 |
| 29485 | `t3_ab_b` | **item 12** arm *before* (`mas/nips`) | 2 GPU x 90 GB | Oct 25 ~11:57 |
| 29486 | `t3_ab_a` | **item 12** arm *after* (`mas/nips-merge-prep`) | 2 GPU x 90 GB | Oct 25 ~11:57 |
| 29487 | `t3_ws4_a` | **item 11** — runs at the production 4-GPU world size, config exactly as written (`gacc 8`) | 4 GPU x 90 GB | Oct 26 ~09:57 |

Logs land in `~/tier_tests/logs/<name>.out` on entropy. Both A/B arms use `n_steps=20`,
`warmup_steps=5` (so the cosine phase gets `T_max=14` and both LR phases run), `save.path=null` (the
traces are what matter, not GBs of checkpoints) and `gradient_accumulation_steps=16`, which keeps the
per-GPU micro-batch at the production value of 16 at world_size 2. The two arms are identical in every
respect but the tree, and the fineweb seed is fixed at 123, so the comparison is valid.

**To read the result:** extract `train/loss` and `train/grad_norm` from both logs and diff them. They
should be **bit-identical**; any divergence means the deletion was not inert and Step 1's premise is
wrong. Current holders' end times: `mateuszwys` 4 GPUs until Oct 26 09:57, `j321m`'s own
`eval-lcb-olympiccoder` until Oct 25 11:57, `hegel` x2 until Oct 25 09:54 / 11:25.

**Entropy capacity, measured with `sbatch --test-only` (probes the scheduler without submitting):**

| Request on h100a | Would start |
|---|---|
| 1 GPU, ≤ 28 GB | immediately |
| 1 GPU, ≥ 30 GB | 2026-10-25 09:54 |
| 2 GPUs, **any** memory down to 8 GB | 2026-10-25 09:54 — it is the GPU count, not RAM |
| 2-4 GPUs on a100 | 2026-10-24 18:05 (but a100 cannot see the data — see above) |

`j321m`'s QOS ceiling (`gres/gpu=8`) is never the constraint; **host RAM is**. Entropy's own default of
`mem_per_gpu: 90G` is far past the cliff, which is why the real 4-GPU runs need an emptier node. Three
runs were lost to this before the shape was understood (two OOM at 24 GB, one deferred ~41 h and
cancelled), plus one to a bad `n_steps=2` override of mine (finding 11).

**Launcher note.** These tests do not use `run_exp.py`. Its `version_code` versions the *current working
tree*, commits untracked files and pushes to the cemetery repo, so it cannot run two fixed commits side by
side — which an A/B needs. `job.sh` instead replicates the env block of `_cluster/entropy.yaml` and the
`srun torchrun` call of `grid_generator/sbatch_builder.py`. The only launcher argument omitted is
`+checkpoint_config.slurm_array_task_id`, and nothing reads it as a config key: `main.py:75`,
`src/core/checkpointing.py:68` and `src/core/metric_loggers.py:179` all read `SLURM_ARRAY_TASK_ID` from
the environment directly. So jobs are launched the same way real runs are.



### Second round — bugs found and fixed after Tier 1 (findings 8-10)

Triage first: composing every config on a real `origin/main` worktree and diffing the failure sets showed
the 38 remaining failures are **byte-identical** between `origin/main` and this branch. So this branch
introduces **zero** new config failures, and fixing those 38 belongs to a separate task — the same
reasoning the plan applies to `src/core/moe.py`. The `lem.yaml` typo was the only config regression.

Then the four bugs of findings 8-10, each verified:

| Check | Result |
|---|---|
| `hf_only` on core `Trainer` | fails with **0 training steps logged** (was: 5 steps, then a misleading network error) |
| typo'd `save.type=nanoo` | fails at startup, `ValueError` naming the 3 supported types |
| `_build_save_path` per type | `huggingface`→`checkpoint`, `nano`→`checkpoint/step_4`, `hf_only`→`checkpoint/step_4/hf`, `pc_finalize`→`checkpoint/step_4` — all now match where the model is written |
| every config's startup validation | **0 of 71** composable configs would newly fail; `SUPPORTED_SAVE_TYPES` did not leak into the attrs constructor signature |
| `lem.yaml` env resolution | all 10 exported variables byte-identical before vs after |
| Tier 1 re-run | exit 0, loss/grad-norm trace **still bit-identical** to pre-change `mas/nips`, `model.safetensors` sha256 unchanged (`63b9d215…`) |
| Tier 0 re-run | black 46/46, pytest 2 passed, imports OK, 139/139 configs parse |

---

## Lem / Helios health check + 1-GPU PC smoketests (2026-10-23)

Question asked: do we still have access to lem and helios, and do experiments still run there?
Answer: **yes on both, but neither worked out of the box.** Three cluster-side blockers (two on lem,
one on helios), plus two repo bugs that are not cluster-specific at all -- both already broken on
`origin/main`, so they would fail on entropy or locally too. All five fixed in the
`mas/nips-merge-prep` worktree. The repo bugs surfaced here only because helios has no Llama-1B PC
assets, so its smoketest had to use `llama_tiny_compressor`; lem ran `projected_mem_eff_1B`, which
was never affected. Both smoketests were launched with the real `run_exp.py`
(cemetery branch -> clone -> tmux -> `sbatch`), not a hand-rolled script.

### Results

| | lem | helios |
|---|---|---|
| job | `6155713` on `r10-5` | `25023297` on `x1002c2s6b0n0` |
| config | `configs/pc_project_mas/smoketest_pc_1gpu_lem.yaml` | `..._helios.yaml` |
| what | real Llama-1B PC **mem-eff** (`PCTrainer`) @90%, 1xH100 | tiny PC (`llama_tiny_compressor`), 1xGH200 |
| state | COMPLETED 00:29:16, exit 0 | COMPLETED 00:10:37, exit 0 |
| steps | 10/10 | 20/20 |
| loss | 11.375 -> 8.000 | 22.000 -> 9.562 |
| grad norm | 90.5 -> 1.46 | 29.4 -> 7.50 |
| wandb | `ideas_cv/llm-random-test/p9ha3yel` | `.../t1oqkv3y` |

lem also logged the per-block mem-eff timings (`bwd=0.147s opt+d2h=0.052s total=0.200s`) and
`projection_grad_norm_block_{0..15}` + `_embedding` + `_head`, i.e. the full PC instrumentation.

Grant status: helios `plgllmefficont3` active to 2027-03-11, 11 251/100 000 GPU-h used; lem account
`plgllmefficont3`, no hour quota reported.

### Blockers found -- cluster-side (1-3)

**1. lem `$HOME` is over quota.** `quota -s` shows 28 822M used against a 20 480M limit, grace expired.
Every `git clone` for a new experiment dies with `fatal: ... Disk quota exceeded`, which `run_exp.py`
surfaces only as `invoke.exceptions.UnexpectedExit ... Exit code: 128`. `~/llmrandom_cemetery` alone is
13G across **253** clones since April. Worked around per-config with
`infrastructure.cemetery_experiments_dir` pointed at lustre; nothing deleted. **The clones still need
pruning by a human** -- that is the actual fix.

**2. lem HF cache is not usable by us (fixed in `_cluster/lem.yaml`).**
`HF_HOME=$PROJECT_HOME_PATH/hf_cache` -> `/lustre/pd03/.../nano/hf_cache`, owned by `plgggwozdz02`,
mode `drwxr-sr-x` (no group write), and its `hub/` holds only gpt2. So `llama_tokenize_fn` fails with
`OSError: PermissionError at .../models--meta-llama--Llama-3.1-8B ... Check cache directory permissions`
before step 1 -- a message that reads like a stale lock but is really "you cannot write here".
Fix: one line, `export HF_HUB_CACHE="$USER_HOME_PATH/hf_cache"`, which is per-user and writable and
already holds the Llama-3.1-8B tokenizer snapshot.

**3. helios `PIXI_HOME` does not exist.** `_cluster/helios.yaml` sets
`PIXI_HOME=$PROJECT_HOME_PATH/$USER/pixi` (commit `7e1e115` "user dependent pixi paths"), but
`$PROJECT_HOME_PATH` has **no per-user directories at all** -- so this is broken for every member, not
just us, and nothing has run on helios since 2026-04-18. The working shared env is at
`$PROJECT_HOME_PATH/pixi` (aarch64, python 3.12, torch 2.6.0+cu126 -- exactly what `pixi.toml` pins for
`linux-aarch64`). Symlinked `$PROJECT_HOME_PATH/plgmstefaniak/pixi -> ../pixi` so the config resolves.
A real per-user install would be `python update_pixi.py --config-path ... --config-name ...`.
Note the helios **login node is x86_64** while compute is aarch64, so the shared `pixi` binary cannot be
run on the login node at all (`Exec format error`) -- only inside an `srun`.

### Repo bugs found along the way (4-5) -- NOT cluster-specific

**Two PC config API drifts, across 4 files.** Both are broken on `origin/main`, so they are not a lem
or helios problem and not something this branch introduced; they would fail on any cluster.
Only the helios smoketest hit them, because it was the one using `llama_tiny_compressor`.
These are instantiation-time failures, which is why the earlier full *compose* sweep did not catch them.

| config | rot | since |
|---|---|---|
| `_model/llama_tiny_compressor.yaml`, `_model/llmrandom/base_pc_model.yaml`, `_model/smollm/base_pc_model.yaml`, `_model/llama_3_8_pc.yaml` | pass a nested `embedding: {_target_: ...Embedding, ...}` to `ProjectedEmbedding`, which takes flat `num_embeddings`/`embedding_dim` | `d66a3510` "emb.emb fix (#147)" updated only 2 of the 6 call sites |
| `_model/llama_tiny_compressor.yaml`, `_model/llama_3_8_pc.yaml` | call `init_compression(dmodel=, dff=)`; the signature is `(model, dimensions_importances_path, target_dmodel, target_dff)` | same era |

Verified after the fix by instantiating every `_model/**.yaml` on a meta device: **20/20 concrete model
configs instantiate**, including smollm 360/1700, llmrandom 300/800, llama tiny/1B/8B PC and mem-eff.
The 4 remaining non-instantiations are base files whose `???` placeholders are filled by their children.
No config census change: `llama_3_8_pc.yaml` is referenced by nothing, and the only user of
`llama_tiny_compressor.yaml` (`pc_project/llama_tiny_comp.yaml`) was already in the failing set for its
own `learning_rate: ???`.

### Other observations

- **First-batch latency dominates a short run.** Both jobs sat at 0% GPU for ~20 min (lem) / ~8 min
  (helios) after "Model initialized", filling `FineWebEduDataset.BUFFER_SIZE = 10000` shuffle buffer
  through `llama_tokenize_fn`. The 30-minute walltime was nearly too short for a *10-step* job. Budget
  for this when sizing smoke tests; it is startup, not a hang.
- **`full_save_checkpoints_path` logs `None/<jobid>/0/step_N`** when `checkpoint.save.path` is null.
  Cosmetic only -- `_should_save_final_checkpoint` guards on `save.path is not None`, so no `None/`
  directory is ever created. Not fixed.
- `pixi run` still rewrites `pixi.lock` v6 -> v7 on every invocation; restored each time with
  `git show HEAD:pixi.lock > pixi.lock`. Same open item as before.

### Files changed (uncommitted in `nano-merge`)

```
 M configs/_cluster/lem.yaml                   (+5)   HF_HUB_CACHE
 M configs/_model/llama_3_8_pc.yaml                   ProjectedEmbedding + init_compression
 M configs/_model/llama_tiny_compressor.yaml          ProjectedEmbedding + init_compression
 M configs/_model/llmrandom/base_pc_model.yaml        ProjectedEmbedding
 M configs/_model/smollm/base_pc_model.yaml           ProjectedEmbedding
 ?? configs/pc_project_mas/smoketest_pc_1gpu_lem.yaml
 ?? configs/pc_project_mas/smoketest_pc_1gpu_helios.yaml
```

Cluster-side, additive only: `$PROJECT_HOME_PATH/plgmstefaniak/pixi` symlink on helios, and
`.../plgmstefaniak/pc_assets/tiny_random_dimensions_importances.pt` (seeded random, 512 / 4x2048,
matching `llama_tiny_compressor`'s base dims -- helios has no real PC assets).
