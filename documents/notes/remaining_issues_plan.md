# Plan: the issues left open after the baseline audit

Companion to `main_baseline_audit.md`, which covers the audit itself. This one takes the
list of bugs and config debt that audit left standing and says, for each, what the fix is,
what it risks and how it gets verified.

**Three items on the original list are already fixed** and are recorded here only so the
list reconciles with the report that prompted it:

| item | state |
|---|---|
| `run_exp.py` leaves the branch broken if killed | **fixed** — `restoring_repo_state()` handles SIGTERM/SIGINT, 4 tests |
| `save_to_llama_3_hf` writes no tokenizer | **fixed** — saves the tokenizer from `SOURCE_LLAMA_REPO` |
| `pixi run` rewrites `pixi.lock` v6 → v7 | **fixed** — `requires-pixi = ">=0.60,<0.77"`, mutation-tested |

What follows is what is actually left.

---

## 1. The lock fix does not reach the cluster environments

**What is wrong.** Every cluster sets `PIXI_HOME=$PROJECT_HOME_PATH/$USER/pixi`, a
directory with its own copy of `pixi.toml` and `pixi.lock`. The repo's lock is the source,
but nothing carries it across, so repairing the lock in a PR leaves every cluster
environment untouched. Measured on entropy (job 30716): both locks at v6, both files
differ, cluster lock has **0** pytest entries against the repo's **1**.

**Why it is not already done.** This is a deployment action, not a code change, and the
cluster environments are *shared*. Applying it on entropy would move `transformers`
5.12.1 → 4.56.2 under everyone using that account.

**Plan.**
1. `scripts/sync_cluster_env.sh` already exists and is dry-run verified against entropy.
   No further code is needed for the mechanism.
2. Add the one thing missing: a **test that the script keeps working**, since it is a
   shell script nothing exercises. Drive it against a temporary fake `PIXI_HOME` and
   assert the dry run reports a difference and writes nothing.
3. Document the per-cluster command in the audit doc's branch-point protocol so it is run
   at branch time rather than rediscovered.
4. **Not** run against any shared cluster env. That stays a human decision.

**Risk.** None for 1–3. Step 4 is deliberately out of scope.

---

## 2. GPU count silently changes training data

**What is wrong.** `world_size_independent: false` shards the corpus per rank, so a run is
reproducible only at a fixed `(seed, world_size)`. The eval side is fixed; the train side
is deliberate and stays.

**Plan.** Make the dependency visible instead of changing it: log `world_size` as run
metadata at startup, so the number is in the run record next to the seed rather than
inferable only from the SLURM script. Assert it is logged.

**Risk.** Low — one log line plus a test.

---

## 3. `configs/_model/llama_3_8_pc.yaml` is an orphan

**What is wrong.** Referenced by no config. Dead weight that still has to be maintained —
it was edited under two earlier issues purely for consistency.

**Plan.** Do **not** delete it: the user's standing instruction is not to delete things,
and an orphan model config is plausibly a staging post for work not yet landed. Instead:
1. Confirm it is unreferenced with a search over `configs/` and `src/`.
2. Add a test that lists orphaned `_model` configs and asserts the set has not *grown*,
   with the known ones allowlisted — the same shape as the account-name and eval-path
   guards. That stops new dead weight without deleting someone's work-in-progress.

**Risk.** None. Nothing is removed.

---

## 4. `full_save_checkpoints_path` logs `None/<jobid>/<task>/step_N`

**What is wrong.** `_build_save_path` (`main.py:41`) interpolates the literal string
`None` when `checkpoint.save.path` is null. Cosmetic — `_should_save_final_checkpoint`
guards on `save.path is not None`, so nothing is written to a path called `None`.

**Plan.** It is a log line that names a path which does not exist, which is the same class
as F17 (`save_training_state` logging the wrong directory) and misleads the same way.
Log that checkpointing is disabled instead of interpolating a path, and test that no log
line contains the literal `None/`.

**Risk.** Low — a log-only change, guarded by a test.

---

## 5. `configs/mastest/` never reaches `main`

**What is wrong.** 50 configs, 252 KB, matched by `.gitignore:194 *mastest*`; 47 are
already broken. Nothing carries them over.

**Plan.** Nothing to fix in code: a gitignored directory is working as intended. The risk
is a *silent* one — someone assumes those configs are on main. Record it in the
branch-point protocol as something to check deliberately before branching, and leave the
files alone.

**Risk.** None.

---

## 6. `configs/ff_layer/` sits outside the `_`-prefix convention

**What is wrong.** Component groups live in `_`-prefixed directories. `ff_layer` does not,
so `tests/test_repo_hygiene.py:142` special-cases it.

**Plan.** Renaming it to `_ff_layer` would remove the special case, but it is referenced
from model configs by path, so the rename touches every referencing config and would
collide with anyone's in-flight branch. The special case costs one line and a comment.
**Decide against renaming**, and make the test's comment say why it is special-cased
rather than leaving it looking like an oversight.

**Risk.** None — comment only.

---

## 7. `test_slicegpt.py` sits at the repo root

**What is wrong.** It calls hydra's `initialize()`, which resolves `config_path` relative
to the calling file, so moving it under `tests/` breaks it.

**Plan.** Switch it to `initialize_config_dir` with an absolute path — the pattern
`tests/test_core_framework.py` already uses — then move it under `tests/`. Verify by
running it before and after and comparing results.

**Risk.** Medium, and the highest of this set: it is someone else's test covering the
SliceGPT baseline. Mitigation: confirm it passes before the move, make the move
mechanical, confirm the same tests pass after. If it does not pass beforehand, leave it
where it is and record why.

---

## Order of execution

Cheapest and safest first, so that anything that goes wrong is isolated:

1. §6 comment (no behaviour)
2. §3 orphan guard (test only)
3. §5 branch-point note (docs)
4. §4 `None/` log line (small code + test)
5. §2 log `world_size` (small code + test)
6. §1 sync-script test + protocol note
7. §7 move `test_slicegpt.py` (riskiest, done last and verified both sides)

Each step: make the change, run the full suite, mutation-test anything new, commit
separately so a bad step is revertible on its own.
