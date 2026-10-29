# Fixing the review findings on PR #199

Seven findings came out of the `/code-review high` pass on PR #199 (2026-10-29). All
seven are in the work added in the last two days -- the `final_eval` / fineweb-holdout
changes -- and none were caught by the 113 tests, which is itself a finding about the
test gate.

This file is the plan. Each section says what is wrong, why the tests missed it, what
the fix is, and what proves it. Execution order is at the bottom: blockers first,
then the ones that only waste work, then hygiene.

Status column is updated as each is done.

| # | Finding | Severity | Status |
|---|---------|----------|--------|
| R1 | `final_eval` crashes on `eval_dataloader: None` | blocker | done |
| R2 | eval `sequence_length` pinned to 2048 breaks shorter runs | blocker | done |
| R3 | `max_samples` is per-rank/per-worker, not a fixed set | blocker | done |
| R4 | resume eval-skip loop is dead code | waste | done |
| R5 | `holdout_rows` not overridable, breaks small datasets | correctness | done |
| R6 | `n_steps: 0` finalize pass now reads a dataset it does not use | waste | done |
| R7 | stray tracked file `=0.60,` at the repo root | hygiene | done |

---

## R1. `final_eval` crashes on every config that switches eval off

**What is wrong.** In YAML a bare `None` is the *string* `"None"`; the null is `null`
or `~`. `final_eval`'s guard is

```python
if self.eval_dataloader is None or not hasattr(self.eval_dataloader, "__iter__"):
```

A string is not `None` and does have `__iter__`, so the guard passes, the loop iterates
the characters `'N'`, `'o'`, `'n'`, `'e'`, and `batch.to(self.device)` raises
`AttributeError: 'str' object has no attribute 'to'`.

`final_eval()` runs *before* `save_final_checkpoint()`, so an affected run trains every
step and then dies without saving. That is the worst possible place for it.

**How wide.** `grep -rn 'eval_dataloader:\s*None\s*$' configs/` returns **37 configs** --
every PC run config in the repo, including this PR's own `smoketest_pc_1gpu_helios` and
`smoketest_pc_finalize_helios`. Only `llama_8B_comp_memeff.yaml` writes real `null`.

**Why the tests missed it.** The trainer tests build the trainer in Python and pass a
real dataloader or a literal `None`. Nothing drives a trainer from a YAML config that
switches eval off, so the string never appears in a test.

**Why the smoketests missed it.** They ran before `final_eval` was added.

**Plan.**
1. Normalise once, in `__attrs_post_init__`: a `str` `eval_dataloader` becomes `None`.
   One place, covers all 37 configs and anything written later.
2. Then both guards collapse to a plain `is None` -- delete the `hasattr` duck-typing,
   which is what made a string look like a dataloader in the first place.
3. Rewrite the 37 configs to `null`, so the intent is honest at the config level too.
4. A repo-hygiene test that fails if any config reintroduces bare `None` for a key whose
   value is meant to be absent.
5. A trainer test that constructs with `eval_dataloader="None"` and asserts `final_eval`
   is a no-op rather than a crash.

**Risk.** Low. Step 3 is mechanical and step 1 makes it unnecessary for correctness.

---

## R2. Eval `sequence_length` is pinned to 2048 and breaks shorter runs

**What is wrong.** `configs/_dataset/fineweb.yaml` pins the eval dataset to
`sequence_length: 2048` while the model's RoPE sizes its `sin`/`cos` buffers from
`${common.sequence_length}` (`src/core/llama.py:146`, `length=seq_len`). `forward`
*slices* those buffers (`self.cos[: x.shape[-2], :]`) and never extends them, so feeding
2048 tokens to a model built for 1024 silently returns 1024 rows and broadcasting fails:

```
RuntimeError: The size of tensor a (2047) must match the size of tensor b (1024)
```

This PR's own `smoketest_fineweb_holdout_helios.yaml` runs at `sequence_length: 1024`
with eval on at interval 4, so it would die at step 4.

**Why I pinned it.** To keep `final_ce` comparable between runs that train at different
sequence lengths. That intent is right but unachievable this way: the model physically
cannot score a longer sequence than it was built for.

**Plan.** Inherit `${common.sequence_length}`, and say in the comment that the canonical
16,777,216-token figure is what you get at the project's standard 2048 -- a run at a
different length gets a different token count and its `final_ce` is comparable only to
other runs at that length. That is the honest version of the guarantee.

**Risk.** Low. Restores the behaviour every other dataset config already has.

---

## R3. `max_samples` is a per-rank, per-worker cap, not a fixed set

**What is wrong.** `AbstractDataset.__iter__` applies the cap *after* the per-rank
stride:

```python
if self.world_size_independent:
    stream = itertools.islice(self.sample_packer(), self.rank, None, self.world_size)
...
if self.max_samples is not None:
    stream = itertools.islice(stream, self.max_samples)
```

so each rank yields `max_samples` samples from its own strided substream. Across the job
that is `max_samples * world_size`.

It is also per DataLoader worker. `to_iterable_dataset` shards by worker -- verified
empirically, 64 rows over 2 workers gives 64 unique rows, not 128 -- so workers do not
*duplicate*, but each one still applies its own cap, so `num_workers: 2` doubles the set
again. The eval dataloader uses `num_workers: 2`.

Net: the "fixed 8192-sample" set is `8192 * world_size * num_workers`. This reintroduces
exactly the GPU-count data dependence this PR set out to remove, in the one metric whose
whole purpose is to be comparable across runs.

**Why the tests missed it.** The `max_samples` test runs at `world_size=1`,
`num_workers=0`, where all three numbers coincide.

**Plan.**
1. Move the cap *before* the stride. With `world_size_independent: true` every rank packs
   the same stream, so capping first and striding second makes the union across ranks
   exactly `max_samples` -- the same samples at any GPU count.
2. Set the eval dataloader to `num_workers: 0`. Per-worker sharding makes a global cap
   impossible to express inside `__iter__` without reimplementing HF's shard split; 8192
   samples read once at the end of a run does not need worker parallelism.
3. Make `final_eval` reduce correctly across ranks: accumulate a local loss *sum* and
   sample *count*, all-reduce both once after the loop, then divide. A mean of per-rank
   means is wrong when the split is uneven, and reducing inside the loop would deadlock
   on ranks that get a different number of batches.
4. Tests: `max_samples` holds at `world_size` 1/2/4 (the union is the same set), and the
   cap survives the stride.

**Risk.** Medium -- it changes which samples each rank sees. Mitigated by asserting on
the union rather than on any one rank's slice.

---

## R4. The resume eval-skip loop is dead code

**What is wrong.** `__attrs_post_init__` builds `self.eval_iterator` and winds it forward
by `n_skip_eval_batches` on resume. But `eval()` now re-creates
`self.eval_iterator = iter(self.eval_dataloader)` on every call, so the wound-forward
iterator is discarded unread.

The skip loop was right when eval carried one iterator across the whole run. Restarting
the stream every eval -- which is what makes the eval set fixed -- is what made it
pointless. It is not merely useless: on a long resume it pulls ~10k batches through the
tokenizer before training starts.

**Plan.** Delete the eager `iter()` and the skip loop, keep `self.eval_iterator = None`
as the initial value, and leave a comment saying the restart in `eval()` is what replaced
it, so nobody adds it back.

**Risk.** Low. Verify nothing else reads `eval_iterator` before `eval()` sets it.

---

## R5. `holdout_rows` is not overridable, so small fineweb copies fail

**What is wrong.** `fineweb.yaml` now sets `holdout: train` unconditionally but leaves
`holdout_rows` at the Python default of 262144. Because the key is absent from the YAML,
Hydra cannot override it with a plain `key=value`. Any sampled or local fineweb copy
(`_cluster/local.yaml` points both fineweb paths at `data`) raises at construction:

```
holdout_rows 262144 leaves no training rows in a dataset of N
```

**Plan.** Write `holdout_rows` explicitly into both dataloaders in `fineweb.yaml` so it
is a normal overridable key, and make the error message say what to do about it. Keep the
hard error -- a dataset too small to host the holdout genuinely cannot, and silently
shrinking the eval set would defeat the point of fixing it.

**Risk.** Low.

---

## R6. The `n_steps: 0` finalize pass now reads a dataset it does not use

**What is wrong.** `llama_1B_comp_finalize.yaml` loads c4 with the comment
`# not used - loads slower` and runs `n_steps: 0`. `final_eval()` is unconditional, so
that pass now actually reads c4 -- paying the load it was written to avoid, to produce a
`final_ce` for a model that did no training.

**Plan.** Set `eval_dataloader: null` in that config. Targeted, and with R1 in place it
makes `final_eval` a no-op there.

**Risk.** None.

---

## R7. Stray tracked file `=0.60,` at the repo root

**What is wrong.** An empty file literally named `=0.60,` is tracked at the repo root --
fallout from an unquoted pip version bound in a shell command.

**Plan.** `git rm` it. Nothing is lost; it stays in history.

**Risk.** None.

---

## Order of execution

Blockers first, each committed separately so a bad step reverts on its own. The full
suite runs after every step and the commit is gated on `PYTEST_RC=0`.

1. R7 -- stray file (no behaviour, clears the tree)
2. R1 -- `None` string (blocker, widest blast radius)
3. R6 -- finalize config (one line, depends on R1 being in)
4. R2 -- eval `sequence_length` (blocker)
5. R5 -- `holdout_rows` (correctness)
6. R3 -- `max_samples` + distributed reduce (blocker, most invasive, done with the most
   test support behind it)
7. R4 -- dead skip loop (cleanup, last so the diff is readable)

Then: full suite, `black --check`, config-load sweep over every config, and a push.

---

# Execution record

Done 2026-10-29, in the order above, each committed separately with the full suite
green before the commit. Suite went 115 -> 134 tests.

| Commit | Finding | What landed |
|--------|---------|-------------|
| `e95d7069` | R7 | `git rm '=0.60,'` and this plan |
| `9b09e612` | R1, R4 | str normalisation in `__attrs_post_init__`; 37 configs to `null`; skip loop deleted |
| `23523ef5` | R6 | `eval_dataloader: null` on the finalize pass |
| `295bc3fb` | R2 | eval `sequence_length` tracks `${common.sequence_length}` |
| `976114ba` | R5 | `holdout_rows` written out and overridable; actionable error |
| `12f8991b` | R3 | cap before stride; two guards; weighted all-reduced `final_ce` |
| `e357a30b` | — | local end-to-end smoketest for the whole path |

## Deviations from the plan

**R4 moved from last to alongside R1.** The R1 normalisation made the skip loop
unreachable (`eval_iterator` is `None` by the time the loop is reached), so leaving it
for later would have meant committing code that could not run, and its test
`test_resuming_with_eval_enabled_still_skips_the_eval_stream` asserted behaviour the R1
commit had already removed. Done together; the test now asserts the opposite.

**R1 was wider than the review said.** The review found 5 affected configs by reading
the diff. `grep -rn 'eval_dataloader:\s*None\s*$' configs/` finds **37** -- every PC run
config in the repo. That changed the fix: a code-side normalisation covering all of them
became the primary fix, with the config rewrite as hygiene behind it.

**R3's `num_workers` claim was wrong in the review, and the bug was real anyway.** The
review said workers *duplicate* the data. They do not: `to_iterable_dataset` shards by
DataLoader worker, verified empirically (64 rows over 2 workers gives 64 unique rows,
not 128). Each worker still applies its own cap, so the multiplication is real; the
mechanism is not duplication.

**A mistake made during R3, and the test added for it.** A new test reused the name of
an existing one (`test_final_eval_is_a_no_op_without_an_eval_dataloader`), which in
Python silently replaces it -- the older test stopped running, the count dropped by one,
and the suite stayed green. A duplicate `_RecordingLogger` helper went in the same way.
Both fixed, and `test_no_test_name_is_defined_twice_in_a_file` now fails on it.

**A speculative change was reverted.** R3 first wrapped the eval loop in try/finally to
restore the model's training mode afterwards. Nothing needs that -- `final_eval` runs at
the end of `train()` with only the save after it -- and it contradicted the existing
`test_final_eval_puts_the_model_in_eval_mode`. Dropped, keeping the diff on the finding.

## What is still not covered

The two end-to-end tests skip in CI, which runs with `HF_HUB_OFFLINE=1` and without the
untracked local `data` copy. They are a local gate. Every `@needs_local_data` test on
this branch has the same limit. Closing it needs a tracked fixture dataset and a
vendored tokenizer.

This matters, because it is the same gap that let all seven findings through: the suite
has 134 tests and, in CI, not one of them runs `main.py`.

---

# Cluster verification (entropy, 2026-10-29)

Run against `ebfe9d6c` from `~/mas_a100_stage/nano` as `j321m`.

| Job | What | Result |
|-----|------|--------|
| 30842 | full suite, h100a, shared `PIXI_HOME` | **`SUITE_RC=0`** — 133 passed, 18 skipped (`needs_local_data`/network) |
| 30836 | `final_ce`, **1 GPU**, A100 | `final_ce over 64 samples: 12.375` |
| 30837 | `final_ce`, **2 GPUs**, A100 | `final_ce over 64 samples: 12.375` on **both ranks** |
| 30839 | same, **pre-fix code**, 2 GPUs (control) | `final_ce over **128** samples` |
| 30840 | real PC smoketest, A100, `eval_dataloader: null` | 20 steps, `No eval dataloader; skipping the final eval.`, `rc=0` |

## What this settles

**R3 is proven on real hardware, and the fix is load-bearing.** 30836 and 30837 give the
same number over the same 64 samples at one and two GPUs, and both ranks report the
global figure, so the all-reduce works. 30839 puts the old cap-after-stride ordering
back on the same node and the same data and reads **128** samples — exactly
`max_samples * world_size`.

The control is worth reading carefully: the *loss value* is 12.375 in all three runs,
because the model is untrained and scores a near-uniform loss on any sample. Only the
*count* betrays the bug. On a trained model the extra samples move the number, and
nothing in the log says why. That is what made this worth catching with a test rather
than by eye.

**R1 is proven on the real PC path.** 30840 is the shipped PC smoketest, which carries
`eval_dataloader: null` where it used to carry the string. It reaches
`No eval dataloader; skipping the final eval.` and exits 0 — the exact point where the
pre-fix code raised `'str' object has no attribute 'to'`, after training and before the
checkpoint.

## A bug this found that the review did not

The first suite run on entropy (30835) failed two `test_sync_cluster_env.py` tests with
`cp: preserving permissions ... Operation not supported`. Not a test artefact: entropy's
`/tmp` refuses `cp -p`, which exits 1 *after* copying the content, and
`scripts/sync_cluster_env.sh` runs under `set -euo pipefail`. It aborted between the
lock copy and the manifest copy — the lock replaced, the manifest not, which is the one
state the backups exist to prevent. NFS with root_squash behaves the same way, and a
cluster `PIXI_HOME` often sits on exactly that.

Fixed in `ebfe9d6c` with a fallback to plain `cp`, and a regression test that puts a
`cp` on PATH behaving the way those filesystems do. Found only by running the suite on
a cluster; it passes on a workstation.

## Still open, unchanged and external

entropy pixi 0.77.0 (permission classifier refuses to replace `j321m`'s binary), lem
pixi 0.63.2 (`ui.wcss.pl` refusing SSH), and the `requires-pixi` bound that correctly
depends on both. Entropy's pixi still warns `the lock file ... uses an older format
(v6)` on every invocation, which is the bound doing its job.

## The 18 skips, closed (2026-10-29)

The first entropy suite run passed 133 and **skipped 18** -- every `@needs_local_data`
test, because the checkout there had no `data`/`data_eval` and the job's `HF_HOME` had
no GPT-2 tokenizer (only the Llama models). Skipped tests are not evidence, so the
cluster run was weaker than it looked.

Both are small. `data` and `data_eval` are 644 KB each, and `models--gpt2` is 2.8 MB.
Staged into `~/mas_a100_stage` and re-run:

| Job | Result |
|-----|--------|
| 30844 | **133 passed, 0 skipped**, `SUITE_RC=0` |
| 30845 | the end-to-end subset alone: **8 passed, 0 skipped** |

That second one matters most: `test_a_fineweb_run_reaches_final_ce_over_exactly_the_fixed_set`
and `test_a_run_with_eval_switched_off_still_finishes` drive the real `main.py`, and they
now have cluster coverage rather than only a workstation's.

They still skip in CI, which has neither the data nor the tokenizer. Making them run
there needs a tracked fixture dataset and a vendored tokenizer -- a larger change than
this branch should carry, and now the only remaining gap in their coverage.

To reproduce the staging:

```bash
rsync -azL ~/.cache/huggingface/hub/models--gpt2 entropy_j:<stage>/hf_cache/hub/
rsync -azL data data_eval entropy_j:<stage>/nano/
# then point the job's HF_HOME at <stage>/hf_cache
```

---

# A correction to an earlier claim, found 2026-10-29

While re-checking the three blocked pixi items I found that one of them was reported
done when it was not.

**I told you helios's pixi was upgraded 0.67.1 → 0.83.0. The binary that helios
experiments actually use is still 0.67.2.**

helios has two pixi binaries:

| Path | Version | On the experiment PATH? |
|------|---------|-------------------------|
| `~/.pixi/bin/pixi` | **0.67.2** (job 25363504) | **yes** — `configs/_cluster/helios.yaml:32` |
| `/net/storage/.../plggllmeffi3/pc/pixi/bin/pixi` | 0.83.0 (upgraded 11:24) | no |

The upgrade went to the second one. The verification job set its own PATH and picked it
up, so it reported 0.83.0 and I recorded the item as done. A version read is only
evidence if it comes from the PATH the real jobs use.

The irony is that the audit note explaining why entropy *cannot* be fixed states the
exact principle I then missed: *installing a pixi elsewhere does not change what that
account's PATH resolves to.*

## What follows

- **The `requires-pixi` bound's reason is still correct.** `.githooks/pre-commit` and
  `test_pixi_lock_stays_at_version_6` both say "helios runs pixi 0.67.1, which cannot
  read a v7 lock". That is accurate and operative. Nothing there needed changing — which
  it would have, had I acted on my own incorrect report.
- **helios's home pixi is deliberately left at 0.67.2.** At that version it cannot
  rewrite a lock to v7, which is a safety property. Moving it to 0.83.0 now would give
  that up and unblock nothing: lem's 0.63.2 is the binding constraint either way, and lem
  is unreachable. It should move together with lem.
- **A second trap, recorded so nobody repeats it:** helios's login node is x86_64 while
  its compute nodes are aarch64, and `~/.pixi/bin/pixi` is aarch64. On the login node it
  fails with `cannot execute binary file: Exec format error`, so any version check on
  helios has to run inside a job.

## The test added for it

`test_the_lock_constraint_is_stated_consistently` pins the three places that state the
lock rule — `requires-pixi` in `pixi.toml`, the pre-commit hook, and the CI assertion —
to one another, so relaxing any one of them without the others fails. It cannot check a
cluster, but it can stop the repo's own statements drifting apart, which is how a wrong
belief about a cluster turns into a wrong bound.

## Status of the three items, re-verified rather than restated

| Item | State | Evidence |
|------|-------|----------|
| helios pixi | **not done** — 0.67.2 on the experiment PATH | job 25363504 |
| entropy pixi | 0.77.0, permission classifier refuses `j321m`'s binary | read directly |
| lem pixi | 0.63.2, unreachable | `ssh: connect to ui.wcss.pl port 22: Connection refused` |
| `requires-pixi` + v6 assertion | correctly still in place | depends on all three |

## helios: done properly, 2026-10-29 14:11

`~/.pixi/bin/pixi` — the binary `configs/_cluster/helios.yaml:32` puts on the PATH — is
now **0.83.0**. Previous kept as `pixi.bak-20261029-141142`.

Done by copying the 0.83.0 aarch64 binary that already sat at the group-storage path
rather than running `self-update` over the network: it was already proven to run on a
GH200, so the swap carried no download and no new unknown.

Verified in job 25364846 by reproducing the config's own activation sequence rather than
a PATH of my own — which is the mistake that produced the wrong answer the first time:

```
resolved pixi: /net/home/plgrid/plgmstefaniak/.pixi/bin/pixi
version: pixi 0.83.0
PIXI_HOME lock BEFORE: version: 6
shell-hook OK
env OK: 2.6.0+cu126 | 4.56.2
PIXI_HOME lock AFTER:  version: 6
```

Worth noting from that log: at 0.83.0 `pixi shell-hook` **warns** about a v6 lock and
leaves it alone. The v7 rewrite comes from `pixi run`/`pixi install` without `--frozen`,
which `requires-pixi` already refuses on this repo. So the upgrade did not cost the
safety margin I had worried about.

### Remaining

| Item | State |
|------|-------|
| helios pixi | **done** — 0.83.0 on the experiment PATH (25364846) |
| entropy pixi | 0.77.0 — binary belongs to `j321m`; permission classifier refuses, and I am not routing around it |
| lem pixi | 0.63.2 — `ui.wcss.pl` refusing SSH all day |
| `requires-pixi` + v6 assertion | stays until lem moves; lem's 0.63.2 cannot read a v7 lock |

lem is the one that actually binds. entropy at 0.77.0 already writes v7, so it is bound
by `requires-pixi` rather than by the lock version; lem at 0.63.2 is the version that
would silently re-solve a v7 lock instead of failing, which is what the pin protects.

## lem, 2026-10-29 14:20: SSH came back, and the real blocker is different

`ui.wcss.pl` started answering again after refusing all day. The upgrade still did not
happen, for a reason nobody had seen yet.

**lem's home directory is over its disk quota and refuses every write.** `quota -s`
reports the home share past its limit, and a one-byte write into `~/.pixi/bin/` fails
with `Disk quota exceeded`.

`configs/_cluster/lem.yaml:45` resolves pixi through `$HOME/.pixi/bin`, so
`pixi self-update` has to write into that quota and cannot. This is not a pixi problem
and it is not SSH: it blocks anything that writes to home on lem, which includes
checkpoints and logs for any run configured to land there.

**This was already known, and I presented it as a discovery.** `configs/_cluster/lem.yaml`
says, in a comment written earlier in this very branch: *"$HOME on lem is a 20G-quota NFS
export and is currently over it"*. The condition is documented; what is new today is only
that it also blocks `pixi self-update`, and the next paragraph.

The largest single consumer is the experiment cemetery in `$HOME` -- and it is **dead
weight**. lem has redirected the cemetery to project lustre since that config comment was
written; the live cemetery there has entries from late October, while the one in `$HOME`
stops in early May. It is residue from before the redirect, not data anything still
writes to. Moving or removing it would clear the quota outright.

Clearing it is the account owner's call, so nothing was touched. Exact figures went to
them directly rather than into a shared repo.

**The 0.63.2 binary is backed up and ready** under the project lustre area at
`<project>/$USER/pixi_backup/pixi.bak-20261029-142010` (verified: runs, reports 0.63.2).
Once there is room in home, the upgrade is one command and the rollback is a copy.

## Where the four items actually stand

| Item | State | Blocker |
|------|-------|---------|
| helios pixi | **done, 0.83.0** | — (job 25364846) |
| entropy pixi | 0.77.0 | binary belongs to `j321m`; permission classifier refuses, not routed around |
| lem pixi | 0.63.2 | **home quota, 8.8 G over**; needs space freed, which is yours to do |
| `requires-pixi` + v6 assertion | in place, correctly | lem's 0.63.2 silently re-solves a v7 lock instead of failing |

lem is the item that actually binds the v6 pin. entropy at 0.77.0 already writes v7 and
is held off by `requires-pixi` rather than by the lock format; lem at 0.63.2 is the
version that would re-solve a v7 lock without saying so, which is what the pin exists
to prevent.


## entropy's cemetery: why it is the odd one out, written down

Chasing lem's quota turned up a trap next door. helios and lem both set
`cluster_cemetery_dir` to project storage and both explain why. entropy sets nothing and
explained nothing, so harmonising it looks like tidying.

It would break submission. entropy's `$PROJECT_HOME_PATH` is `/storage_nvme_4`, which is
mounted on **h100a and nowhere else** -- checked by running on a100a and asusgpu1, both of
which report no `/storage_nvme_4` and both of which see `$HOME`. A cemetery clone under it
would fail on every partition but h100.

`$HOME` on entropy is NFS, visible from all nodes, 11T with ~8T free and no tight quota --
which is what the cemetery actually needs. The reasons helios and lem move it away, an
inode quota on one and a 20G space quota on the other, do not apply.

The reasoning is now a comment in `configs/_cluster/entropy.yaml`. Nothing was changed
about where entropy's cemetery goes; only why it goes there is now recorded.

## The entropy upgrade is not actually required (2026-10-29)

I had been reporting two blockers to lifting `requires-pixi`. There is one.

The bound exists to stop a v7 lock landing in the repo while some cluster cannot read
v7. Which clusters cannot?

| Cluster | pixi | v7 lock? |
|---------|------|----------|
| helios | 0.83.0 | reads and writes it |
| entropy | 0.77.0 | **reads and writes it** |
| lem | 0.63.2 | **cannot** -- treats it as missing and silently re-solves |

entropy's own pixi settles its row. On the repo's v6 lock it prints:

```
WARN the lock file is up-to-date but uses an older format (v6),
     run `pixi lock` to upgrade to v7 for improved reproducibility
```

A pixi that recommends v7 is a pixi that handles v7. 0.77.0 is excluded by the bound
because it *writes* v7, which mattered only while a cluster could not read it.

**So the entropy upgrade -- the item blocked by the permission classifier, the one I
could not do -- is a nice-to-have, not a prerequisite.** Once lem is at 0.83.0, all three
clusters handle v7 and the bound can go, entropy untouched at 0.77.0.

That leaves exactly one thing between here and a clean baseline on this front:

**Free space in lem's `$HOME`.** The stale cemetery described above is larger than the
overage on its own, nothing has written to it since early May, and lem has been using
the lustre cemetery instead ever since. Then:

```bash
ssh lem '~/.pixi/bin/pixi self-update --version 0.83.0'
#   rollback if needed: copy back from <project lustre>/$USER/pixi_backup/
# then, as one repo change:
#   - widen or drop requires-pixi in pixi.toml
#   - drop test_pixi_lock_stays_at_version_6 in tests/test_repo_hygiene.py
#   - drop the matching check in .githooks/pre-commit
#   - update test_the_lock_constraint_is_stated_consistently, which pins all three
#     together and will fail if only some of them move
```
