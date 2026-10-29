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
