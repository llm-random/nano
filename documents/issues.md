# issues.md

Known repo-level problems: things wrong with the code or configs themselves, as opposed to
cluster or environment trouble, which belongs in `notes/`. Each entry says how it was found,
what it breaks, and its current status.

Cluster-side problems found at the same time (lem `$HOME` quota, lem HF cache permissions,
helios `PIXI_HOME`) are deliberately **not** here -- see
[notes/cluster_fixes_lem_helios.md](notes/cluster_fixes_lem_helios.md).

---

## 1. `ProjectedEmbedding` configs still pass the pre-#147 nested `embedding:` argument

**Status:** **fixed**, merged in #193 (2026-10-26).

**Breaks:** every non-mem-eff PC model built from one of these files dies at instantiation with

```
TypeError: ProjectedEmbedding.__init__() got an unexpected keyword argument 'embedding'
```

`src/projected_compression/model.py:568` takes flat arguments:

```python
class ProjectedEmbedding(Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, result_out_features: int):
```

but four configs still nest a `src.projected_compression.model.Embedding` under an
`embedding:` key. Commit `d66a3510` "emb.emb fix (#147)" changed the class and updated only
two of the six call sites.

**Affected files** (all four fixed together):

| file | reaches |
|---|---|
| `configs/_model/llama_tiny_compressor.yaml` | `pc_project/llama_tiny_comp.yaml` |
| `configs/_model/llmrandom/base_pc_model.yaml` | `projected_300`, `projected_800` -> `llmr_300_comp`, `llmr_800_comp` |
| `configs/_model/smollm/base_pc_model.yaml` | `projected_360`, `projected_1700` -> `smollm_*_comp` |
| `configs/_model/llama_3_8_pc.yaml` | nothing -- orphan, see issue 3 |

Already correct, for reference: `configs/_model/llama/base_pc_model.yaml`,
`configs/_model/llama_3_1_pc.yaml`.

**Fix:** drop the nested node and hoist its two arguments.

```yaml
# before
_target_: src.projected_compression.model.ProjectedEmbedding
embedding:
  _target_: src.projected_compression.model.Embedding
  num_embeddings: ${common.vocab_size}
  embedding_dim: ${common.base_dmodel}
result_out_features: ${common.dmodel}

# after
_target_: src.projected_compression.model.ProjectedEmbedding
num_embeddings: ${common.vocab_size}
embedding_dim: ${common.base_dmodel}
result_out_features: ${common.dmodel}
```

---

## 2. Two configs call `init_compression` with its pre-importances signature

**Status:** **fixed**, merged in #193 (2026-10-26).

**Breaks:** the run reaches model construction and then dies with

```
TypeError: init_compression() got an unexpected keyword argument 'dff'
```

`src/projected_compression/compression.py:135` is

```python
def init_compression(model, dimensions_importances_path, target_dmodel, target_dff):
```

so `dmodel`/`dff` were renamed to `target_dmodel`/`target_dff` and
`dimensions_importances_path` became **required** -- the function `torch.load`s it and takes
`topk` over `dmodel_importances` / `dff_importances`. Two configs still use the old names and
supply no importances path at all: `configs/_model/llama_tiny_compressor.yaml` and
`configs/_model/llama_3_8_pc.yaml`. Every other caller already passes the three current
arguments.

**Fix:** rename the two keys and add `dimensions_importances_path: ???`, matching
`configs/_model/smollm/base_pc_model.yaml`. The `???` is deliberate -- there is no sensible
default, and it turns a mid-run `TypeError` into a compose-time missing-key error.

> The importances file is `{"dmodel_importances": Tensor[base_dmodel],
> "dff_importances": Tensor[n_blocks, base_dff]}`, float32 (verified against
> `.../plgj321m/pc/importances/1b/5401128/0/minitron_dimensions_importances.pt`:
> `[2048]` and `[16, 8192]`).

### Why 1 and 2 were not caught earlier

Both are **instantiation**-time failures. The config sweep done for the merge review only
*composed* configs, which passes for both. Composing is not enough; instantiate the `model`
node on a meta device to catch this class of rot:

```python
with torch.device("meta"):
    instantiate(cfg.model, _convert_="all")
```

Import `src.projected_compression.model` and `torch._dynamo` **eagerly** first -- a lazy
import inside the meta-device context hits a circular import and reports a misleading
`Error locating target '...model.LLM'`.

After fixing 1 and 2, all **20** concrete `_model/**.yaml` configs instantiate
(llama tiny/1B/8B and mem-eff, llmrandom 300/800, smollm 135/360/1700, tiny). The 4 that do
not are base files whose `???` placeholders are filled by their children.

---

## 3. Other known-open items (not fixed)

| item | detail |
|---|---|
| ~~`full_save_checkpoints_path` logs `None/<jobid>/<task>/step_N`~~ **FIXED** | When `checkpoint.save.path` was null, `_build_save_path` (`main.py:41`) interpolated the literal string `None`, so the run record carried a path like `None/<jobid>/<task>/step_N`. Cosmetic -- `_should_save_final_checkpoint` guards on `save.path is not None`, so nothing is written there -- but it is a path in the run record that does not exist, which is the same way F17 wasted time. It now logs `<checkpointing disabled>`; mutation-tested. |
| Orphan `_model` configs -- **five, not one; now guarded** | `llama_3_8_pc.yaml` was recorded as the orphan. Searching rather than spot-checking finds **five**: `llama_3_8_pc`, `llama_3_1_pc`, `llama/projected_tiny`, `llama/projected_mem_eff_tiny`, `llama/small` and `smollm/projected_tiny`. (`small_moe` pulls in `base_model`, not `small`; the only `- small` elsewhere is a wandb tag.) They are dead weight that still gets maintained -- `llama_3_8_pc` was edited under two earlier issues purely for consistency with configs nobody runs. **Not deleted**: an orphan model config is plausibly staging for unlanded work, and that is not an audit's call. `test_no_new_orphan_model_configs` allowlists these and fails if the set grows; mutation-tested by adding one. Wiring up or dropping any of them is a decision for whoever wrote them. |
| ~~`configs/pc_project/llama_tiny_comp.yaml` cannot run as shipped~~ **NOT A BUG** | Both values are `??? # CHANGE`, which is this repo's template convention for a value the user must supply -- `llama_8B_comp.yaml`, `smollm_135.yaml`, `smollm_360_importances.yaml` and `memory_efficient/1b.yaml` all do the same. The config composes and resolves (Tier 0 tolerates `???` while still catching broken interpolations behind one); it simply cannot be *run* unedited, which is what a template is. Filling in a learning rate would be guessing at someone's intent, the same trap as the `llmr_300` batch size. Reclassified 2026-10-28, not fixed. |
| ~~~38 configs fail to compose~~ **FIXED** | Was pre-existing and byte-identical between `origin/main` and `mas/nips`. The `Could not find 'ff_layer/dense'` family came from an absolute group reference in `_model/llama/base_model.yaml` that only resolved when the search-path root was `configs/`. Fixed in the baseline audit, along with a duplicate YAML key (`llmr_300.yaml` declared `batch_size` twice, 512 then 16, so Hydra rejected the file and no run ever loaded it -- **kept 512, confirmed by Maciej 2026-10-28**, matching sibling `llmr_800`) and two missing `common.batch_size`; all 82 runnable configs now compose **and** resolve, enforced by `tests/test_repo_hygiene.py`. See `notes/main_baseline_audit.md`. |
| ~~`pixi run` rewrites `pixi.lock` v6 -> v7~~ **FIXED** | Measured across four versions rather than inferred: pixi **0.63.2** (lem) writes lock v6, **0.67.1** (helios) leaves it at v6, but **0.77.0** and **0.81.0** both bump it to v7. **The entry only named 0.81; entropy's own pixi is 0.77.0, which corrupts the lock too** -- previously unrecorded. **Measured what a v7 lock actually does on helios's pixi 0.67.1: it does not fail, it reports `the lock-file will be treated as missing and regenerated` and re-solves.** So the cluster quietly builds an environment the lock never described, with the pinned versions gone and nothing saying so -- worse than the hard failure the entry implied. One `pixi run` from the repo on a modern pixi rewrote +6234/-6089 lines of lock and did that to the cluster. `pixi.toml` now declares `requires-pixi = ">=0.60,<0.77"`, so a too-new pixi stops with a clear message instead of silently rewriting. Verified both ways: local 0.81 refuses, lem's 0.63.2 runs normally and leaves the lock byte-identical at v6. It does not disturb the clusters -- they activate from their own `PIXI_HOME`, which has its own manifest, so the bound applies only inside the repo, which is where the damage happened. To run code on a newer pixi, bypass it: `./.pixi/envs/default/bin/python run_exp.py ...`. `test_pixi_version_is_bounded_below_the_lock_bump` keeps the upper bound at or below 0.77 and the lower bound admitting 0.63.2; mutation-tested both ways. Verified against the command the entry names: `pixi run` is refused, not just `pixi lock`. The rewrite itself is pixi's behaviour and not something this repo can change -- what a bound can do is make it loud instead of silent, which is the difference between a broken cluster and an error message. **The bound broke two things on its way in, both now fixed and tested.** `scripts/sync_cluster_env.sh` had to stop copying the bound into a cluster `PIXI_HOME` (entropy's pixi 0.77 is excluded by it), and `.githooks/pre-commit` called `pixi run --frozen black`, which the bound refuses -- every commit would have failed for anyone with the hook installed. The hook now calls black through `./.pixi/envs/default/bin/python` and its advice text no longer recommends a command that cannot work. Nothing had ever exercised that hook, which is how it was free to rot; `tests/test_pre_commit_hook.py` now does. Note also that the hook is **opt-in**: `core.hooksPath` is unset unless someone installs it, so it protects only those who did -- the Tier 0 test and CI are the guards that always run. **And until 2026-10-29 nobody on a modern pixi could install it.** The only documented route was `pixi run install_hooks`, and `requires-pixi` refuses every pixi command in the repo, that task included: on pixi 0.81 it answers `Error: this project requires pixi '>=0.60,<0.77', but you have pixi 0.81.0`. The hook's job is catching a lock a too-new pixi rewrote, so the people who needed it were precisely the people who could not install it. `scripts/install_hooks.sh` now does it with plain git and no pixi; the task calls that script. Found by running the documented command instead of reading it. Fixed 2026-10-28, hook repaired 2026-10-29. |
| ~~`run_exp.py` leaves the branch broken if killed~~ **FIXED** | `version_code()` commits onto the branch you are standing on and restored in a `finally`, which covers exceptions but not a signal -- the default SIGTERM handler kills the process outright, so a cancelled or preempted submission left the repo on the experiment branch with your own branch unrestored. Now wrapped in `restoring_repo_state()`, a context manager that installs SIGTERM/SIGINT handlers which restore, put the previous handler back and re-raise the signal, so the process still dies as asked. Guarded against double-restore, and a no-op off the main thread where `signal()` is unavailable. Four tests in `tests/test_run_exp_versioning.py`, mutation-tested. `notes/cluster_fixes_lem_helios.md` keeps the repair recipe for repos already in that state. |
| **GPU count silently changes which data a run trains and evals on** | Two runs of the same config at the same seed are **not comparable if they used a different number of GPUs**, and nothing warns you. A 1-GPU and a 2-GPU run diverge by up to 0.75 in loss (helios job 25259068). Cause: every shipped dataset config sets `world_size_independent: false`, under which `_load_dataset` calls `split_dataset_by_node` at load time, so each rank reads a different share of the 64 shards; shuffle and packing boundaries follow. Measured in three separate single-rank processes, `world_size=2` rank 0, rank 1 and the `world_size=1` stream share **zero** sequences of 24, pairwise. Ranks never duplicate data -- that part is sound -- but the corpus order is a function of `world_size`, not of the seed alone. **The eval dataloader has the same setting, which contradicts the code's own docstring**: `C4Dataset` states "world_size_independent should be 'True' for tests and eval", yet `_dataset/fineweb.yaml`, `c4.yaml` and `local_dummy.yaml` all set `false` on **both** dataloaders. So eval loss is computed on a rank-dependent subset too, and eval numbers do not compare across GPU counts either. Consequences: a run preempted and requeued on a different GPU count silently trains on different data; any A/B whose arms ran at different world sizes is confounded; `world_size` belongs in run provenance alongside the seed. Fixing the eval side is probably just `world_size_independent: true` on the three eval dataloaders, matching the docstring -- but it changes what every eval number means, so it is left for a decision rather than changed inside an audit. The train side is a deliberate sharding strategy and should stay; what it needs is to be documented and recorded per run. Partially guarded: `test_ranks_never_see_the_same_sample` and `test_world_size_independent_partitions_one_stream` pin the sharding contract so it cannot silently become duplication. Found 2026-10-28. |
| ~~4-GPU production world size not directly observed~~ **NOW OBSERVED** | Resolved 2026-10-27 by the baseline audit's PCD-50 run on lem (job 6174094, `pcd50_prefix`): four distinct ranks logged (`device:0`–`device:3`), four `Sharding done` lines (FSDP2 sharded once per rank) and four allocated H100s, running the production `PCDistillationTrainer` on the real 1B model at 50% compression for 450 steps. The transitivity argument this row recorded is no longer needed. |
| ~~PCD-50 repro only covers a prefix~~ **FULL RUN REPRODUCES IT** | The baseline audit's full 4096-step PCD-50 run on lem (job 6174690, 4×H100, 4h41m, rc=0) reaches final `steps/100/train/loss` **2.674062** against reference run `uhhl03fy` **2.670469** -- **+0.0036, 0.13%**. Across all 40 logged points mean \|delta\| is 0.0088; from step 201 onward every delta is +0.0003 to +0.0092, a constant offset with no drift. The single 0.14 outlier is the step-101 window, the steepest part of the curve, on different hardware from the reference. `main` reproduces the paper number. Recorded 2026-10-28. |
| ~~`save_training_state` logged the wrong path~~ **FIXED** | It writes to `<save.path>/step_<n>/` but logged `<save.path>`, the parent, where no such file exists. Resume looks for the training state at exactly `load.path`, so following the log meant pointing it at the parent, getting `Training state file ... not found. Starting training from scratch`, and silently restarting at step 0 -- F14 one layer out, where the loud part is right and the part you act on is wrong. One-line fix plus a test asserting the logged path is a real file. Found 2026-10-28 while checking whether sharded-checkpoint resume had GPU coverage. |
| ~~Resume crashed outright whenever eval was switched off~~ **FIXED** | `Trainer.__attrs_post_init__` advanced the eval stream on resume with `(start_step - 1) // self.eval_interval * n_eval_steps`. `eval_interval` is **0** whenever eval is off -- **31 shipped configs set it**, the PCD-50 repro among them -- so every such run raised `ZeroDivisionError` before training began and **could not be resumed at all**. A preempted-and-requeued job was unrecoverable; combined with the F14 replay bug, the resume path was broken in both directions. Needs no GPU and no data to reproduce, but only fires on a resume, which is why it sat behind the one path nothing routinely runs. Found 2026-10-28 on helios job 25327019. Guard added plus four regression tests; mutation-tested. |
| ~~Resume replays training data from batch 0~~ **FIXED** | `train()` zipped `range(start_step, n_steps)` against a *fresh* dataloader iterator, so a resumed run re-trained on the earliest batches while the step counter carried on. On `tiny_local` a 20-step run resumed at step 10 reported loss 10.17 against 12.39 uninterrupted -- it fails *downwards*, so it reads as a good run. Fixed by `Trainer._batches_from_start_step()` in all three training loops; resumed runs are now bit-identical to uninterrupted ones. The path was marked `# not tested` in `_checkpoints/none.yaml`, so published results are unlikely to be affected, but any preempted-and-requeued job would have been. Found 2026-10-27. |
| ~~Fixing the repo's lock does not reach the clusters~~ **DEMONSTRATED END TO END** | Each cluster sets `PIXI_HOME=$PROJECT_HOME_PATH/$USER/pixi`, a directory holding **its own copy** of `pixi.toml` and `pixi.lock`. The repo's lock is the source, but nothing syncs it, so regenerating the lock on this branch repairs the repo and leaves every cluster env as it was -- confirmed on entropy, whose env still has no pytest and carries `transformers 5.12.1` against a manifest pinning `==4.56.2`. This cannot be fixed *by* a PR, but it no longer has to be folklore: **`scripts/sync_cluster_env.sh`** diffs the repo's manifest and lock against a given `PIXI_HOME` and, with `--apply`, copies them over (keeping `.bak` copies) and runs `pixi install --frozen` -- frozen on purpose, since a re-solve would rewrite the lock to v7 and helios's pixi 0.67.1 cannot read v7. Dry run by default, and it warns that a cluster PIXI_HOME is usually shared. **Dry-run against entropy (job 30716) confirms the drift and that the script works**: both locks are v6, but `pixi.toml` and `pixi.lock` both differ, and the cluster lock carries **0** pytest entries against the repo's **1**. `--apply` was deliberately **not** passed: that env is shared, and moving `transformers` 5.12.1 → 4.56.2 would change it under anyone else using it. Running the sync is a decision for whoever owns that environment, and the script now makes it one command plus a backup. |
| ~~`pytest` declared but not locked; lock did not satisfy the manifest~~ **FIXED** | `pixi.toml` declared `pytest` and `accelerate` that `pixi.lock` did not carry, and pinned `transformers == 4.56.2` while the lock held 4.55.0, so any env installed from the lock had no pytest (confirmed on lem and locally; helios's env was re-solved, not installed from the lock, so it had pytest). The blocker was a wrong assumption that regenerating forces lock v7 -- the format follows the **writer**. lem runs pixi 0.63.2, helios 0.67.1, so regenerating with **0.63.2** keeps it at **v6** and both can read it. Result: 168 insertions / 41 deletions (a pixi 0.81 rewrite is +6234/-6089), idempotent on re-run, adds pytest 8.4.2 + accelerate 1.15.0, moves transformers to 4.56.2 and tokenizers to 0.22.2. Not a new risk: helios already runs transformers 4.56.2 and pytest 8.4.2, verified on a compute node, so the lock now codifies production and brings lem into line. The v6 pin and its hygiene test stand (F7: do not let pixi 0.81 rewrite it). Closed end to end 2026-10-29: an entropy environment built from this lock (job 30739) runs the suite green (job 30740, `SUITE_RC=0`) at pytest 8.4.2 and transformers 4.56.2 -- the gate on a cluster through pixi rather than a layered venv, which is what this entry originally blocked. |
| ~~fineweb eval is the training data, on every cluster~~ **FIXED** | `_dataset/fineweb.yaml` gave its eval dataloader `split: train` and `seed: 123`, the same as the train one, and `eval_path_fineweb` equalled `train_path_fineweb` on entropy, helios and lem, so the eval dataloader yielded **the batches the model was training on, in the same order** -- flattering, and affecting **29 of the 64 fineweb configs**. Not a typo: `datasets/fineweb/` holds only `train` on both clusters (verified directly) while `datasets/c4/` has `train` and `validation`; no fineweb validation split was ever downloaded. I first recorded this as needing a decision about experiments. It did not: `load_from_disk` returns a map-style dataset before `to_iterable_dataset`, so a **`holdout`** parameter can carve a disjoint slice out of the one directory that exists -- the last `holdout_fraction` (default 1%) of rows to eval, the rest to train, both computed from the same row count so they cannot overlap. `holdout=None` keeps the old whole-dataset behaviour. Mutation-tested: defeat the holdout and 20 of 20 sequences appear in both slices. The hygiene test now accepts a shared path **only** where a disjoint holdout is declared, and its allowlist shrank from 5 entries to 1. **Consequence for experiments: fineweb training runs now see 99% of the rows, so new runs are not bit-comparable with older ones** -- inherent to having an honest eval; the fraction is a config knob. Paper headline numbers are train loss and downstream evals come from lm_eval, so those are unaffected. Fixed 2026-10-28. |
| ~~Eval depended on the GPU count~~ **FIXED** | `AbstractDataset`'s contract states world_size_independent "should be 'True' for tests and eval", but `_dataset/fineweb.yaml`, `c4.yaml` and `local_dummy.yaml` all set **False** on the eval dataloader as well as the train one. With False the rows are split across ranks by `split_dataset_by_node`, so each rank evaluated on a different slice and the reported eval number changed with the number of GPUs. Set to True on all three eval dataloaders, matching the docstring, and `test_eval_dataloaders_are_world_size_independent` now enforces it. The train side keeps False: that sharding is deliberate (see the GPU-count row). Fixed 2026-10-28. |
| lem fineweb paths hardcode an account name -- **ACCEPTED, no action** | Same two lines, under `/plgllmefficont2/crewtool/proba2/...`. This is framework config every experiment inherits, and the file already states the rule it breaks ("Derived from `$USER`: no account name belongs in a shared cluster config"). The c4 paths next to it use a shared `datasets/` directory, so the fix is probably `.../datasets/fineweb-edu/...`, but guessing a data path breaks everyone's runs. Allowlisted in `tests/test_repo_hygiene.py` so new violations still fail. **Maciej reviewed this on 2026-10-28 and is fine with it as-is**; kept here as a record, not as an open defect. |
| ~~`save_to_llama_3_hf` writes no tokenizer~~ **FIXED** | `src/core/conversion_to_hf.py` saved `config.json` + weights but no tokenizer, unlike `PCTrainer.save_hf_checkpoint`, so HP/HPD HF checkpoints came out unusable by lm_eval until someone supplied one by hand -- and the mismatch was silent. It now saves the tokenizer from the same source model the config comes from, named once as `SOURCE_LLAMA_REPO` so the two cannot drift apart. Pre-existing on `origin/main`. |
| ~~Finalized checkpoints land in `step_-1`~~ **FIXED** | `Trainer.step` is `start_step - 1`, so -1 before the first step. A run with `n_steps: 0` -- how every finalize and conversion pass is configured -- never advances it, and `_should_save_final_checkpoint` is satisfied at `step >= n_steps - 1`, so the checkpoint was written to a folder named literally `step_-1`. Not PC-specific: any save type with `n_steps: 0` hits it. I first left this alone on the grounds that changing it moves where production checkpoints are written -- but such a pass has completed no steps, so step 0 is simply what it is, and a negative index leaking into a path is the kind of thing an operator then has to go hunting for. `step_checkpoint_path` now clamps to 0; ordinary steps are untouched and tested to be. Observed 2026-10-28 in helios job 25325254, the first end-to-end run of that config. Fixed the same day. |
| ~~Both mem-eff -> HF conversion configs were dead since March~~ **FIXED (two stacked bugs)** | `convert_memeff_to_hf.convert_pc_to_hf` is the `init_model_opt_sched_fn` entry point for **both** `llama_1B_memeff_to_hf_conversion` and `llama_8B_memeff_to_hf_conversion`, and it had two independent defects in the same four lines. **(1)** It called `get_metric_logger(..., neptune_run_id=None)`; the neptune -> wandb transition (d25c326b, #169) renamed that to `tracker_run_id` and missed this caller, the only one left in the repo -- `TypeError` on the first line of real work. **(2)** With that fixed it got one line further and raised `AttributeError: 'dict' object has no attribute 'type'`: it passed `instantiate(cfg.infrastructure.metric_logger, _convert_="all")`, which yields a plain dict, while `get_metric_logger` reads `.type` by attribute. The two working call sites (`main.py:228`, `initialization.py:49`) pass the config node directly; this one also silently omitted `full_config`, which is what populates the wandb run config. So the mem-eff PC checkpoints every `*_comp_memeff` run produces could not be converted to HF at all -- the step before any lm_eval number. Neither defect is reachable without a trained 1B checkpoint, and import-time checks cannot see either: both sit in a function body no test executes. Found 2026-10-28 by running the conversion on lem (jobs 6183879, 6183912). Caller now matches the convention; `get_metric_logger` normalises a plain dict to a DictConfig, since both forms describe the same config; `test_call_sites_match_the_function_signature` walks the AST for stale keywords and `test_metric_logger_accepts_either_config_form` covers the type. Both mutation-tested. |
| ~~`pc_finalize` still couples `src/core/` to PC~~ **FIXED** | The note was right that repointing at `PCTrainer` would break it, since `PCTrainer` assumes `model.target_model`. Resolved instead with a new `PCFinalizeTrainer(Trainer)` in `src/projected_compression/trainer.py`: core's final-save branch became the overridable `save_final_checkpoint()`, and `llama_1B_comp_finalize.yaml` targets the subclass. `grep -rn projected_compression src/core/` is empty and a test keeps it that way. |
| `configs/mastest/` never reaches `main` -- **checked; do not 'fix' the ignore rule** | 51 files under `configs/mastest/`, matched by `.gitignore:194 *mastest*`; 47 were already broken. Gitignored on purpose, so nothing is wrong with the files -- the risk is someone assuming they travel, which is now step 5 of the branch-point protocol. **The unanchored pattern looks like the `slicegpt/` bug this repo already fixed by anchoring, and it is not.** `*mastest*` matches any path containing the string, and in the working checkout that is deliberate: **25 root-level files** (`mastestt.ipynb`, `mastest_nips_ds.ipynb`, `mastest_pc_vs_hpr_speedup.png` and so on) rely on it. Anchoring it to `/configs/mastest/` would stop ignoring those and fill `git status` with personal scratch work. Verified 2026-10-29 by listing what the pattern actually covers. Left exactly as it is. |
| cemetery clone per submission -- **checked, not a bug** | `run_exp.py:205` runs `mkdir -p {cemetery_dir}` on every submission, then clones into `{cemetery_dir}/{experiment_branch_name}` guarded by `test -d`. Branch names are timestamped, so each submission does get a fresh clone. The quota problem was clones *accumulating* (253 of them, 13 G on lem), never that they were missing. |
| ~~cemetery redirect never exercised end to end~~ **NOW EXERCISED** | Done 2026-10-27 on **both** clusters. `infrastructure.cemetery_experiments_dir` composes to the cluster-specific path rather than the `~/llmrandom_cemetery` default (helios `/net/storage/pr3/plgrid/plggllmeffi3/pc/$USER/llmrandom_cemetery`, lem `/lustre/pd03/plgrid/plgllmefficont3/nano/$USER/llmrandom_cemetery`), and running exactly what `run_exp.py:205-212` runs -- `mkdir -p` then a clone into `{cemetery_dir}/{name}` -- lands the clone under the redirected directory with `$USER` expanded, leaving `$HOME/llmrandom_cemetery` untouched. Probe directories were removed afterwards. **Caveat:** this exercised the redirect, not `run_exp.py`'s own submission machinery (`version_code()` commits onto the branch you are standing on, so it was deliberately not driven from the audit branch). A real submission still exercises that part. |
| lem data is symlinked, not moved | The per-user path is now `<grant-root>/nano/$USER`, but the 414 G of data (pixi env, HF caches, checkpoints) still sits at the old `<grant-root>/mstefaniak`, and **10 `configs/pc_project_mas/*.yaml` hardcode that old path** for checkpoints and evals. A symlink `nano/plgmstefaniak -> ../mstefaniak` makes both resolve, so nothing was copied and nothing broke. Moving it for real is a separate task that must update those 10 configs in lockstep. |
| ~~unanchored `slicegpt/` in `.gitignore`~~ | **Fixed in #195.** `slicegpt/` without a leading slash matches a directory of that name at *any* depth, so it silently excluded `src/projected_compression/slicegpt/` and `configs/slicegpt/` as well as the third-party clone it was written for -- `git add` did nothing, with no error. Now `/slicegpt/`. Worth remembering as a pattern: unanchored directory rules in `.gitignore` are depth-independent. |
