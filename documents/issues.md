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
| `full_save_checkpoints_path` logs `None/<jobid>/<task>/step_N` | When `checkpoint.save.path` is null, `_build_save_path` (`main.py:41`) interpolates the literal string `None`. Cosmetic only: `_should_save_final_checkpoint` (`src/core/trainer.py:110`) guards on `save.path is not None`, so no `None/` directory is ever created. |
| `configs/_model/llama_3_8_pc.yaml` is an orphan | Referenced by no config. It was fixed under issues 1 and 2 for consistency, but it is dead weight -- either wire it up or drop it. |
| `configs/pc_project/llama_tiny_comp.yaml` cannot run as shipped | `learning_rate: ???`, and now also `dimensions_importances_path: ???`. It is the only consumer of `llama_tiny_compressor.yaml`. |
| ~~~38 configs fail to compose~~ **FIXED** | Was pre-existing and byte-identical between `origin/main` and `mas/nips`. The `Could not find 'ff_layer/dense'` family came from an absolute group reference in `_model/llama/base_model.yaml` that only resolved when the search-path root was `configs/`. Fixed in the baseline audit, along with a duplicate YAML key (`llmr_300.yaml` declared `batch_size` twice, 512 then 16, so Hydra rejected the file and no run ever loaded it -- **kept 512, confirmed by Maciej 2026-10-28**, matching sibling `llmr_800`) and two missing `common.batch_size`; all 82 runnable configs now compose **and** resolve, enforced by `tests/test_repo_hygiene.py`. See `notes/main_baseline_audit.md`. |
| `pixi run` rewrites `pixi.lock` v6 -> v7 | pixi 0.81.0 locally. Shipping that lock to a cluster whose pixi is older breaks it (**helios runs 0.67.1**, which cannot read v7). **Decision 2026-10-26: keep the committed lock at v6 and do not regenerate it**, even though `pixi.toml` now declares `pytest` that the lock does not carry -- `pixi install` re-solves anyway, and pixi 0.81 already reports the lock out-of-date independently. Workaround for launchers: `./.pixi/envs/default/bin/python run_exp.py ...`, which leaves the lock alone. |
| `run_exp.py` leaves the branch broken if killed | `version_code()` commits onto the branch you are standing on and only restores in a `finally`, which a SIGTERM bypasses. See `notes/cluster_fixes_lem_helios.md` for the repair recipe. |
| **GPU count silently changes which data a run trains and evals on** | Two runs of the same config at the same seed are **not comparable if they used a different number of GPUs**, and nothing warns you. A 1-GPU and a 2-GPU run diverge by up to 0.75 in loss (helios job 25259068). Cause: every shipped dataset config sets `world_size_independent: false`, under which `_load_dataset` calls `split_dataset_by_node` at load time, so each rank reads a different share of the 64 shards; shuffle and packing boundaries follow. Measured in three separate single-rank processes, `world_size=2` rank 0, rank 1 and the `world_size=1` stream share **zero** sequences of 24, pairwise. Ranks never duplicate data -- that part is sound -- but the corpus order is a function of `world_size`, not of the seed alone. **The eval dataloader has the same setting, which contradicts the code's own docstring**: `C4Dataset` states "world_size_independent should be 'True' for tests and eval", yet `_dataset/fineweb.yaml`, `c4.yaml` and `local_dummy.yaml` all set `false` on **both** dataloaders. So eval loss is computed on a rank-dependent subset too, and eval numbers do not compare across GPU counts either. Consequences: a run preempted and requeued on a different GPU count silently trains on different data; any A/B whose arms ran at different world sizes is confounded; `world_size` belongs in run provenance alongside the seed. Fixing the eval side is probably just `world_size_independent: true` on the three eval dataloaders, matching the docstring -- but it changes what every eval number means, so it is left for a decision rather than changed inside an audit. The train side is a deliberate sharding strategy and should stay; what it needs is to be documented and recorded per run. Partially guarded: `test_ranks_never_see_the_same_sample` and `test_world_size_independent_partitions_one_stream` pin the sharding contract so it cannot silently become duplication. Found 2026-10-28. |
| ~~4-GPU production world size not directly observed~~ **NOW OBSERVED** | Resolved 2026-10-27 by the baseline audit's PCD-50 run on lem (job 6174094, `pcd50_prefix`): four distinct ranks logged (`device:0`–`device:3`), four `Sharding done` lines (FSDP2 sharded once per rank) and four allocated H100s, running the production `PCDistillationTrainer` on the real 1B model at 50% compression for 450 steps. The transitivity argument this row recorded is no longer needed. |
| ~~PCD-50 repro only covers a prefix~~ **FULL RUN REPRODUCES IT** | The baseline audit's full 4096-step PCD-50 run on lem (job 6174690, 4×H100, 4h41m, rc=0) reaches final `steps/100/train/loss` **2.674062** against reference run `uhhl03fy` **2.670469** -- **+0.0036, 0.13%**. Across all 40 logged points mean \|delta\| is 0.0088; from step 201 onward every delta is +0.0003 to +0.0092, a constant offset with no drift. The single 0.14 outlier is the step-101 window, the steepest part of the curve, on different hardware from the reference. `main` reproduces the paper number. Recorded 2026-10-28. |
| ~~`save_training_state` logged the wrong path~~ **FIXED** | It writes to `<save.path>/step_<n>/` but logged `<save.path>`, the parent, where no such file exists. Resume looks for the training state at exactly `load.path`, so following the log meant pointing it at the parent, getting `Training state file ... not found. Starting training from scratch`, and silently restarting at step 0 -- F14 one layer out, where the loud part is right and the part you act on is wrong. One-line fix plus a test asserting the logged path is a real file. Found 2026-10-28 while checking whether sharded-checkpoint resume had GPU coverage. |
| ~~Resume crashed outright whenever eval was switched off~~ **FIXED** | `Trainer.__attrs_post_init__` advanced the eval stream on resume with `(start_step - 1) // self.eval_interval * n_eval_steps`. `eval_interval` is **0** whenever eval is off -- **31 shipped configs set it**, the PCD-50 repro among them -- so every such run raised `ZeroDivisionError` before training began and **could not be resumed at all**. A preempted-and-requeued job was unrecoverable; combined with the F14 replay bug, the resume path was broken in both directions. Needs no GPU and no data to reproduce, but only fires on a resume, which is why it sat behind the one path nothing routinely runs. Found 2026-10-28 on helios job 25327019. Guard added plus four regression tests; mutation-tested. |
| ~~Resume replays training data from batch 0~~ **FIXED** | `train()` zipped `range(start_step, n_steps)` against a *fresh* dataloader iterator, so a resumed run re-trained on the earliest batches while the step counter carried on. On `tiny_local` a 20-step run resumed at step 10 reported loss 10.17 against 12.39 uninterrupted -- it fails *downwards*, so it reads as a good run. Fixed by `Trainer._batches_from_start_step()` in all three training loops; resumed runs are now bit-identical to uninterrupted ones. The path was marked `# not tested` in `_checkpoints/none.yaml`, so published results are unlikely to be affected, but any preempted-and-requeued job would have been. Found 2026-10-27. |
| ~~`pytest` declared but not locked; lock did not satisfy the manifest~~ **FIXED** | `pixi.toml` declared `pytest` and `accelerate` that `pixi.lock` did not carry, and pinned `transformers == 4.56.2` while the lock held 4.55.0, so any env installed from the lock had no pytest (confirmed on lem and locally; helios's env was re-solved, not installed from the lock, so it had pytest). The blocker was a wrong assumption that regenerating forces lock v7 -- the format follows the **writer**. lem runs pixi 0.63.2, helios 0.67.1, so regenerating with **0.63.2** keeps it at **v6** and both can read it. Result: 168 insertions / 41 deletions (a pixi 0.81 rewrite is +6234/-6089), idempotent on re-run, adds pytest 8.4.2 + accelerate 1.15.0, moves transformers to 4.56.2 and tokenizers to 0.22.2. Not a new risk: helios already runs transformers 4.56.2 and pytest 8.4.2, verified on a compute node, so the lock now codifies production and brings lem into line. The v6 pin and its hygiene test stand (F7: do not let pixi 0.81 rewrite it). Fixed 2026-10-28. |
| ~~fineweb eval is the training data, on every cluster~~ **FIXED** | `_dataset/fineweb.yaml` gave its eval dataloader `split: train` and `seed: 123`, the same as the train one, and `eval_path_fineweb` equalled `train_path_fineweb` on entropy, helios and lem, so the eval dataloader yielded **the batches the model was training on, in the same order** -- flattering, and affecting **29 of the 64 fineweb configs**. Not a typo: `datasets/fineweb/` holds only `train` on both clusters (verified directly) while `datasets/c4/` has `train` and `validation`; no fineweb validation split was ever downloaded. I first recorded this as needing a decision about experiments. It did not: `load_from_disk` returns a map-style dataset before `to_iterable_dataset`, so a **`holdout`** parameter can carve a disjoint slice out of the one directory that exists -- the last `holdout_fraction` (default 1%) of rows to eval, the rest to train, both computed from the same row count so they cannot overlap. `holdout=None` keeps the old whole-dataset behaviour. Mutation-tested: defeat the holdout and 20 of 20 sequences appear in both slices. The hygiene test now accepts a shared path **only** where a disjoint holdout is declared, and its allowlist shrank from 5 entries to 1. **Consequence for experiments: fineweb training runs now see 99% of the rows, so new runs are not bit-comparable with older ones** -- inherent to having an honest eval; the fraction is a config knob. Paper headline numbers are train loss and downstream evals come from lm_eval, so those are unaffected. Fixed 2026-10-28. |
| ~~Eval depended on the GPU count~~ **FIXED** | `AbstractDataset`'s contract states world_size_independent "should be 'True' for tests and eval", but `_dataset/fineweb.yaml`, `c4.yaml` and `local_dummy.yaml` all set **False** on the eval dataloader as well as the train one. With False the rows are split across ranks by `split_dataset_by_node`, so each rank evaluated on a different slice and the reported eval number changed with the number of GPUs. Set to True on all three eval dataloaders, matching the docstring, and `test_eval_dataloaders_are_world_size_independent` now enforces it. The train side keeps False: that sharding is deliberate (see the GPU-count row). Fixed 2026-10-28. |
| lem fineweb paths hardcode an account name -- **ACCEPTED, no action** | Same two lines, under `/plgllmefficont2/crewtool/proba2/...`. This is framework config every experiment inherits, and the file already states the rule it breaks ("Derived from `$USER`: no account name belongs in a shared cluster config"). The c4 paths next to it use a shared `datasets/` directory, so the fix is probably `.../datasets/fineweb-edu/...`, but guessing a data path breaks everyone's runs. Allowlisted in `tests/test_repo_hygiene.py` so new violations still fail. **Maciej reviewed this on 2026-10-28 and is fine with it as-is**; kept here as a record, not as an open defect. |
| `save_to_llama_3_hf` writes no tokenizer | `src/core/conversion_to_hf.py` saves `config.json` + weights but no tokenizer, unlike `PCTrainer.save_hf_checkpoint` which saves one. So HP/HPD HF checkpoints need a tokenizer supplied separately for lm_eval. Pre-existing on `origin/main`. |
| ~~Finalized checkpoints land in `step_-1`~~ **FIXED** | `Trainer.step` is `start_step - 1`, so -1 before the first step. A run with `n_steps: 0` -- how every finalize and conversion pass is configured -- never advances it, and `_should_save_final_checkpoint` is satisfied at `step >= n_steps - 1`, so the checkpoint was written to a folder named literally `step_-1`. Not PC-specific: any save type with `n_steps: 0` hits it. I first left this alone on the grounds that changing it moves where production checkpoints are written -- but such a pass has completed no steps, so step 0 is simply what it is, and a negative index leaking into a path is the kind of thing an operator then has to go hunting for. `step_checkpoint_path` now clamps to 0; ordinary steps are untouched and tested to be. Observed 2026-10-28 in helios job 25325254, the first end-to-end run of that config. Fixed the same day. |
| ~~`pc_finalize` still couples `src/core/` to PC~~ **FIXED** | The note was right that repointing at `PCTrainer` would break it, since `PCTrainer` assumes `model.target_model`. Resolved instead with a new `PCFinalizeTrainer(Trainer)` in `src/projected_compression/trainer.py`: core's final-save branch became the overridable `save_final_checkpoint()`, and `llama_1B_comp_finalize.yaml` targets the subclass. `grep -rn projected_compression src/core/` is empty and a test keeps it that way. |
| `configs/mastest/` never reaches `main` | 50 configs, 252 KB, matched by `.gitignore:194 *mastest*`. 47 of them are already broken (`MissingConfigException`). Nothing carries them over; if any matter they must be committed deliberately. |
| cemetery clone per submission -- **checked, not a bug** | `run_exp.py:205` runs `mkdir -p {cemetery_dir}` on every submission, then clones into `{cemetery_dir}/{experiment_branch_name}` guarded by `test -d`. Branch names are timestamped, so each submission does get a fresh clone. The quota problem was clones *accumulating* (253 of them, 13 G on lem), never that they were missing. |
| ~~cemetery redirect never exercised end to end~~ **NOW EXERCISED** | Done 2026-10-27 on **both** clusters. `infrastructure.cemetery_experiments_dir` composes to the cluster-specific path rather than the `~/llmrandom_cemetery` default (helios `/net/storage/pr3/plgrid/plggllmeffi3/pc/$USER/llmrandom_cemetery`, lem `/lustre/pd03/plgrid/plgllmefficont3/nano/$USER/llmrandom_cemetery`), and running exactly what `run_exp.py:205-212` runs -- `mkdir -p` then a clone into `{cemetery_dir}/{name}` -- lands the clone under the redirected directory with `$USER` expanded, leaving `$HOME/llmrandom_cemetery` untouched. Probe directories were removed afterwards. **Caveat:** this exercised the redirect, not `run_exp.py`'s own submission machinery (`version_code()` commits onto the branch you are standing on, so it was deliberately not driven from the audit branch). A real submission still exercises that part. |
| lem data is symlinked, not moved | The per-user path is now `<grant-root>/nano/$USER`, but the 414 G of data (pixi env, HF caches, checkpoints) still sits at the old `<grant-root>/mstefaniak`, and **10 `configs/pc_project_mas/*.yaml` hardcode that old path** for checkpoints and evals. A symlink `nano/plgmstefaniak -> ../mstefaniak` makes both resolve, so nothing was copied and nothing broke. Moving it for real is a separate task that must update those 10 configs in lockstep. |
| ~~unanchored `slicegpt/` in `.gitignore`~~ | **Fixed in #195.** `slicegpt/` without a leading slash matches a directory of that name at *any* depth, so it silently excluded `src/projected_compression/slicegpt/` and `configs/slicegpt/` as well as the third-party clone it was written for -- `git add` did nothing, with no error. Now `/slicegpt/`. Worth remembering as a pattern: unanchored directory rules in `.gitignore` are depth-independent. |
