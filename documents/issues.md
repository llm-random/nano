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
| ~~~38 configs fail to compose~~ **FIXED** | Was pre-existing and byte-identical between `origin/main` and `mas/nips`. The `Could not find 'ff_layer/dense'` family came from an absolute group reference in `_model/llama/base_model.yaml` that only resolved when the search-path root was `configs/`. Fixed in the baseline audit, along with a duplicate YAML key and two missing `common.batch_size`; all 82 runnable configs now compose **and** resolve, enforced by `tests/test_repo_hygiene.py`. See `notes/main_baseline_audit.md`. |
| `pixi run` rewrites `pixi.lock` v6 -> v7 | pixi 0.81.0 locally. Shipping that lock to a cluster whose pixi is older breaks it (**helios runs 0.67.1**, which cannot read v7). **Decision 2026-10-26: keep the committed lock at v6 and do not regenerate it**, even though `pixi.toml` now declares `pytest` that the lock does not carry -- `pixi install` re-solves anyway, and pixi 0.81 already reports the lock out-of-date independently. Workaround for launchers: `./.pixi/envs/default/bin/python run_exp.py ...`, which leaves the lock alone. |
| `run_exp.py` leaves the branch broken if killed | `version_code()` commits onto the branch you are standing on and only restores in a `finally`, which a SIGTERM bypasses. See `notes/cluster_fixes_lem_helios.md` for the repair recipe. |
| ~~4-GPU production world size not directly observed~~ **NOW OBSERVED** | Resolved 2026-10-27 by the baseline audit's PCD-50 run on lem (job 6174094, `pcd50_prefix`): four distinct ranks logged (`device:0`–`device:3`), four `Sharding done` lines (FSDP2 sharded once per rank) and four allocated H100s, running the production `PCDistillationTrainer` on the real 1B model at 50% compression for 450 steps. The transitivity argument this row recorded is no longer needed. |
| ~~Resume replays training data from batch 0~~ **FIXED** | `train()` zipped `range(start_step, n_steps)` against a *fresh* dataloader iterator, so a resumed run re-trained on the earliest batches while the step counter carried on. On `tiny_local` a 20-step run resumed at step 10 reported loss 10.17 against 12.39 uninterrupted -- it fails *downwards*, so it reads as a good run. Fixed by `Trainer._batches_from_start_step()` in all three training loops; resumed runs are now bit-identical to uninterrupted ones. The path was marked `# not tested` in `_checkpoints/none.yaml`, so published results are unlikely to be affected, but any preempted-and-requeued job would have been. Found 2026-10-27. |
| lem's pixi environment has no pytest | Both envs on lem (`nano/$USER/pixi` and the older `plgllmefficont3/mstefaniak/pixi`) are at lock v6, which predates the `pytest` dependency `pixi.toml` declares, so `python -m pytest` there gives `No module named pytest` and the new test gate cannot run on lem. helios's env does have it. Fixing it means regenerating the lock (breaks helios, which runs pixi 0.67.1) or installing pytest out of band. Found 2026-10-27 while running the suite on both clusters. |
| **fineweb eval is the training data, on every cluster** | `_dataset/fineweb.yaml` gives its eval dataloader `split: train` and `seed: 123`, the same as the train one, and `eval_path_fineweb` equals `train_path_fineweb` in entropy, helios and lem. The datasets are a deterministic function of their seed, so the eval dataloader yields **the batches the model is training on, in the same order** -- eval loss re-scores training data rather than measuring generalisation, and it errs in the flattering direction. **29 of the 64 fineweb configs have eval enabled.** Not a typo: `datasets/fineweb/` holds only `train` on both helios and lem (verified directly), while `datasets/c4/` has `train` and `validation`; no fineweb validation split was ever downloaded, and c4 is configured correctly. The fix is to build a held-out split or to stop reporting fineweb eval -- both decisions about experiments, so left for the user. Guarded by `tests/test_repo_hygiene.py` with the affected pairs allowlisted so no other dataset acquires the shape. Paper headline numbers are train loss and the downstream evals come from lm_eval, so those are unaffected. Found 2026-10-27. |
| lem fineweb paths hardcode an account name | Same two lines, under `/plgllmefficont2/crewtool/proba2/...`. This is framework config every experiment inherits, and the file already states the rule it breaks ("Derived from `$USER`: no account name belongs in a shared cluster config"). The c4 paths next to it use a shared `datasets/` directory, so the fix is probably `.../datasets/fineweb-edu/...`, but guessing a data path breaks everyone's runs. Allowlisted in `tests/test_repo_hygiene.py` so new violations still fail. |
| `save_to_llama_3_hf` writes no tokenizer | `src/core/conversion_to_hf.py` saves `config.json` + weights but no tokenizer, unlike `PCTrainer.save_hf_checkpoint` which saves one. So HP/HPD HF checkpoints need a tokenizer supplied separately for lm_eval. Pre-existing on `origin/main`. |
| ~~`pc_finalize` still couples `src/core/` to PC~~ **FIXED** | The note was right that repointing at `PCTrainer` would break it, since `PCTrainer` assumes `model.target_model`. Resolved instead with a new `PCFinalizeTrainer(Trainer)` in `src/projected_compression/trainer.py`: core's final-save branch became the overridable `save_final_checkpoint()`, and `llama_1B_comp_finalize.yaml` targets the subclass. `grep -rn projected_compression src/core/` is empty and a test keeps it that way. |
| `configs/mastest/` never reaches `main` | 50 configs, 252 KB, matched by `.gitignore:194 *mastest*`. 47 of them are already broken (`MissingConfigException`). Nothing carries them over; if any matter they must be committed deliberately. |
| cemetery clone per submission -- **checked, not a bug** | `run_exp.py:205` runs `mkdir -p {cemetery_dir}` on every submission, then clones into `{cemetery_dir}/{experiment_branch_name}` guarded by `test -d`. Branch names are timestamped, so each submission does get a fresh clone. The quota problem was clones *accumulating* (253 of them, 13 G on lem), never that they were missing. |
| ~~cemetery redirect never exercised end to end~~ **NOW EXERCISED** | Done 2026-10-27 on **both** clusters. `infrastructure.cemetery_experiments_dir` composes to the cluster-specific path rather than the `~/llmrandom_cemetery` default (helios `/net/storage/pr3/plgrid/plggllmeffi3/pc/$USER/llmrandom_cemetery`, lem `/lustre/pd03/plgrid/plgllmefficont3/nano/$USER/llmrandom_cemetery`), and running exactly what `run_exp.py:205-212` runs -- `mkdir -p` then a clone into `{cemetery_dir}/{name}` -- lands the clone under the redirected directory with `$USER` expanded, leaving `$HOME/llmrandom_cemetery` untouched. Probe directories were removed afterwards. **Caveat:** this exercised the redirect, not `run_exp.py`'s own submission machinery (`version_code()` commits onto the branch you are standing on, so it was deliberately not driven from the audit branch). A real submission still exercises that part. |
| lem data is symlinked, not moved | The per-user path is now `<grant-root>/nano/$USER`, but the 414 G of data (pixi env, HF caches, checkpoints) still sits at the old `<grant-root>/mstefaniak`, and **10 `configs/pc_project_mas/*.yaml` hardcode that old path** for checkpoints and evals. A symlink `nano/plgmstefaniak -> ../mstefaniak` makes both resolve, so nothing was copied and nothing broke. Moving it for real is a separate task that must update those 10 configs in lockstep. |
| ~~unanchored `slicegpt/` in `.gitignore`~~ | **Fixed in #195.** `slicegpt/` without a leading slash matches a directory of that name at *any* depth, so it silently excluded `src/projected_compression/slicegpt/` and `configs/slicegpt/` as well as the third-party clone it was written for -- `git add` did nothing, with no error. Now `/slicegpt/`. Worth remembering as a pattern: unanchored directory rules in `.gitignore` are depth-independent. |
