# Context-scaling eval

Two ways to run the same thing:

- **Main (automated)**: one local command does everything — setup, push, ssh, submit, tail log.
- **Legacy (manual)**: original 4-script flow if you want to control each step on the cluster yourself.

---

## Main (automated)

Two local commands, no manual ssh:

```
pixi run python src/context_scaling/scripts/setup_ctx_eval.py   # fetch wandb, write jobs.json
pixi run python src/context_scaling/scripts/run_ctx_eval.py     # push, ssh, sbatch, tail
```

All knobs live in `configs/ctx_eval.yaml` — `eval.tags`, `eval.out_dir`, `eval.dataset_dir`, `eval.seq_len`, `eval.model_step`, `eval.batch_size`, `infrastructure.server`, `infrastructure.slurm.*`. Both commands read the same config.

SLURM resources (`partition`, `gres`, `time`, `mem_per_gpu`, etc.) come from `cfg.infrastructure.slurm` — defaults from `configs/_cluster/<server>.yaml`, overridden in `configs/ctx_eval.yaml`. They get baked into the generated sbatch at setup time.

### What happens

**Step 1 — `setup_ctx_eval.py` (local):**

1. Fetches wandb runs matching `eval.tags` / `eval.negative_tags` (project from `wandb_utils.WANDB_PROJECT`).
2. Writes `{out_dir}/jobs.json` and per-run `yaml_cache/<run_id>.yaml`.
3. Generates `{out_dir}/ctx_eval.sbatch` — SLURM directives from `cfg.infrastructure.slurm`, `--array=0-(N-1)` baked in, activation block + srun line are static (don't vary across pixi clusters). Inspect both files before submitting.

Note: `eval.model_step` must be set explicitly — setup runs locally and can't discover the latest checkpoint on the cluster filesystem.

**Step 2 — `run_ctx_eval.py` (local driver, ssh-automated):**

4. Checks `{out_dir}/jobs.json` + `{out_dir}/ctx_eval.sbatch` exist (errors with a "run setup first" message if not).
5. `version_code` stages the working tree (force-adding `{out_dir}/` so the generated artifacts travel with the branch), commits onto `ctxeval_<name>_<ts>`, pushes to the cemetery remote, restores your local branch + HEAD.
6. SSH to `cfg.infrastructure.server`, clone the branch into `cfg.infrastructure.cemetery_experiments_dir/<branch>`, open a tmux session.
7. In tmux: `cd <experiment_dir>`, export local `WANDB_API_KEY` + `HF_TOKEN` (these propagate to the job via sbatch's default `--export=ALL`).
8. `sbatch {out_dir}/ctx_eval.sbatch` submits the array.
9. Each array task is `srun python src/context_scaling/scripts/ctx_eval.py`:
   - Reads `SLURM_ARRAY_TASK_ID`, picks that row from `jobs.json`.
   - Loads the checkpoint via FSDP + `torch.distributed.checkpoint`.
   - Runs the per-token loss loop on `eval.dataset_dir`.
   - Saves CSV at `{out_dir}/{run_id}_step_{step}.csv`.
   - Resumes the corresponding wandb run and writes the batch-mean loss as a list under `summary["eval/per_position_loss/step{N}_seq{N}"]`.
   - Appends a row to `{out_dir}/index.jsonl` with the flat config.
10. Driver captures the SLURM job ID from the tmux pane and pipes `tail -f slurm-<id>_0.out` into the same pane.

To attach to that pane: `ssh <cluster> -t tmux attach -t <branch>` (or launch with `LOGLEVEL=DEBUG` to attach immediately).

### Files

- `configs/ctx_eval.yaml` — knobs (Hydra config, shared by both commands)
- `src/context_scaling/scripts/setup_ctx_eval.py` — local: wandb fetch + generate sbatch from `cfg.infrastructure.slurm`
- `src/context_scaling/scripts/run_ctx_eval.py` — local: push + ssh + sbatch + tail
- `src/context_scaling/scripts/ctx_eval.py` — cluster python: hydra entrypoint, reads `SLURM_ARRAY_TASK_ID`, runs one eval
- `src/context_scaling/scripts/wandb_utils.py` — `WANDB_PROJECT`, `get_wandb_table`, `upload_mean_loss_to_wandb`
- `{out_dir}/ctx_eval.sbatch` — generated bash (not committed to repo; force-added by `version_code` so it ships with the branch)

---

## Legacy (manual)

`src/context_scaling/scripts/setup_eval.sh` \
which handles \
`src/context_scaling/scripts/setup_eval.py` \
and \
`src/context_scaling/scripts/eval_models.sbatch` \
which handles \
`src/context_scaling/scripts/eval_models.py`

### How to use them

1. setup `src/context_scaling/scripts/setup_eval.sh`
    1. create unique set of neptune tags for grid you want to eval (WARNNG: all runs need to have same number of steps)
    2. update `--tags` and `--out_dir` in `src/context_scaling/scripts/setup_eval.sh`
    3. optionally pass `--model_step` to pin a specific checkpoint step (otherwise uses latest step_* per run)
    4. this script creates a jobs_json, a list[{"jobID","ckpt_path","yaml_config_path","seq_len","model_step"}] for each run. If you rsynced model checkpoints update ckpt_path in the json.
2. setup `src/context_scaling/scripts/eval_models.sbatch`
    1. make sure that `--jobs_json` points to the json created by setup script (model_step is now in jobs.json, `--model_step` on sbatch overrides it)
3. run eval
    1. commit push changes to github
    2. ssh to cluster
    3. git pull
    4. run `bash -l src/context_scaling/scripts/setup_eval.sh` (it modifies number of jobs in slurm array in `eval_models.sbatch`)
    5. run `sbatch src/context_scaling/scripts/eval_models.sbatch`

Note: the legacy flow does **not** upload per-position loss to wandb — that's a feature of the automated `ctx_eval.py` only. The legacy flow just writes the per-token loss CSV.

---

## Loading results in notebooks

Both flows append to `{out_dir}/index.jsonl` with the full flat config. Use the index to filter on any config field:

```python
from src.context_scaling.eval_index import load_eval_index, load_eval_csvs

idx = load_eval_index("path/to/eval_dir")

# filter on any config field with normal pandas
subset = idx[idx["common.kv_heads"] == 1]
subset = subset[subset["common.sequence_length"] == 2048]

# load CSVs with readable labels
labels, dfs = load_eval_csvs(subset, "path/to/eval_dir",
                             label_cols=["common.kv_heads", "common.dmodel"])
```
