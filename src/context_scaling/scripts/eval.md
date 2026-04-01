There are 4 scripts

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

### Loading results in notebooks

Each eval run appends to `{out_dir}/index.jsonl` with the full flat config. Use the index to filter on any config field:

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

