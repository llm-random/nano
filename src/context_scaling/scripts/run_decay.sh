#!/bin/bash -l
set -euo pipefail

# Small test harness for run_decay.py.
# Assumes a base run produced by `configs/test_decay.yaml` already exists in wandb
# (tagged "test_decay", 1001 training steps, checkpoints every 100 steps on entropy).

pixi run python src/context_scaling/scripts/run_decay.py \
    --tags test_decay \
    --negative_tags decay \
    --out_dir test_decay_grid \
    --steps 200 500 \
    --decay_fraction 0.1 \
    --train_data_seed 999 \
    --save_ckpt_base /storage_nvme_4/nano/models/test_decay_decay \
    --job_name test_decay_decay \
    --max_concurrent_jobs 2 \
    --slurm_time "00:10:00" \
    --submit
