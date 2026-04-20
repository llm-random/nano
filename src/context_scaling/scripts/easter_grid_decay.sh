#!/bin/bash -l
set -euo pipefail

pixi run python src/context_scaling/scripts/run_decay.py \
    --tags easter_grid \
    --negative_tags decay \
    --out_dir easter_grid_decay \
    --steps 32000 64000 96000 128000 160000 192000 224000 256000 288000 \
    --decay_fraction 0.1 \
    --train_data_seed 456 \
    --eval_config configs/_eval/incontext.yaml \
    --job_name easter_decay \
    --max_concurrent_jobs 40 \
    --slurm_time "12:00:00" \
    --submit
