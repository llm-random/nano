#!/bin/bash -l
set -euo pipefail

# Smoke test for run_decay.py against MoE base runs produced by
# configs/test_decay_moe.yaml (tiny MoE model, 1001 training steps,
# checkpoints every 100 steps on entropy).
#
# Exercises the same decay features as run_decay.sh but on MoE checkpoints:
#   - --eval_config: custom evaluator block overrides the base run's evaluator
#   - --decay_fraction 0.1: short linear LR decay from intermediate checkpoints

pixi run python src/context_scaling/scripts/run_decay.py \
    --tags test_decay_moe \
    --negative_tags decay \
    --out_dir test_decay_moe_grid \
    --steps 200 500 \
    --decay_fraction 0.1 \
    --train_data_seed 999 \
    --eval_config configs/_eval/test_tasks.yaml \
    --job_name test_decay_moe_eval \
    --max_concurrent_jobs 2 \
    --slurm_time "00:10:00" \
    --submit
