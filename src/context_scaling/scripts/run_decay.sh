#!/bin/bash -l
set -euo pipefail

# Smoke test for run_decay.py against base runs produced by configs/test_decay.yaml
# (tiny model, 1001 training steps, checkpoints every 100 steps on entropy).
#
# Exercises the new features:
#   - --eval_config: custom evaluator block overrides the base run's evaluator
#   - --decay_fraction 0.0: eval-only mode (no LR decay, zero training steps,
#     trainer post-loop hook fires the evaluator once per checkpoint)

pixi run python src/context_scaling/scripts/run_decay.py \
    --tags test_decay \
    --negative_tags decay \
    --out_dir test_decay_grid \
    --steps 200 500 \
    --decay_fraction 0.1 \
    --train_data_seed 999 \
    --eval_config configs/_downstream_eval/test_tasks.yaml \
    --job_name test_decay_eval \
    --max_concurrent_jobs 2 \
    --slurm_time "00:10:00" \
    --submit
