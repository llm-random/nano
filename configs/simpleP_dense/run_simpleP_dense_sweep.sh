#!/usr/bin/env bash
# Launches k4, k8, k12, k16, k20, k24 in the simpleP_dense variant via run_exp.py.
# Vanilla baselines were already run separately and are not re-launched here.
set -euo pipefail

# MODELS=(k4 k8 k12 k16 k20 k24)
MODELS=(k12 k4 k24)

for model in "${MODELS[@]}"; do
    echo "=== launching simpleP_dense ${model} ==="
    pixi run python run_exp.py --config-name="simpleP_dense/${model}"
done
