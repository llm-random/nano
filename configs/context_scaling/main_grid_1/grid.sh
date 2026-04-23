#!/usr/bin/env bash
# Launches the four [MHA, MQA] x [dense, MoE] axis combinations.
# Each invocation grids batch_size * learning_rate (24 jobs) inside one sbatch
# array on lem hopper:4. simpleP and n_blocks=12 are fixed.
set -euo pipefail

DIR="context_scaling/main_grid_1"

for COMBO in dense_mha dense_mqa moe_mha moe_mqa; do
    echo "=== launching ${COMBO} ==="
    pixi run python run_exp.py --config-path configs --config-name ${DIR}/${COMBO}
done
