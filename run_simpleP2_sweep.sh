#!/usr/bin/env bash
# Launches k4, k8, k16, k24 in both simpleP_long and vanilla_long variants via run_exp.py.
# The ^learning_rate grid expands to {4, 5, 6, 7, 8, 9, 10, 11}, i.e. 2^-4 ... 2^-11.
# Runs 40_000 steps (previous sweep was 5_000) within the same 4h slurm budget.
# At k4 (dmodel == base_dmodel) every simpleP scale is 1.0, so simpleP_long and vanilla_long
# should produce identical curves — useful as a sanity check on the plumbing.
set -euo pipefail

MODELS=(k4 k8 k16 k24)

for model in "${MODELS[@]}"; do
    echo "=== launching simpleP_long ${model} ==="
    pixi run python run_exp.py \
        --config-path=configs/simpleP --config-name="${model}"

    echo "=== launching vanilla_long ${model} ==="
    pixi run python run_exp.py \
        --config-path=configs/simpleP --config-name="${model}" \
        ~simpleP \
        infrastructure.metric_logger.name="vanilla_long_${model}" \
        "infrastructure.metric_logger.tags=[nano, vanilla_long, ${model}]"
done
