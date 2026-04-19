#!/usr/bin/env bash
# Launches k4, k12, k20 in both vanilla and simpleP2 variants via run_exp.py.
# The ^learning_rate grid in each config expands to {7, 9, 11, 13}, i.e. 2^-7 ... 2^-13.
# At k4 (dmodel == base_dmodel) every simpleP scale is 1.0, so simpleP2 and vanilla
# should produce identical curves — useful as a sanity check on the plumbing.
set -euo pipefail

MODELS=(k4 k12 k20)

for model in "${MODELS[@]}"; do
    echo "=== launching simpleP2 ${model} ==="
    pixi run python run_exp.py \
        --config-path=configs/simpleP --config-name="${model}"

    echo "=== launching vanilla ${model} ==="
    pixi run python run_exp.py \
        --config-path=configs/simpleP --config-name="${model}" \
        ~simpleP \
        infrastructure.metric_logger.name="vanilla2_${model}" \
        "infrastructure.metric_logger.tags=[nano, vanilla2, ${model}]"
done
