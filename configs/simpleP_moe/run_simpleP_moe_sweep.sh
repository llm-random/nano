#!/usr/bin/env bash
# Launches k4, k8, k12, k16, k20, k24 in both simpleP_moe and vanilla_moe variants via run_exp.py.
# At k4 (dmodel == base_dmodel) every simpleP scale is 1.0, so simpleP_moe and vanilla_moe
# should produce identical curves — useful as a sanity check on the plumbing.
set -euo pipefail

# MODELS=(k4 k8 k12 k16 k20 k24)
MODELS=(k12 k8 k4 k24 k20 k16)

for model in "${MODELS[@]}"; do
    echo "=== launching simpleP_moe ${model} ==="
    pixi run python run_exp.py --config-name="simpleP_moe/${model}"

    echo "=== launching vanilla_moe ${model} ==="
    pixi run python run_exp.py --config-name="simpleP_moe/${model}" \
        ~simpleP \
        infrastructure.metric_logger.name="vanilla_moe_${model}" \
        "infrastructure.metric_logger.tags=[nano, vanilla_moe3, ${model}]"
done
