#!/usr/bin/env bash
# Submit the MoE vs dense step profiles to lem: configs/jm/mfu/h{12,16,20}_sl1024_{moe,dense}_profile.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."
for h in 12 16 20; do
  for kind in moe dense; do
    pixi run python run_exp.py --config-path configs/jm/mfu --config-name "h${h}_sl1024_${kind}_profile"
  done
done
