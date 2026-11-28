#!/usr/bin/env bash
# Submit lem profile jobs from configs/jm/mfu: the config names given, else h{12,16,20}_sl1024_{moe,dense}_profile.
# AFTER=<job id> (e.g. a pack job) makes them start only if it succeeds. A failed submission is reported, not fatal.
set -uo pipefail
cd "$(dirname "$0")/.."
configs=("$@")
if [ ${#configs[@]} -eq 0 ]; then
  for h in 12 16 20; do
    for kind in moe dense; do
      configs+=("h${h}_sl1024_${kind}_profile")
    done
  done
fi
after=()
if [ -n "${AFTER:-}" ]; then
  after=("++infrastructure.slurm.dependency=afterok:$AFTER" "++infrastructure.slurm.kill_on_invalid_dep=yes")
fi
failed=()
for name in "${configs[@]}"; do
  pixi run python run_exp.py --config-path configs/jm/mfu --config-name "$name" ${after[@]+"${after[@]}"} || failed+=("$name")
done
if [ ${#failed[@]} -gt 0 ]; then
  echo "Failed to submit: ${failed[*]}"
  exit 1
fi
