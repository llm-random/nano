#!/bin/bash
# Run one generated SliceGPT smoke scenario.
#
# Everything that varies by user or cluster comes from the environment, with defaults
# that work anywhere:
#
#   SMOKE_CONFIG_DIR  generated configs        (default: smoke/generated)
#   SMOKE_RESULTS     where logs are written   (default: ./smoke_results)
#   SMOKE_GPUS        processes for torchrun   (default: 1, or 2 for the rft/lr scenarios)
#
# The environment itself (pixi, HF caches) is NOT activated here: activate it the way
# you normally do for this cluster first, so this script stays cluster-agnostic.
#
#   python smoke/generate.py --work-dir /scratch/$USER/slicegpt_smoke
#   bash smoke/run_smoke.sh s1_invariance
#   SMOKE_GPUS=2 bash smoke/run_smoke.sh s4_rft
#
# With no argument, lists the scenarios that have been generated.
set -euo pipefail

CONFIG_DIR="${SMOKE_CONFIG_DIR:-smoke/generated}"
RESULTS="${SMOKE_RESULTS:-./smoke_results}"

if [ $# -lt 1 ]; then
  echo "usage: bash smoke/run_smoke.sh <scenario>"
  if [ -d "$CONFIG_DIR" ]; then
    echo "generated scenarios in $CONFIG_DIR:"
    for f in "$CONFIG_DIR"/*.yaml; do [ -e "$f" ] && echo "  $(basename "${f%.yaml}")"; done
  else
    echo "no configs yet - run: python smoke/generate.py --work-dir <dir>"
  fi
  exit 1
fi

SCENARIO="$1"
CONFIG="$CONFIG_DIR/$SCENARIO.yaml"
[ -f "$CONFIG" ] || { echo "no such scenario: $CONFIG (run smoke/generate.py first)"; exit 1; }

# the recovery-training scenarios are the multi-GPU ones
case "$SCENARIO" in
  s4_*|s5_*) DEFAULT_GPUS=2 ;;
  *)         DEFAULT_GPUS=1 ;;
esac
GPUS="${SMOKE_GPUS:-$DEFAULT_GPUS}"

mkdir -p "$RESULTS"
LOG="$RESULTS/$SCENARIO.log"
echo "=== $SCENARIO : $GPUS GPU(s) -> $LOG"

if [ "$GPUS" -gt 1 ]; then
  torchrun --nnodes=1 --nproc-per-node="$GPUS" --rdzv-backend=c10d \
    --rdzv-endpoint="localhost:$((29500 + RANDOM % 1000))" \
    main.py --config-path="$(cd "$CONFIG_DIR" && pwd)" --config-name="$SCENARIO" > "$LOG" 2>&1
else
  python main.py --config-path="$(cd "$CONFIG_DIR" && pwd)" --config-name="$SCENARIO" > "$LOG" 2>&1
fi
rc=$?
echo "=== $SCENARIO exit $rc  (log: $LOG)"
exit $rc
