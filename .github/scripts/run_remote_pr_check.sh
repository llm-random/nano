#!/bin/bash -l

#SBATCH --cpus-per-gpu=16
#SBATCH --gres=gpu:2
#SBATCH --job-name=pr_check_entropy
#SBATCH --mem-per-gpu=125G
#SBATCH --nodes=1
#SBATCH --partition=h100
#SBATCH --time=00:10:00

set -euo pipefail  # exit on error, treat unset vars as error
set -x             # print commands for debugging

# --- Resolve PR_TEST_CONFIG_NAME from array index ---

if [ -z "${PR_TEST_CONFIGS_FILE:-}" ]; then
    echo "Error: PR_TEST_CONFIGS_FILE not set"
    exit 1
fi

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID not set (are you running as an array job?)"
    exit 1
fi

# SLURM_ARRAY_TASK_ID is 0-based; sed is 1-based
line=$((SLURM_ARRAY_TASK_ID + 1))
PR_TEST_CONFIG_NAME=$(sed -n "${line}p" "$PR_TEST_CONFIGS_FILE" || true)

if [ -z "$PR_TEST_CONFIG_NAME" ]; then
    echo "Error: failed to resolve config for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID from $PR_TEST_CONFIGS_FILE"
    exit 1
fi

echo "Running CI check for config: $PR_TEST_CONFIG_NAME"

#---------- SCRIPT ----------
export PROJECT_HOME_PATH=/storage_ssd_1/nano
export HF_HOME=$PROJECT_HOME_PATH/hf_cache
export HYDRA_FULL_ERROR=1
export PIXI_HOME=/storage_ssd_1/nano/pixi
export PATH="$PIXI_HOME/bin:$PATH"
export XDG_DATA_HOME="$PIXI_HOME/data"
export XDG_CACHE_HOME="$PIXI_HOME/cache"
export XDG_STATE_HOME="$PIXI_HOME/state"

# Save current directory and setup pixi
cd "$PIXI_HOME" || { echo "Failed to cd to $PIXI_HOME"; exit 1; }
eval "$(pixi shell-hook)" || { echo "Failed to run pixi shell-hook"; exit 1; }
cd -
#-------- SCRIPT END --------

export MASTER_ADDR=$(scontrol show hostname "${SLURM_NODELIST}" | head -n 1)
export MASTER_PORT=$((40000 + ${SLURM_JOB_ID} % 10000))

echo "Running training with config: $PR_TEST_CONFIG_NAME"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

srun torchrun --nnodes="${SLURM_NNODES}" \
  --nproc-per-node="${SLURM_GPUS_ON_NODE}" \
  --rdzv-id="${SLURM_JOBID}" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  main.py \
    --config-path=configs/pr_tests \
    --config-name="$PR_TEST_CONFIG_NAME"
