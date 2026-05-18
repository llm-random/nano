#!/bin/bash -l

#SBATCH --array=0-0
#SBATCH --requeue
#SBATCH --cpus-per-gpu=14
#SBATCH --gres=gpu:4
#SBATCH --job-name=paralel_eval
#SBATCH --mem-per-gpu=90G
#SBATCH --nodes=1
#SBATCH --partition=h100
#SBATCH --time=01:00:00

set -euo pipefail

#---------- SCRIPT ----------
ml CUDA/12.4.0
export PROJECT_HOME_PATH=/storage_nvme_2/nano/$USER
export HYDRA_FULL_ERROR=1
export PIXI_HOME=$PROJECT_HOME_PATH/pixi
export HF_HOME=$PROJECT_HOME_PATH/hf_cache
export PATH="$HOME/.pixi/bin:$PATH"

cd "$PIXI_HOME"
eval "$(pixi shell-hook)"
cd -

# --- avoid Triton/Inductor cache collisions across array tasks ---
export TORCH_COMPILE_DISABLE=1
# ---

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + (SLURM_ARRAY_JOB_ID % 20000) + SLURM_ARRAY_TASK_ID))
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0

#-------- SCRIPT END --------

srun --export=ALL --ntasks=1 --nodes=1 --gpus=1 python src/context_scaling/scripts/eval_models.py \
    --dataset_dir /storage_nvme_1/llm-random/datasets/c4/long_context_2048n8192 \
    --jobs_json 2503_restart \
    --out_dir 2503_restart \
    --seq_len 2048 \
    --batch_size 32 \
