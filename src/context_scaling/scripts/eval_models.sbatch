#!/bin/bash -l

#SBATCH --array=0-19
#SBATCH --cpus-per-gpu=14
#SBATCH --gres=gpu:1
#SBATCH --job-name=tiny_remote
#SBATCH --mem-per-gpu=90G
#SBATCH --nodes=1
#SBATCH --partition=h100
#SBATCH --time=00:30:00

set -euo pipefail

#---------- SCRIPT ----------
ml CUDA/12.4.0
export PROJECT_HOME_PATH=/lustre/pd01/plgrid/plgllmefficont2/nano/context_scaling
export HF_HOME=$PROJECT_HOME_PATH/hf_cache
export HYDRA_FULL_ERROR=1
export PIXI_HOME=$PROJECT_HOME_PATH/pixi
export PATH="$HOME/.pixi/bin:$PATH"
export XDG_DATA_HOME="$PROJECT_HOME_PATH/data"
export XDG_CACHE_HOME="$PROJECT_HOME_PATH/cache"
export XDG_STATE_HOME="$PROJECT_HOME_PATH/state"
cd "$PIXI_HOME"
eval "$(pixi shell-hook)"
cd -

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + (SLURM_ARRAY_JOB_ID % 20000) + SLURM_ARRAY_TASK_ID))
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0

#-------- SCRIPT END --------

srun --export=ALL --ntasks=1 --nodes=1 --gpus=1 python src/context_scaling/scripts/eval_models.py \
    --dataset_dir /lustre/pd01/plgrid/plgllmefficont2/datasets/c4/long_context_2048n8192 \
    --out_csv_format "dmodel,kv_heads,trainer/learning_rate,model_seed,data_seed" \
    --out_dir lr_grid \
    --seq_len 2048 \
    --batch_size 32 \
    --model_step 320000
