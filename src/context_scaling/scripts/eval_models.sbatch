#!/bin/bash -l

#SBATCH --array=0-0
#SBATCH --cpus-per-gpu=14
#SBATCH --gres=gpu:1
#SBATCH --job-name=tiny_remote
#SBATCH --mem-per-gpu=90G
#SBATCH --nodes=1
#SBATCH --partition=h100
#SBATCH --time=00:30:00

set -euo pipefail

#---------- SCRIPT ----------
export PROJECT_HOME_PATH=/storage_nvme_4/nano
export HF_HOME=$PROJECT_HOME_PATH/hf_cache
export HYDRA_FULL_ERROR=1
export PIXI_HOME=/storage_nvme_4/nano/pixi
export PATH="$PIXI_HOME/bin:$PATH"
export XDG_DATA_HOME="$PIXI_HOME/data"
export XDG_CACHE_HOME="$PIXI_HOME/cache"
export XDG_STATE_HOME="$PIXI_HOME/state"
cd "$PIXI_HOME"
eval "$(pixi shell-hook)"
cd -
#-------- SCRIPT END --------

ls -la /home/j321m/nano/src/context_scaling/scripts

srun python src/context_scaling/eval_models.py \
    --tags context_scaling fineweb_edu WSD_scheduler lr_grid \
    --dataset_dir /storage_nvme_4/llm-random/datasets/fineweb/train \
    --out_csv_format "dmodel,kv_heads,learning_rate,model_seed,data_seed" \
    --out_dir lr_grid \
    --tmp_ckpt_path /storage_nvme_4/nano/models/from_lem/lr_grid
