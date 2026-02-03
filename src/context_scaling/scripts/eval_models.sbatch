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
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
#-------- SCRIPT END --------

srun python src/context_scaling/scripts/eval_models.py \
    --tags context_scaling fineweb_edu WSD_scheduler lr_grid \
    --dataset_dir /storage_nvme_1/llm-random/datasets/c4/long_context_2048n8192 \
    --out_csv_format "dmodel,kv_heads,trainer/learning_rate,model_seed,data_seed" \
    --out_dir lr_grid \
    --tmp_ckpt_path /storage_nvme_4/nano/models/from_lem/lr_grid \
    --model_cluster lem \
