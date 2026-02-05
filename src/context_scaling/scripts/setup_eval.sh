#!/bin/bash -l
set -euo pipefail

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

# Creates:
#   lr_grid/main/main.csv
#   lr_grid/main/jobs.json
#   lr_grid/main/yaml_cache/<RUN_ID>.yaml
#
# Optionally updates your sbatch array length if you pass --sbatch_path.
python src/context_scaling/scripts/setup_eval.py \
    --tags context_scaling fineweb_edu WSD_scheduler eval_dmodel_768 \
    --out_dir eval_dmodel_768 \
    --sbatch_path src/context_scaling/scripts/eval_models.sbatch
