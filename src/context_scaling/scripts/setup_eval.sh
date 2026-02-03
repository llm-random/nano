#!/bin/bash -l

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

python src/context_scaling/scripts/setup_eval.py \
    --tags context_scaling fineweb_edu WSD_scheduler lr_grid \
    --csv_path lr_grid/main/df.csv
