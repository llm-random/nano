#!/bin/bash -l

python python src/context_scaling/scripts/setup_eval.py \
    --tags context_scaling fineweb_edu WSD_scheduler lr_grid \
    --out_dir lr_grid/main/df.csv
