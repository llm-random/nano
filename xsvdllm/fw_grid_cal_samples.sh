#!/bin/bash

# --- Environment Setup ---
PROJECT_HOME_PATH=/storage_nvme_4/projected_compression
SVDLLM_HOME_PATH=/storage_nvme_2/mstefaniak/svdllm/wip
export HF_HOME=$PROJECT_HOME_PATH/hf_cache
export HF_DATASETS_TRUST_REMOTE_CODE=1

# 1. Check if your path variable is set
if [ -z "$SVDLLM_HOME_PATH" ]; then
    echo "Error: SVDLLM_HOME_PATH is not set. Please export it first."
    exit 1
fi

# --- Main Loop ---
SAMPLES=(256 512 1024 2048 4096 8192 16384 32768)

for N in "${SAMPLES[@]}"; do
    echo "Running SVDLLM with whitening_nsamples = $N..."

    python SVDLLM.py \
        --model meta-llama/Llama-3.1-8B \
        --step 1 \
        --ratio 0.5 \
        --whitening_nsamples "$N" \
        --dataset fineweb-edu \
        --data_path /storage_nvme_4/llm-random/datasets/fineweb/train \
        --eval_batch_size 1 \
        --seed 3 \
        --model_seq_len 2048 \
        --save_path "$SVDLLM_HOME_PATH/dec/$N"
    
    # Check for failure
    if [ $? -ne 0 ]; then
        echo "Error: Failed at whitening_nsamples = $N"
        exit 1
    fi

    echo "Finished whitening_nsamples = $N"
    echo "----------------------------------------"
done

echo "All runs completed successfully."