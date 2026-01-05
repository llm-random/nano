#!/bin/bash

PROJECT_HOME_PATH=/storage_nvme_4/projected_compression
SVDLLM_HOME_PATH=/storage_nvme_2/mstefaniak/svdllm/wip
export HF_HOME=$PROJECT_HOME_PATH/hf_cache
export HF_DATASETS_TRUST_REMOTE_CODE=1

# Ensure the script stops if a command fails (optional, remove if you want it to keep going)
set -e

# Check if the environment variable is set
if [ -z "$SVDLLM_HOME_PATH" ]; then
    echo "Error: SVDLLM_HOME_PATH is not set."
    exit 1
fi

# List of sizes to iterate over
# SIZES=(512 1024 2048 4096 8192)
SIZES=(16384 32768)

echo "Starting sequential execution..."

for size in "${SIZES[@]}"; do
    echo "Running SVDLLM for size: $size"
    python SVDLLM.py --step 4 --model_path "$SVDLLM_HOME_PATH/$size/meta_llama_Llama_3.1_8B_whitening_only_0.5.pt"
    echo "Completed size: $size"
    echo "-----------------------------------"
done

echo "All tasks finished successfully."