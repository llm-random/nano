#!/bin/bash

PROJECT_HOME_PATH=/storage_nvme_4/projected_compression
SVDLLM_HOME_PATH=/storage_nvme_2/mstefaniak/svdllm/wip
export HF_HOME=$PROJECT_HOME_PATH/hf_cache
export HF_DATASETS_TRUST_REMOTE_CODE=1

# 1. Check if your path variable is set
if [ -z "$SVDLLM_HOME_PATH" ]; then
    echo "Error: SVDLLM_HOME_PATH is not set. Please export it first."
    exit 1
fi

# 2. List of sample sizes to iterate through
SAMPLES=(512 1024 2048 4096 8192 16384 32768)

# 3. Loop through each sample size
for n in "${SAMPLES[@]}"; do
    echo "----------------------------------------"
    echo "Running SVDLLM with whitening_nsamples: $n"
    echo "Saving to: $SVDLLM_HOME_PATH/$n"
    echo "----------------------------------------"

    python SVDLLM.py \
        --model meta-llama/Llama-3.1-8B \
        --step 1 \
        --ratio 0.3 \
        --whitening_nsamples "$n" \
        --dataset wikitext2 \
        --seed 3 \
        --model_seq_len 2048 \
        --save_path "$SVDLLM_HOME_PATH/$n"

    # Check if the last command failed
    if [ $? -ne 0 ]; then
        echo "Error detected at nsamples=$n. Stopping script."
        exit 1
    fi
done

echo "All jobs completed successfully."