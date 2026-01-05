#!/bin/bash

# example of compressing LLaMA-7B with SVDLLM
FINE_TUNE_PATH="."
# run data whitening with 20% compression ratio
python SVDLLM.py --model jeffwan/llama-7b-hf --step 1 --ratio 0.2 --whitening_nsamples 256 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path .
## you can also run the following command for low-resource gpu (ex. llama 7b will only need 15G gpu memory to compress) or to compress large-scale llm (ex. llama 65b)
# python SVDLLM.py --model jeffwan/llama-7b-hf --step 1 --ratio 0.2 --whitening_nsamples 256 --dataset wikitext2 --model_seq_len 2048 --save_path ./ --run_low_resource
python SVDLLM.py --step 4 --model_path jeffwan_llama_7b_hf_whitening_only_0.8.pt
# finetune the compressed model with lora
python utils/LoRA.py --prune_model  --data_path yahma/alpaca-cleaned --output_dir $FINE_TUNE_PATH/first_half --lora_target_modules q_u_proj,k_u_proj,v_u_proj,o_u_proj,gate_u_proj,down_u_proj,up_u_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64
python SVDLLM.py --model_path jeffwan_llama_7b_hf_whitening_only_0.8.pt --lora $FINE_TUNE_PATH/first_half /first_half --step 4
python utils/LoRA.py --prune_model $FINE_TUNE_PATH/first_half/merge.pt --data_path yahma/alpaca-cleaned --output_dir $FINE_TUNE_PATH/second_half --lora_target_modules q_v_proj,k_v_proj,v_v_proj,o_v_proj,gate_v_proj,down_v_proj,up_v_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64
python SVDLLM.py --model_path jeffwan_llama_7b_hf_whitening_only_0.8.pt --lora $FINE_TUNE_PATH/first_half /first_half --step 4
python SVDLLM.py --model_path $FINE_TUNE_PATH/first_half/merge.pt --lora $FINE_TUNE_PATH/second_half --step 4




#!/bin/bash

# example of compressing LLaMA-7B with SVDLLM
FINE_TUNE_PATH="."
# run data whitening with 20% compression ratio

python SVDLLM.py --model meta-llama/Llama-2-7b-hf --step 1 --ratio 0.2 --whitening_nsamples 256 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path $SVDLLM_HOME_PATH
python SVDLLM.py --model meta-llama/Llama-2-7b-hf --step 1 --ratio 0.0 --whitening_nsamples 256 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path $SVDLLM_HOME_PATH
python SVDLLM.py --model jeffwan/llama-7b-hf --step 1 --ratio 0.2 --whitening_nsamples 256 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path $SVDLLM_HOME_PATH

python SVDLLM.py --model meta-llama/Llama-3.1-8B --step 1 --ratio 0.2 --whitening_nsamples 256 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path $SVDLLM_HOME_PATH

python SVDLLM.py --model meta-llama/Llama-3.2-1B --step 1 --ratio 0.2 --whitening_nsamples 256 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path $SVDLLM_HOME_PATH
python SVDLLM.py --model meta-llama/Llama-3.2-1B --step 1 --ratio 0.0 --whitening_nsamples 256 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path $SVDLLM_HOME_PATH
python SVDLLM.py --model meta-llama/Llama-3.2-1B --step 1 --ratio -2.0 --whitening_nsamples 256 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path $SVDLLM_HOME_PATH


python SVDLLM.py --model meta-llama/Llama-3.1-8B --step 1 --ratio 0.5 --whitening_nsamples 256 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path $SVDLLM_HOME_PATH/256
python SVDLLM.py --model meta-llama/Llama-3.1-8B --step 1 --ratio 0.5 --whitening_nsamples 512 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path $SVDLLM_HOME_PATH/512
python SVDLLM.py --model meta-llama/Llama-3.1-8B --step 1 --ratio 0.5 --whitening_nsamples 1024 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path $SVDLLM_HOME_PATH/1024
python SVDLLM.py --model meta-llama/Llama-3.1-8B --step 1 --ratio 0.5 --whitening_nsamples 2048 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path $SVDLLM_HOME_PATH/2048
python SVDLLM.py --model meta-llama/Llama-3.1-8B --step 1 --ratio 0.5 --whitening_nsamples 4096 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path $SVDLLM_HOME_PATH/4096
python SVDLLM.py --model meta-llama/Llama-3.1-8B --step 1 --ratio 0.5 --whitening_nsamples 8192 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path $SVDLLM_HOME_PATH/8192
python SVDLLM.py --model meta-llama/Llama-3.1-8B --step 1 --ratio 0.5 --whitening_nsamples 16384 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path $SVDLLM_HOME_PATH/16384
python SVDLLM.py --model meta-llama/Llama-3.1-8B --step 1 --ratio 0.5 --whitening_nsamples 32768 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path $SVDLLM_HOME_PATH/32768

mkdir $SVDLLM_HOME_PATH/256
mkdir $SVDLLM_HOME_PATH/512
mkdir $SVDLLM_HOME_PATH/1024
mkdir $SVDLLM_HOME_PATH/2048
mkdir $SVDLLM_HOME_PATH/4096
mkdir $SVDLLM_HOME_PATH/8192
mkdir $SVDLLM_HOME_PATH/16384
mkdir $SVDLLM_HOME_PATH/32768

python SVDLLM.py --step 4 --model_path $SVDLLM_HOME_PATH/256/meta_llama_Llama_3.1_8B_whitening_only_0.5.pt
python SVDLLM.py --step 4 --model_path $SVDLLM_HOME_PATH/256/meta_llama_Llama_3.1_8B_whitening_only_0.8.pt
python SVDLLM.py --step 4 --model_path $SVDLLM_HOME_PATH/meta_llama_Llama_3.2_1B_whitening_only_0.8.pt

# python SVDLLM.py --step 4 --model_path $SVDLLM_HOME_PATH/jeffwan_llama_7b_hf_whitening_only_0.8.pt
python SVDLLM.py --step 4 --model_path $SVDLLM_HOME_PATH/meta_llama_Llama_3.2_1B_whitening_only_0.8.pt
python SVDLLM.py --step 4 --model_path $SVDLLM_HOME_PATH/meta_llama_Llama_3.2_1B_whitening_only_0.99.pt
python SVDLLM.py --step 4 --model_path $SVDLLM_HOME_PATH/meta_llama_Llama_3.2_1B_whitening_only_1.0.pt
python SVDLLM.py --step 4 --model_path $SVDLLM_HOME_PATH/meta_llama_Llama_3.2_1B_whitening_only_2.0.pt
python SVDLLM.py --step 4 --model_path original --model meta-llama/Llama-3.2-1B

python SVDLLM.py --step 4 --model_path $SVDLLM_HOME_PATH/meta_llama_Llama_3.1_8B_whitening_only_0.5.pt
python SVDLLM.py --step 4 --model_path $SVDLLM_HOME_PATH/meta_llama_Llama_3.1_8B_whitening_only_0.8.pt
python SVDLLM.py --step 4 --model_path $SVDLLM_HOME_PATH/meta_llama_Llama_3.1_8B_whitening_only_1.0.pt
python SVDLLM.py --step 4 --model_path original --model meta-llama/Llama-3.1-8B 
 

python SVDLLM.py --step 4 --model_path $SVDLLM_HOME_PATH/meta_llama_Llama_2_7b_hf_whitening_only_0.8.pt
python SVDLLM.py --step 4 --model_path $SVDLLM_HOME_PATH/meta_llama_Llama_2_7b_hf_whitening_only_1.0.pt
python SVDLLM.py --step 4 --model_path $SVDLLM_HOME_PATH/meta_llama_Llama_2_7b_hf_whitening_only_2.0.pt
python SVDLLM.py --step 4 --model_path original --model meta-llama/Llama-2-7b-hf

ls -al $SVDLLM_HOME_PATH
mkdir $SVDLLM_HOME_PATH/256

# finetune the compressed model with lora

# python utils/LoRA.py --prune_model $SVDLLM_HOME_PATH/jeffwan_llama_7b_hf_whitening_only_0.8.pt --data_path yahma/alpaca-cleaned --output_dir $SVDLLM_HOME_PATH/first_half --lora_target_modules q_u_proj,k_u_proj,v_u_proj,o_u_proj,gate_u_proj,down_u_proj,up_u_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64
# python SVDLLM.py --model_path $SVDLLM_HOME_PATH/jeffwan_llama_7b_hf_whitening_only_0.8.pt --lora $FINE_TUNE_PATH/first_half /first_half --step 4

# python utils/LoRA.py --prune_model $FINE_TUNE_PATH/first_half/merge.pt --data_path yahma/alpaca-cleaned --output_dir $FINE_TUNE_PATH/second_half --lora_target_modules q_v_proj,k_v_proj,v_v_proj,o_v_proj,gate_v_proj,down_v_proj,up_v_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64
# python SVDLLM.py --model_path jeffwan_llama_7b_hf_whitening_only_0.8.pt --lora $FINE_TUNE_PATH/first_half /first_half --step 4
# python SVDLLM.py --model_path $FINE_TUNE_PATH/first_half/merge.pt --lora $FINE_TUNE_PATH/second_half --step 4



python utils/LoRA.py --prune_model $SVDLLM_HOME_PATH/meta_llama_Llama_3.2_1B_whitening_only_0.8.pt --data_path yahma/alpaca-cleaned --output_dir $SVDLLM_HOME_PATH/first_half --lora_target_modules q_proj.v_proj,k_proj.v_proj,v_proj.v_proj,o_proj.v_proj,gate_proj.v_proj,up_proj.v_proj,down_proj.v_proj,q_proj.u_proj,k_proj.u_proj,v_proj.u_proj,o_proj.u_proj,gate_proj.u_proj,up_proj.u_proj,down_proj.u_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64
python SVDLLM.py --model_path $SVDLLM_HOME_PATH/meta_llama_Llama_3.2_1B_whitening_only_0.8.pt --lora $SVDLLM_HOME_PATH/first_half/checkpoint-2331 --step 4
python SVDLLM.py --model_path $SVDLLM_HOME_PATH/first_half/checkpoint-2331/merge.pt --lora $SVDLLM_HOME_PATH/second_half/checkpoint-1600 --step 4



python SVDLLM.py --model meta-llama/Llama-3.1-8B --step 1 --ratio 0.2 --whitening_nsamples 256 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path $SVDLLM_HOME_PATH
python SVDLLM.py \
    --model meta-llama/Llama-3.1-8B \
    --step 1 \
    --ratio 0.0 \
    --whitening_nsamples 256 \
    --dataset fineweb-edu \
    --data_path /storage_nvme_4/llm-random/datasets/fineweb/train \
    --eval_batch_size 4 \
    --seed 3 \
    --model_seq_len 2048 \
    --save_path $SVDLLM_HOME_PATH/comp_fw


python SVDLLM.py --step 4 --dataset fineweb-edu --data_path /storage_nvme_4/llm-random/datasets/fineweb/train --whitening_nsamples 64 --eval_batch_size 1 --model_path $SVDLLM_HOME_PATH/comp_fw/meta_llama_Llama_3.1_8B_whitening_only_0.8.pt
python SVDLLM.py --step 4 --model_path $SVDLLM_HOME_PATH/comp_fw/meta_llama_Llama_3.1_8B_whitening_only_0.8.pt

python SVDLLM.py --step 4 --dataset fineweb-edu --data_path /storage_nvme_4/llm-random/datasets/fineweb/train --whitening_nsamples 64 --eval_batch_size 1 --model_path $SVDLLM_HOME_PATH/comp_fw2/meta_llama_Llama_3.1_8B_whitening_only_0.8.pt
python SVDLLM.py --step 4 --model_path $SVDLLM_HOME_PATH/comp_fw2/meta_llama_Llama_3.1_8B_whitening_only_0.8.pt

python SVDLLM.py --step 4 --dataset fineweb-edu --data_path /storage_nvme_4/llm-random/datasets/fineweb/train --whitening_nsamples 64 --eval_batch_size 1 --model_path original --model meta-llama/Llama-2-7b-hf
python SVDLLM.py --step 4 --dataset fineweb-edu --data_path /storage_nvme_4/llm-random/datasets/fineweb/train --whitening_nsamples 64 --eval_batch_size 1 --model_path original --model meta-llama/Llama-3.1-8B 

    --tasks "hellaswag,piqa,arc_easy,winogrande" \
python eval_svd_checkpoint.py \
    --checkpoint_path "/storage_nvme_2/mstefaniak/svdllm/wip/ret/test_hand_save" \
    --tasks "winogrande" \
    --output_file et1.json\
    --num_fewshot 5 \
    --batch_size 8


python SVDLLM.py --step 4 --model_path $SVDLLM_HOME_PATH/meta_llama_Llama_3.1_8B_whitening_only_0.8.pt

python utils/LoRA.py --prune_model $SVDLLM_HOME_PATH/meta_llama_Llama_3.1_8B_whitening_only_0.8.pt --data_path yahma/alpaca-cleaned --output_dir $SVDLLM_HOME_PATH/first_half --lora_target_modules q_proj.u_proj,k_proj.u_proj,v_proj.u_proj,o_proj.u_proj,gate_proj.u_proj,up_proj.u_proj,down_proj.u_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64

python SVDLLM.py --model_path $SVDLLM_HOME_PATH/meta_llama_Llama_3.1_8B_whitening_only_0.8.pt --lora $SVDLLM_HOME_PATH/first_half/checkpoint-2331 --step 4

python utils/LoRA.py --prune_model $SVDLLM_HOME_PATH/first_half/checkpoint-2331/merge.pt --data_path yahma/alpaca-cleaned --output_dir $SVDLLM_HOME_PATH/second_half --lora_target_modules q_proj.v_proj,k_proj.v_proj,v_proj.v_proj,o_proj.v_proj,gate_proj.v_proj,up_proj.v_proj,down_proj.v_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64

python SVDLLM.py --model_path $SVDLLM_HOME_PATH/meta_llama_Llama_3.1_8B_whitening_only_0.8.pt --lora $SVDLLM_HOME_PATH/first_half/checkpoint-2331 --step 4

python SVDLLM.py --model_path $SVDLLM_HOME_PATH/first_half/checkpoint-2331/merge.pt --lora $SVDLLM_HOME_PATH/second_half/checkpoint-2331 --step 4

# /checkpoint-2331/


python utils/LoRA.py --prune_model $SVDLLM_HOME_PATH/reproduce/meta_llama_Llama_3.1_8B_whitening_only_0.8.pt --data_path yahma/alpaca-cleaned --output_dir $SVDLLM_HOME_PATH/both_half --lora_target_modules q_proj.u_proj,k_proj.u_proj,v_proj.u_proj,o_proj.u_proj,gate_proj.u_proj,up_proj.u_proj,down_proj.u_proj,q_proj.v_proj,k_proj.v_proj,v_proj.v_proj,o_proj.v_proj,gate_proj.v_proj,up_proj.v_proj,down_proj.v_proj --lora_r 8 --num_epochs 6 --learning_rate 1e-4 --batch_size 64

python utils/LoRA.py --prune_model $SVDLLM_HOME_PATH/reproduce/meta_llama_Llama_3.1_8B_whitening_only_0.8.pt --data_path yahma/alpaca-cleaned --output_dir $SVDLLM_HOME_PATH/long_test --lora_target_modules q_proj.u_proj,k_proj.u_proj,v_proj.u_proj,o_proj.u_proj,gate_proj.u_proj,up_proj.u_proj,down_proj.u_proj,q_proj.v_proj,k_proj.v_proj,v_proj.v_proj,o_proj.v_proj,gate_proj.v_proj,up_proj.v_proj,down_proj.v_proj --lora_r 700 --num_epochs 12 --learning_rate 1e-4 --batch_size 25

python SVDLLM.py --model_path $SVDLLM_HOME_PATH/reproduce/meta_llama_Llama_3.1_8B_whitening_only_0.8.pt --lora $SVDLLM_HOME_PATH/both_half/checkpoint-4662 --step 4


############## RETRAINING THIS WORKS !!!!!!!!!!! #########################################################
torchrun --nproc_per_node=2 --master_port=29500 utils/noLoRA_fsdp.py \
    --prune_model "$SVDLLM_HOME_PATH/comp_fw/meta_llama_Llama_3.1_8B_whitening_only_0.8.pt" \
    --dataset_type "fineweb-edu" \
    --data_path "/storage_nvme_4/llm-random/datasets/fineweb/train" \
    --extra_val_dataset "/storage_nvme_4/llm-random/datasets/fineweb/train" \
    --total_batch_size 16 \
    --micro_batch_size 8 \
    --max_steps 101 \
    --eval_steps 20 \
    --cutoff_len 2048 \
    --learning_rate 5e-5 \
    --output_dir "$SVDLLM_HOME_PATH/ret/test/t64_8" \
    --wandb_project "" \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer" \
    --gradient_checkpointing









DUMP ####################################################################################################################################################################################################################################################################################################################







RETRAINING

torchrun --nproc_per_node=2 --master_port=29500 utils/noLoRA_2fsdp.py     --prune_model "$SVDLLM_HOME_PATH/256/meta_llama_Llama_3.1_8B_whitening_only_0.5.pt"     --dataset_type "fineweb-edu"     --data_path "/storage_nvme_4/llm-random/datasets/fineweb/train"     --extra_val_dataset "/storage_nvme_4/llm-random/datasets/fineweb/train"     --batch_size 8     --grad_acc_steps 4     --max_steps 101     --cutoff_len 2048     --learning_rate 5e-5     --output_dir "$SVDLLM_HOME_PATH/retraining"     --wandb_project ""     --fsdp "full_shard auto_wrap"     --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer"



    # --prune_model "$SVDLLM_HOME_PATH/meta_llama_Llama_3.1_8B_whitening_only_0.8.pt" \
python utils/noLoRA.py \
    --prune_model "$SVDLLM_HOME_PATH/256/meta_llama_Llama_3.1_8B_whitening_only_0.5.pt" \
    --dataset_type "fineweb-edu" \
    --data_path "/storage_nvme_4/llm-random/datasets/fineweb/train" \
    --extra_val_dataset "/storage_nvme_4/llm-random/datasets/fineweb/train" \
    --batch_size 512 \
    --grad_acc_steps 512 \
    --max_steps 101 \
    --cutoff_len 2048 \
    --learning_rate 5e-5 \
    --output_dir "$SVDLLM_HOME_PATH/retraining" \
    --wandb_project ""


# Using torchrun for 2 GPUs
torchrun --nproc_per_node=2 --master_port=29500 utils/noLoRA_fsdp_base.py \
    --prune_model "$SVDLLM_HOME_PATH/256/meta_llama_Llama_3.1_8B_whitening_only_0.5.pt" \
    --dataset_type "fineweb-edu" \
    --data_path "/storage_nvme_4/llm-random/datasets/fineweb/train" \
    --extra_val_dataset "/storage_nvme_4/llm-random/datasets/fineweb/train" \
    --batch_size 512 \
    --grad_acc_steps 256 \
    --max_steps 101 \
    --cutoff_len 2048 \
    --learning_rate 5e-5 \
    --output_dir "$SVDLLM_HOME_PATH/retraining" \
    --wandb_project "" \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap "SVD_Linear"


torchrun --nproc_per_node=4 --master_port=29500 utils/noLoRA_fsdp.py \
    --prune_model "$SVDLLM_HOME_PATH/comp_fw/meta_llama_Llama_3.1_8B_whitening_only_0.8.pt" \
    --dataset_type "fineweb-edu" \
    --data_path "/storage_nvme_4/llm-random/datasets/fineweb/train" \
    --extra_val_dataset "/storage_nvme_4/llm-random/datasets/fineweb/train" \
    --batch_size 512 \
    --grad_acc_steps 32 \
    --max_steps 101 \
    --cutoff_len 2048 \
    --learning_rate 5e-5 \
    --output_dir "$SVDLLM_HOME_PATH/retraining" \
    --wandb_project "" \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap "SVD_Linear"

torchrun --nproc_per_node=2 --master_port=29500 utils/noLoRA_fsdp.py \
    --prune_model "$SVDLLM_HOME_PATH/256/meta_llama_Llama_3.1_8B_whitening_only_0.5.pt" \
    --dataset_type "fineweb-edu" \
    --data_path "/storage_nvme_4/llm-random/datasets/fineweb/train" \
    --extra_val_dataset "/storage_nvme_4/llm-random/datasets/fineweb/train" \
    --batch_size 512 \
    --grad_acc_steps 256 \
    --max_steps 101 \
    --cutoff_len 2048 \
    --learning_rate 5e-5 \
    --output_dir "$SVDLLM_HOME_PATH/retraining" \
    --wandb_project "" \
    --fsdp "full_shard auto_wrap"

torchrun --nproc_per_node=2 --master_port=29500 utils/noLoRA_fsdp.py \
    --prune_model "$SVDLLM_HOME_PATH/256/meta_llama_Llama_3.1_8B_whitening_only_0.5.pt" \
    --dataset_type "fineweb-edu" \
    --data_path "/storage_nvme_4/llm-random/datasets/fineweb/train" \
    --extra_val_dataset "/storage_nvme_4/llm-random/datasets/fineweb/train" \
    --batch_size 512 \
    --grad_acc_steps 256 \
    --max_steps 101 \
    --cutoff_len 2048 \
    --learning_rate 5e-5 \
    --output_dir "$SVDLLM_HOME_PATH/retraining" \
    --wandb_project "" \
    --fsdp "full_shard auto_wrap"

torchrun --nproc_per_node=2 --master_port=29500 utils/noLoRA_2fsdp.py \
torchrun --nproc_per_node=2 --master_port=29501 utils/noLoRA_3fsdp.py \
    --prune_model "$SVDLLM_HOME_PATH/256/meta_llama_Llama_3.1_8B_whitening_only_0.5.pt" \
    --dataset_type "fineweb-edu" \
    --data_path "/storage_nvme_4/llm-random/datasets/fineweb/train" \
    --extra_val_dataset "/storage_nvme_4/llm-random/datasets/fineweb/train" \
    --batch_size 8 \
    --grad_acc_steps 4 \
    --max_steps 101 \
    --cutoff_len 2048 \
    --learning_rate 5e-5 \
    --output_dir "$SVDLLM_HOME_PATH/retraining" \
    --wandb_project "" \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer"
    --fsdp_transformer_layer_cls_to_wrap "Linear"
