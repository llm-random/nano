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



python utils/LoRA.py --prune_model $SVDLLM_HOME_PATH/meta_llama_Llama_3.2_1B_whitening_only_0.8.pt --data_path yahma/alpaca-cleaned --output_dir $SVDLLM_HOME_PATH/first_half --lora_target_modules v_proj,u_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64
python SVDLLM.py --model_path $SVDLLM_HOME_PATH/meta_llama_Llama_3.2_1B_whitening_only_0.8.pt --lora $FINE_TUNE_PATH/first_half /first_half --step 4






