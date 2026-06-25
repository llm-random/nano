import math

# seq_len = 2048
# batch_size = 32
# n_blocks = 16
# dmodel = 64 * n_blocks
# dff = int(2.66 * dmodel)
# datt = dmodel
# q_heads = n_blocks
# kv_heads = n_blocks
# vocab_size = 50304
# use_swiGLU = True

seq_len = 1024
batch_size = 64
n_blocks = 16
dmodel = 64 * n_blocks
dff = int(2.66 * dmodel)
datt = dmodel
q_heads = n_blocks
kv_heads = n_blocks
vocab_size = 50304
use_swiGLU = True

n_steps = 10001

gpu_flops = 989 * 10**12  # H100 80GB
gpu_mem_bytes = 80 * 10**9  # H100 80GB
mfu = 0.1

bytes_per_param = 2  # fp16

print("--- Model Configuration ---")
print(f"seq_len: {seq_len}")
print(f"batch_size: {batch_size}")
print(f"dmodel: {dmodel}")
print(f"dff: {dff}")
print(f"use_swiGLU: {use_swiGLU}")
print(f"datt: {datt}")
print(f"n_blocks: {n_blocks}")
print(f"q_heads: {q_heads}")
print(f"kv_heads: {kv_heads}")
print(f"vocab_size: {vocab_size}")
print()

p_emb = vocab_size * dmodel
p_attn = 4 * dmodel**2
p_ff = 3 * dmodel * dff if use_swiGLU else 2 * dmodel * dff

total_params = n_blocks * (p_attn + p_ff) + p_emb
model_size_bytes = total_params * bytes_per_param

static_memory_bytes = 6 * total_params * bytes_per_param
activation_memory_bytes = batch_size * seq_len * dmodel * n_blocks * bytes_per_param
total_memory_bytes = static_memory_bytes + activation_memory_bytes

tokens_in_dataset = batch_size * seq_len * n_steps
tokens_param_ratio = tokens_in_dataset / total_params
suggested_tokens = 20 * total_params
suggested_n_steps = math.ceil(suggested_tokens / (batch_size * seq_len))

total_compute_needed = 6 * total_params * tokens_in_dataset
total_compute_needed_chinchilla = 6 * total_params * suggested_tokens

effective_flops = gpu_flops * mfu
seconds_to_train = total_compute_needed / effective_flops
seconds_to_train_chinchilla = total_compute_needed_chinchilla / effective_flops

print("--- Configuration ---")
print(f"Total Parameters: {total_params / 1e6:.2f} Million")
print(f"Tokens / Parameter Ratio (Chinchilla): {tokens_param_ratio:.2f}")
print(f"Suggested Training Steps (Chinchilla): {suggested_n_steps}")
print(f"Model Size (BF16): {model_size_bytes / 1e6:.2f} MB")
print(f"Estimated Training VRAM: {total_memory_bytes / 1e6:.2f} MB")
print(f"GPU Memory Utilization: {(total_memory_bytes / gpu_mem_bytes) * 100:.4f}%")
print(f"--- Training Time (for {tokens_in_dataset/1e6}M tokens) ---")
print(f"Time (using your FLOPS): {seconds_to_train / 3600:.2f} Hours")
print(f"Time to train (Chinchilla optimal): {seconds_to_train_chinchilla / 3600:.2f} Hours")
