import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import load_checkpoint_in_model
import lm_eval
from lm_eval.models.huggingface import HFLM

# 1. Define the Custom SVD Layer
class SVD_Linear(nn.Module):
    def __init__(self, dim_in, dim_out, low_rank_dim, bias=False):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.low_rank_dim = low_rank_dim
        self.v_proj = nn.Linear(self.dim_in, self.low_rank_dim, bias=False)
        self.u_proj = nn.Linear(self.low_rank_dim, self.dim_out, bias=bias)

    def forward(self, x):
        return self.u_proj(self.v_proj(x))

def find_svd_layers_and_ranks(checkpoint_path):
    """
    Scans the safetensors index to identify which layers are SVD
    and infers their rank (low_rank_dim) from weight shapes.
    """
    index_file = os.path.join(checkpoint_path, "model.safetensors.index.json")
    if not os.path.exists(index_file):
        # Fallback for bin files if safetensors index doesn't exist
        index_file = os.path.join(checkpoint_path, "pytorch_model.bin.index.json")
        if not os.path.exists(index_file):
             raise FileNotFoundError(f"Index file not found at {checkpoint_path}")

    with open(index_file, "r") as f:
        index_data = json.load(f)
    
    weight_map = index_data["weight_map"]
    svd_layer_configs = {} # Key: module_path, Value: rank

    # We look for keys ending in 'u_proj.weight' to identify SVD layers
    for key, filename in weight_map.items():
        if key.endswith("u_proj.weight"):
            # module path is everything before ".u_proj.weight"
            module_path = key.rsplit(".u_proj.weight", 1)[0]
            
            # We need to read the file to get the shape (Rank)
            full_file_path = os.path.join(checkpoint_path, filename)
            
            # Helper to open safe or bin
            if filename.endswith(".safetensors"):
                with safe_open(full_file_path, framework="pt", device="cpu") as f:
                    tensor_slice = f.get_slice(key)
                    shape = tensor_slice.get_shape() 
                    rank = shape[1] # [out, rank]
            else:
                # Slower fallback for bin files
                state_dict = torch.load(full_file_path, map_location="cpu")
                shape = state_dict[key].shape
                rank = shape[1]

            svd_layer_configs[module_path] = rank
            
    print(f"Found {len(svd_layer_configs)} SVD layers in checkpoint.")
    return svd_layer_configs

def replace_layers_with_svd(model, svd_configs):
    """
    Recursively replaces target Linear layers with SVD_Linear
    """
    replaced_count = 0
    for name, module in model.named_modules():
        if name in svd_configs:
            rank = svd_configs[name]
            
            if not isinstance(module, nn.Linear):
                print(f"Warning: Expected Linear at {name}, found {type(module)}")
                continue
                
            dim_in = module.in_features
            dim_out = module.out_features
            bias = module.bias is not None
            
            # Create replacement
            new_layer = SVD_Linear(dim_in, dim_out, rank, bias=bias)
            
            # Replace in parent
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, new_layer)
            else:
                # Root level replacement (unlikely for SVD but possible)
                setattr(model, name, new_layer)
            
            replaced_count += 1
            
    print(f"Successfully replaced {replaced_count} layers with SVD architecture.")
    return model

def main():
    parser = argparse.ArgumentParser(description="Evaluate SVD-compressed LLM checkpoints")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the saved checkpoint directory")
    parser.add_argument("--tasks", type=str, default="hellaswag,piqa,arc_easy", help="Comma-separated list of tasks")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda/cpu)")
    parser.add_argument("--output_file", type=str, default="eval_results.json", help="File to save results")
    
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint path {args.checkpoint_path} does not exist.")
        sys.exit(1)

    print(f"--- Loading SVD Model from {args.checkpoint_path} ---")
    
    # 1. Load Config & Tokenizer
    config = AutoConfig.from_pretrained(args.checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    
    # 2. Instantiate Empty Standard Model (Skeleton)
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    print("Creating skeleton model...")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
        
    model = model.to_empty(device="cpu") # Materialize on CPU first to swap layers
    
    # 3. Detect SVD Layers & Ranks
    print("Scanning checkpoint structure...")
    svd_configs = find_svd_layers_and_ranks(args.checkpoint_path)
    
    # 4. Perform Surgery (Swap Linear -> SVD_Linear)
    print("Performing architecture surgery...")
    model = replace_layers_with_svd(model, svd_configs)
    
    # 5. Load Weights
    print("Loading sharded weights...")
    
    # [FIX] Removed device_map="auto" to prevent AttributeError.
    # We load to CPU (default) then move to GPU manually.
    load_checkpoint_in_model(
        model, 
        args.checkpoint_path, 
        dtype=torch_dtype, 
        # device_map="auto"  <-- REMOVED THIS
    )
    
    # Manually move to GPU
    print(f"Moving model to {args.device}...")
    model.to(args.device)
    model.eval()
    
    print(f"Model loaded. Footprint: {model.get_memory_footprint() / 1e9:.2f} GB")

    # 6. Run Evaluation
    task_list = args.tasks.split(",")
    print(f"--- Running Evaluation on: {task_list} ---")
    
    # Wrap in LM-Eval Harness
    # We pass the model object directly. 
    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
    
    results = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=task_list,
        num_fewshot=0,
    )
    
    print("\n--- Results ---")
    # Clean output for printing
    print(json.dumps(results["results"], indent=2))

    # Save results
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main()