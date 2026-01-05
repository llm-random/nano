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

# ... (Previous SVD_Linear and SafeJSONEncoder classes remain identical) ...
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

class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (torch.dtype, type)):
            return str(obj)
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return super().default(obj)

# ... (find_svd_layers_and_ranks and replace_layers_with_svd remain identical) ...
def find_svd_layers_and_ranks(checkpoint_path):
    index_file = os.path.join(checkpoint_path, "model.safetensors.index.json")
    if not os.path.exists(index_file):
        index_file = os.path.join(checkpoint_path, "pytorch_model.bin.index.json")
        if not os.path.exists(index_file):
             raise FileNotFoundError(f"Index file not found at {checkpoint_path}")

    with open(index_file, "r") as f:
        index_data = json.load(f)
    
    weight_map = index_data["weight_map"]
    svd_layer_configs = {} 

    for key, filename in weight_map.items():
        if key.endswith("u_proj.weight"):
            module_path = key.rsplit(".u_proj.weight", 1)[0]
            full_file_path = os.path.join(checkpoint_path, filename)
            
            if filename.endswith(".safetensors"):
                with safe_open(full_file_path, framework="pt", device="cpu") as f:
                    tensor_slice = f.get_slice(key)
                    shape = tensor_slice.get_shape() 
                    rank = shape[1] 
            else:
                state_dict = torch.load(full_file_path, map_location="cpu")
                shape = state_dict[key].shape
                rank = shape[1]

            svd_layer_configs[module_path] = rank
    return svd_layer_configs

def replace_layers_with_svd(model, svd_configs):
    replaced_count = 0
    for name, module in model.named_modules():
        if name in svd_configs:
            rank = svd_configs[name]
            if not isinstance(module, nn.Linear): continue
                
            dim_in = module.in_features
            dim_out = module.out_features
            bias = module.bias is not None
            
            new_layer = SVD_Linear(dim_in, dim_out, rank, bias=bias)
            
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, new_layer)
            else:
                setattr(model, name, new_layer)
            
            replaced_count += 1
    return model

def main():
    parser = argparse.ArgumentParser(description="Evaluate SVD-compressed LLM checkpoints")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--tasks", type=str, default="hellaswag,piqa,arc_easy", help="Comma-separated tasks")
    parser.add_argument("--batch_size", type=int, default=8, help="Eval batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--output_file", type=str, default="eval_results.json", help="Output JSON file")
    
    # [NEW ARGUMENT] Control few-shot shots here
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples")

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint path {args.checkpoint_path} does not exist.")
        sys.exit(1)

    print(f"--- Loading SVD Model from {args.checkpoint_path} ---")
    
    config = AutoConfig.from_pretrained(args.checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    model = model.to_empty(device="cpu") 
    
    svd_configs = find_svd_layers_and_ranks(args.checkpoint_path)
    model = replace_layers_with_svd(model, svd_configs)
    
    load_checkpoint_in_model(model, args.checkpoint_path, dtype=torch_dtype)
    model.to(args.device)
    model.eval()

    task_list = args.tasks.split(",")
    print(f"--- Running Evaluation on: {task_list} (Fewshot: {args.num_fewshot}) ---")
    
    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
    
    results = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=task_list,
        # [FIX] Pass the argument to the harness
        num_fewshot=args.num_fewshot, 
    )
    
    print("\n--- Results (Summary) ---")
    print(json.dumps(results["results"], indent=2, cls=SafeJSONEncoder))

    if "samples" in results: del results["samples"]
    if "requests" in results: del results["requests"]

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2, cls=SafeJSONEncoder)
    print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main()