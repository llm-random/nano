import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_utils import load_sharded_checkpoint
import lm_eval
from lm_eval.models.huggingface import HFLM

# [Required] Class definition needed for torch.load to recognize the object
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

def main():
    parser = argparse.ArgumentParser(description="Evaluate SVD-compressed LLM checkpoints")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint (trained weights)")
    
    # [NEW] The Skeleton Key: Path to the original .pt file
    parser.add_argument("--tokenizer_pt", type=str, required=True, help="Path to original .pt file (provides SVD skeleton & tokenizer)")
    
    # [NEW] Fallback base model (e.g. meta-llama/Llama-3.1-8B)
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B", help="Base model for tokenizer fallback")
    
    parser.add_argument("--tasks", type=str, default="hellaswag,piqa,arc_easy", help="Comma-separated tasks")
    parser.add_argument("--batch_size", type=int, default=8, help="Eval batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--output_file", type=str, default="eval_results.json", help="Output JSON file")
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples")

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint path {args.checkpoint_path} does not exist.")
        sys.exit(1)

    print(f"--- 1. Loading SVD Skeleton from {args.tokenizer_pt} ---")
    try:
        # Load the .pt file (contains 'model' and 'tokenizer')
        loaded_dict = torch.load(args.tokenizer_pt, map_location="cpu", weights_only=False)
        
        # Extract Model Skeleton
        if 'model' in loaded_dict:
            model = loaded_dict['model']
            print("✓ Found SVD Model structure in .pt file.")
        else:
            print("Error: 'model' key not found in .pt file.")
            sys.exit(1)
            
        # Extract Tokenizer
        if 'tokenizer' in loaded_dict:
            tokenizer = loaded_dict['tokenizer']
            print("✓ Found Tokenizer in .pt file.")
        else:
            print(f"Warning: 'tokenizer' not in .pt. Loading from base model: {args.base_model}")
            tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False, trust_remote_code=True)
            
    except Exception as e:
        print(f"CRITICAL ERROR loading .pt file: {e}")
        sys.exit(1)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"--- 2. Injecting Trained Weights from {args.checkpoint_path} ---")
    try:
        # Surgically replace weights using transformers utility
        # This handles safetensors shards and index json automatically
        load_sharded_checkpoint(model, args.checkpoint_path, strict=False)
        print("✓ Successfully merged checkpoint weights into SVD model.")
    except Exception as e:
        print(f"Error merging weights: {e}")
        print("Attempting legacy bin load...")
        try:
            state_dict = torch.load(os.path.join(args.checkpoint_path, "pytorch_model.bin"), map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print("✓ Loaded via pytorch_model.bin")
        except Exception as e2:
            print(f"Failed legacy load: {e2}")
            sys.exit(1)

    # Prepare model for Eval
    model = model.float() # Ensure fp32 for stability
    model.to(args.device)
    model.eval()

    task_list = args.tasks.split(",")
    print(f"--- 3. Running Evaluation on: {task_list} (Fewshot: {args.num_fewshot}) ---")
    
    # Wrap in HFLM for lm_eval harness
    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
    
    results = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        device=args.device
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