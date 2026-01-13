#coding:utf8
import os
import sys
import argparse
import torch.jit
from tqdm import tqdm
import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_utils import load_sharded_checkpoint # Required for merging weights

# [Added] Import your custom datasets
try:
    from utils.custom_datasets import FineWebEduDataset, C4Dataset
except ImportError:
    print("WARNING: Could not import FineWebEduDataset/C4Dataset. Ensure custom_datasets.py is present.")

from component.svd_llama import SVD_Linear, test_svd
from utils.data_utils import *
from utils.model_utils import *
from evaluater import * 

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)
sys.path.append(parent_path)

# [Helper] Collator for SVD
def svd_collator(features):
    if not isinstance(features[0], (list, tuple)):
        return transformers.default_data_collator(features)
    batch_ids = torch.tensor(features, dtype=torch.long)
    return {
        "input_ids": batch_ids,
        "labels": batch_ids.clone(),
        "attention_mask": torch.ones_like(batch_ids)
    }

# ... [Keep profle_svdllm and whitening functions exactly as they were] ...
# (Omitting them here to save space, assuming they are unchanged from previous versions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='jeffwan/llama-7b-hf', help='Base model identifier')
    parser.add_argument('--model_path', type=str, default=None, help='Local checkpoint path (dir or file)')
    
    # [CRITICAL] This file provides the SVD Architecture (Skeleton)
    parser.add_argument('--tokenizer_pt', type=str, default=None, help='Path to the original .pt file containing the SVD model structure')
    
    parser.add_argument('--ratio', type=float, default=0.2, help='Target compression ratio')
    parser.add_argument('--dataset', type=str, default='wikitext2', choices=['wikitext2', 'ptb', 'c4', 'fineweb-edu'])
    parser.add_argument('--data_path', type=str, default=None, help='Path for custom datasets')
    parser.add_argument('--whitening_nsamples', type=int, default=256)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--seed',type=int, default=0)
    parser.add_argument('--DEV', type=str, default="cuda")
    parser.add_argument('--model_seq_len', type=int, default=2048)
    parser.add_argument('--step', type=int, default=4)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--profiling_mat_path', type=str, default=None)
    parser.add_argument('--run_low_resource', action='store_true')
    parser.add_argument('--gen_seq_len', type=int, default=1024)
    parser.add_argument('--lora', type=str, default=None)
    parser.add_argument('--updating_nsamples', type=int, default=16)

    args = parser.parse_args()
    args.ratio = 1 - args.ratio

    if args.step == 1:
        pass # (Omitted)

    elif args.step >= 4:
        print(f"Loading and evaluating...")

        # --- 1. LOAD TOKENIZER ---
        tokenizer = None
        if args.tokenizer_pt:
            print(f"Loading tokenizer from original .pt file: {args.tokenizer_pt}")
            try:
                # We load the whole dict to get the tokenizer
                loaded_dict = torch.load(args.tokenizer_pt, map_location='cpu', weights_only=False)
                if 'tokenizer' in loaded_dict:
                    tokenizer = loaded_dict['tokenizer']
                    print("Successfully loaded tokenizer from .pt file.")
                
                # [CRITICAL FIX] Capture the Model Structure here!
                if 'model' in loaded_dict:
                    print("Found SVD Model structure in .pt file. Using this as skeleton.")
                    model = loaded_dict['model']
                else:
                    model = None
                    print("WARNING: Model structure not found in .pt file.")
                    
            except Exception as e:
                print(f"ERROR loading from .pt: {e}")
                model = None

        # Fallback tokenizer loading
        if tokenizer is None:
            print(f"Fallback: Loading tokenizer from base model: {args.model}")
            tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"Tokenizer Vocab Size: {len(tokenizer)}")

        # --- 2. LOAD MODEL WEIGHTS ---
        if model is None:
            # If we didn't find the model in the .pt, we are in trouble for SVD checkpoints.
            # But we fall back to standard loading just in case.
            print("WARNING: No SVD skeleton found. Attempting standard load (will fail for SVD checkpoints).")
            model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, trust_remote_code=True)
        else:
            # We have the SVD Skeleton (model). Now we inject the trained weights.
            if args.model_path and os.path.isdir(args.model_path):
                print(f"Injecting trained weights from checkpoint folder: {args.model_path}")
                try:
                    # load_sharded_checkpoint handles model.safetensors.index.json automatically
                    load_sharded_checkpoint(model, args.model_path, strict=False)
                    print("Successfully merged checkpoint weights into SVD model.")
                except Exception as e:
                    print(f"Error merging weights: {e}")
                    print("Trying legacy load_state_dict...")
                    # Fallback for non-sharded
                    state_dict = torch.load(os.path.join(args.model_path, "pytorch_model.bin"), map_location="cpu")
                    model.load_state_dict(state_dict, strict=False)

        model.eval()
        model = model.float() 
        model = model.to(args.DEV)

        # --- 3. EVALUATION ---
        if args.dataset in ["fineweb-edu", "c4"]:
            print(f"Evaluating on {args.dataset}...")
            if args.data_path is None: raise ValueError("Provide --data_path")

            def smart_tokenizer_wrapper(examples):
                text_column = "text" if "text" in examples else list(examples.keys())[0]
                texts = [t + tokenizer.eos_token for t in examples[text_column]]
                return tokenizer(texts, truncation=False, max_length=int(1e10))

            dataset_cls = FineWebEduDataset if args.dataset == "fineweb-edu" else C4Dataset
            dataset = dataset_cls(
                path=args.data_path,
                split="validation", 
                world_size_independent=True,
                limit=args.whitening_nsamples,
                sequence_length=args.model_seq_len,
                tokenize_fn=smart_tokenizer_wrapper,
                seed=args.seed,
                use_new_sampling_method=True,
                shuffle=False
            )
            
            loader = torch.utils.data.DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=svd_collator)
            
            nlls, tokens = [], 0
            loss_fct = nn.CrossEntropyLoss(reduction="sum")
            
            with torch.no_grad():
                for batch in tqdm(loader, desc="Eval"):
                    batch = {k: v.to(args.DEV) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    out = model(**batch)
                    
                    shift_logits = out.logits[..., :-1, :].contiguous()
                    shift_labels = batch["labels"][..., 1:].contiguous()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    
                    nlls.append(loss)
                    tokens += shift_labels.ne(loss_fct.ignore_index).sum().item()
            
            if tokens > 0:
                ppl = torch.exp(torch.stack(nlls).sum() / tokens)
                print(f"\nFinal PPL: {ppl.item():.2f}")
                print(f"Final CE: {(torch.stack(nlls).sum() / tokens).item():.2f}")
            else:
                print("No tokens processed.")
        else:
            ppl_eval(model, tokenizer, datasets=[args.dataset], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)