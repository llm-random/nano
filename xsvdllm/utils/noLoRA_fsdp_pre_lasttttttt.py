import os
import sys
import argparse
import logging
import torch
import transformers
from transformers import Trainer, TrainingArguments
from custom_datasets import FineWebEduDataset, C4Dataset 
from collections import defaultdict

# [Helper] Debug function to prove device placement
def check_device_placement(model):
    print(f"\n{'='*30} Device Placement Check {'='*30}")
    print("Checking where the 'Unwrapped' modules actually live...")
    
    # Check Embeddings
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        emb_device = model.model.embed_tokens.weight.device
        print(f" -> model.embed_tokens: {emb_device}")
    
    # Check Head
    if hasattr(model, "lm_head"):
        head_device = model.lm_head.weight.device
        print(f" -> lm_head:            {head_device}")
        
    print(f"{'='*82}\n")

def report_model_structure(model, wrap_policy_name):
    print(f"\n{'='*30} Model Architecture & FSDP Report {'='*30}")
    wrapped_scope_prefixes = []
    if wrap_policy_name:
        for name, module in model.named_modules():
            if module.__class__.__name__ == wrap_policy_name:
                wrapped_scope_prefixes.append(name)
    
    stats = defaultdict(lambda: {"total": 0, "wrapped": 0, "unwrapped": 0})
    for name, module in model.named_modules():
        if name == "": continue
        cls_name = module.__class__.__name__
        stats[cls_name]["total"] += 1
        is_wrapped = False
        if cls_name == wrap_policy_name:
            is_wrapped = True
        else:
            for prefix in wrapped_scope_prefixes:
                if name.startswith(prefix + "."):
                    is_wrapped = True
                    break
        if is_wrapped:
            stats[cls_name]["wrapped"] += 1
        else:
            stats[cls_name]["unwrapped"] += 1

    print(f"{'Module Type':<30} | {'Total':<8} | {'Wrapped':<8} | {'Unwrapped':<10}")
    print("-" * 65)
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True)
    for cls_name, counts in sorted_stats:
        print(f"{cls_name:<30} | {counts['total']:<8} | {counts['wrapped']:<8} | {counts['unwrapped']:<10}")
    print("-" * 65)
    print(f"{'='*82}\n")

class SVD_Linear(torch.nn.Module):
    def __init__(self, dim_in, dim_out, low_rank_dim, bias=False):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.low_rank_dim = low_rank_dim
        self.v_proj = torch.nn.Linear(self.dim_in, self.low_rank_dim, bias=False)
        self.u_proj = torch.nn.Linear(self.low_rank_dim, self.dim_out, bias=bias)

    def forward(self, x):
        return self.u_proj(self.v_proj(x))

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger(__name__)

def svd_collator(features):
    if not isinstance(features[0], (list, tuple)):
        return transformers.default_data_collator(features)
    batch_ids = torch.tensor(features, dtype=torch.long)
    return {
        "input_ids": batch_ids,
        "labels": batch_ids.clone(),
        "attention_mask": torch.ones_like(batch_ids)
    }

def main(args):
    # 0. Setup Distributed Environment & Batch Size Logic
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    
    # Detect world size (number of GPUs)
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # [CRITICAL UPDATE] Dynamic Batch Size Calculation
    # Formula: Total = Micro_Per_GPU * World_Size * Grad_Acc
    # We solve for Grad_Acc
    total_batch_size = args.total_batch_size
    micro_batch_size = args.micro_batch_size
    
    global_micro_batch_size = micro_batch_size * world_size
    
    if total_batch_size < global_micro_batch_size:
        # Fallback if the user asks for a total smaller than the physical minimum
        grad_acc_steps = 1
        if int(os.environ.get("RANK", 0)) == 0:
            print(f"WARNING: Requested Total Batch Size ({total_batch_size}) is smaller than (Micro * WorldSize) ({global_micro_batch_size}). forcing GradAcc=1.")
    else:
        grad_acc_steps = total_batch_size // global_micro_batch_size

    # Logging on main process only
    if int(os.environ.get("RANK", 0)) == 0:
        print(f"\n{'='*30} Batch Size Configuration {'='*30}")
        print(f"Target Total Batch Size : {total_batch_size}")
        print(f"World Size (GPUs)       : {world_size}")
        print(f"Micro Batch (per GPU)   : {micro_batch_size}")
        print(f"Calculated Grad Acc     : {grad_acc_steps}")
        print(f"Effective Check         : {micro_batch_size} * {world_size} * {grad_acc_steps} = {micro_batch_size * world_size * grad_acc_steps}")
        print(f"{'='*82}\n")
        
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    # 1. Load Model
    print(f"Loading model from {args.prune_model}...")
    pruned_dict = torch.load(args.prune_model, map_location='cpu', weights_only=False)
    tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
    
    if tokenizer.pad_token is None or tokenizer.pad_token_id == 0:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # 2. Make All Model Trainable
    model = model.to("cpu")
    model = model.float() 
    
    print("Setting all parameters to trainable (requires_grad=True)...")
    model.requires_grad_(True) 

    # ... after loading model ...
    model.config.use_cache = False # Required for gradient checkpointing
    
    # [ADD THIS BLOCK before creating Trainer]
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_param = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params} || All params: {all_param} || %: {100 * trainable_params / all_param:.4f}")

    # 3. Report Model Structure
    report_model_structure(model, args.fsdp_transformer_layer_cls_to_wrap)

    # 4. Tokenizer Wrapper
    def smart_tokenizer_wrapper(examples):
        text_column = "text" if "text" in examples else list(examples.keys())[0]
        texts = [t + tokenizer.eos_token for t in examples[text_column]]
        batch_encodings = tokenizer(
            texts,
            truncation=False,
            max_length=int(1e10),
            add_special_tokens=True 
        )
        return batch_encodings

    # 5. Datasets
    print(f"Initializing custom {args.dataset_type} dataset...")
    common_ds_args = {
        "sequence_length": args.cutoff_len,
        "tokenize_fn": smart_tokenizer_wrapper,
        "seed": 42,
        "use_new_sampling_method": True,
        "shuffle": True,
    }

    if args.dataset_type == "fineweb-edu":
        DatasetClass = FineWebEduDataset
    elif args.dataset_type == "c4":
        DatasetClass = C4Dataset
    else:
        raise ValueError("dataset_type error")

    train_dataset = DatasetClass(
        path=args.data_path,
        split="train",
        world_size_independent=False,
        **common_ds_args
    )

    eval_dataset = None
    if args.extra_val_dataset:
        print(f"Initializing streaming evaluation dataset from: {args.extra_val_dataset}")
        
        # [FIX] Do NOT use Subset. Just pass the 'limit' argument directly.
        eval_dataset = DatasetClass(
            path=args.extra_val_dataset,
            split="validation", 
            world_size_independent=True, 
            limit=args.n_eval_samples,  # <--- PASS LIMIT HERE
            **common_ds_args
        )

    # 6. Trainer
    fsdp_config = {
        "min_num_params": 0,
        "xla": False,
        "xla_fsdp_grad_ckpt": False,
        "offload_params": False,
        "pre_forward_reshard": False,
        "transformer_layer_cls_to_wrap": [args.fsdp_transformer_layer_cls_to_wrap],
        "sync_module_states": True,
        # "use_orig_params": True,
        "use_orig_params": False,
        "cpu_ram_efficient_loading": False, 
    }

    training_args = TrainingArguments(
        # [CRITICAL FIX] Use calculated values
        per_device_train_batch_size=micro_batch_size, 
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=grad_acc_steps,
        
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=True, 
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps" if eval_dataset else "no",
        save_strategy="no",
        # save_steps=args.save_steps,
        load_best_model_at_end=False,
        eval_steps=args.eval_steps,
        output_dir=args.output_dir,
        save_total_limit=2,
        # load_best_model_at_end=True if eval_dataset else False,
        ddp_find_unused_parameters=False,
        report_to="wandb" if args.wandb_project else "none",
        run_name=args.wandb_project if args.wandb_project else None,
        group_by_length=False,
        fsdp=args.fsdp, 
        fsdp_config=fsdp_config,
        gradient_checkpointing=True, 
        gradient_checkpointing_kwargs={"use_reentrant": False}, # Modern setting for Llama
        
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=svd_collator, 
    )
    
    if hasattr(model, "config"):
        model.config.use_cache = False

    # 7. Pre-flight Check
    print("Starting training...")
    train_result =trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    metrics = train_result.metrics
    if eval_dataset:
        print("Running final evaluation...")
        eval_metrics = trainer.evaluate()
        metrics.update(eval_metrics)

    print(f"Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')

    parser.add_argument('--prune_model', type=str, required=True)
    parser.add_argument('--dataset_type', type=str, default="fineweb-edu", choices=["c4", "fineweb-edu"])
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--extra_val_dataset', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default="./svd-finetune")
 
    # [CRITICAL FIX] Renamed batch_size to Total, and added micro_batch_size
    parser.add_argument('--total_batch_size', type=int, default=128, help='Total Global Batch Size (Constant regardless of GPU count)')
    parser.add_argument('--micro_batch_size', type=int, default=4, help='Physical Batch Size per GPU (VRAM limit)')
    
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--warmup_steps', type=int, default=40)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--cutoff_len', type=int, default=2048)
    
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--n_eval_samples', type=int, default=10)
    # parser.add_argument('--save_steps', type=int, default=1000)
    
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--resume_from_checkpoint', type=str)

    # FSDP CLI Arguments
    parser.add_argument('--fsdp', type=str, default="", help="FSDP strategy, e.g., 'full_shard auto_wrap'")
    parser.add_argument('--fsdp_transformer_layer_cls_to_wrap', type=str, default=None, 
                        help="Transformer layer class name to wrap, e.g., 'LlamaDecoderLayer'")
    
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable Gradient Checkpointing to save memory')

    args = parser.parse_args()

    main(args)