import os
import sys
import argparse
import logging
import json
import torch
import transformers
from transformers import Trainer, TrainingArguments, TrainerCallback
from custom_datasets import FineWebEduDataset, C4Dataset 
from collections import defaultdict

# --- New Callback for Real-Time Logging ---
class FileLoggerCallback(TrainerCallback):
    """
    A custom callback to save logs (training loss, eval loss) to a file
    in real-time, exactly as they are printed to stdout.
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.log_path = os.path.join(output_dir, "loss_history.jsonl")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Only write logs from the main process to avoid concurrency issues
        if state.is_world_process_zero and logs is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(self.log_path, "a") as f:
                # Create a clean log entry with the current step
                log_entry = logs.copy()
                log_entry['step'] = state.global_step
                f.write(json.dumps(log_entry) + "\n")
# ------------------------------------------

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

logger = logging.getLogger(__name__)

# [VERIFIED] Fast Collator for Packed Data
# Since your custom dataset packer guarantees fixed lengths, we don't need padding.
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
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    # Dynamic Batch Size Calculation
    total_batch_size = args.total_batch_size
    micro_batch_size = args.micro_batch_size
    
    global_micro_batch_size = micro_batch_size * world_size
    
    if total_batch_size < global_micro_batch_size:
        grad_acc_steps = 1
        if rank == 0:
            print(f"WARNING: Requested Total Batch ({total_batch_size}) < Physical ({global_micro_batch_size}). Forcing GradAcc=1.")
    else:
        grad_acc_steps = total_batch_size // global_micro_batch_size

    if rank == 0:
        print(f"\n{'='*30} Config {'='*30}")
        print(f"Total Batch Size : {total_batch_size}")
        print(f"Micro Batch      : {micro_batch_size}")
        print(f"World Size       : {world_size}")
        print(f"Grad Acc Steps   : {grad_acc_steps}")
        print(f"Gradient Ckpt    : {args.gradient_checkpointing}")
        print(f"{'='*68}\n")
        
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

    # Gradient Checkpointing Support
    model.config.use_cache = False 
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_param = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params} || All params: {all_param} || %: {100 * trainable_params / all_param:.4f}")

    if rank == 0:
        report_model_structure(model, args.fsdp_transformer_layer_cls_to_wrap)

    # 3. Tokenizer Wrapper
    # [VERIFIED] No Truncation allowed here (Packer needs full stream)
    def smart_tokenizer_wrapper(examples):
        text_column = "text" if "text" in examples else list(examples.keys())[0]
        texts = [t + tokenizer.eos_token for t in examples[text_column]]
        batch_encodings = tokenizer(
            texts,
            truncation=False,       # KEEP FALSE for packing
            max_length=int(1e10),   # KEEP HUGE for packing
            add_special_tokens=True 
        )
        return batch_encodings

    # 4. Datasets
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
        print(f"Initializing evaluation dataset from: {args.extra_val_dataset}")
        
        # [VERIFIED] Limit is passed correctly to custom_datasets.py
        eval_dataset = DatasetClass(
            path=args.extra_val_dataset,
            split="validation", 
            world_size_independent=True, 
            limit=args.n_eval_samples, 
            **common_ds_args
        )
        print(f"Eval dataset limited to {args.n_eval_samples} samples.")

    # 5. Trainer
    fsdp_config = {
        # "min_num_params": 2000_000,
        "xla": False,
        "xla_fsdp_grad_ckpt": False,
        "offload_params": False,
        "pre_forward_reshard": False,
        "transformer_layer_cls_to_wrap": [args.fsdp_transformer_layer_cls_to_wrap],
        "sync_module_states": True,
        "use_orig_params": True, 
        "cpu_ram_efficient_loading": False, 
    }

    training_args = TrainingArguments(
        per_device_train_batch_size=micro_batch_size, 
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=grad_acc_steps,

        lr_scheduler_type="cosine",          # Changed to Cosine
        weight_decay=args.weight_decay,      # Added Weight Decay (0.1)
        max_grad_norm=args.max_grad_norm,    # Added Gradient Clipping (1.0)

        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=True, 
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps" if eval_dataset else "no",
        save_strategy="no",
        load_best_model_at_end=False,
        eval_steps=args.eval_steps,
        output_dir=args.output_dir,
        save_total_limit=2,
        ddp_find_unused_parameters=False,
        report_to="wandb" if args.wandb_project else "none",
        run_name=args.wandb_project if args.wandb_project else None,
        group_by_length=False,
        fsdp=args.fsdp, 
        fsdp_config=fsdp_config,
        
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,

        dataloader_num_workers=args.num_workers,
    )

    # Initialize Callback
    logging_callback = FileLoggerCallback(args.output_dir)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=svd_collator,
        callbacks=[logging_callback] # Register the callback
    )
    
    # 6. Training
    print("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 7. Final Eval & Save
    metrics = train_result.metrics
    
    if eval_dataset:
        print("Running final evaluation...")
        eval_metrics = trainer.evaluate()
        metrics.update(eval_metrics)

    print(f"Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    if rank == 0:
        print(f"\n{'='*30} Final Metrics {'='*30}")
        print(json.dumps(metrics, indent=4))
        print(f"{'='*75}\n")
        
        # Save Final Metrics
        metrics_file = os.path.join(args.output_dir, "train_results.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {metrics_file}")

        # --- NEW: Save Full Loss History (Train + Eval) ---
        history_file = os.path.join(args.output_dir, "full_log_history.json")
        with open(history_file, "w") as f:
            # trainer.state.log_history contains the list of all logs (train and eval)
            json.dump(trainer.state.log_history, f, indent=4)
        print(f"Full loss history saved to {history_file}")
        # --------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')

    parser.add_argument('--prune_model', type=str, required=True)
    parser.add_argument('--dataset_type', type=str, default="fineweb-edu", choices=["c4", "fineweb-edu"])
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--extra_val_dataset', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default="./svd-finetune")
 
    parser.add_argument('--total_batch_size', type=int, default=128, help='Total Global Batch Size')
    parser.add_argument('--micro_batch_size', type=int, default=4, help='Per GPU Batch Size')
    
    parser.add_argument('--max_steps', type=int, default=10000)
    # --- New Arguments ---
    parser.add_argument('--weight_decay', type=float, default=0.1, help="Weight decay for AdamW")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Gradient clipping threshold")
    # ---------------------
    parser.add_argument('--warmup_steps', type=int, default=40)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--cutoff_len', type=int, default=2048)
    
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--n_eval_samples', type=int, default=10, help='Number of validation samples to use')
    
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--resume_from_checkpoint', type=str)

    parser.add_argument('--fsdp', type=str, default="", help="FSDP strategy")
    parser.add_argument('--fsdp_transformer_layer_cls_to_wrap', type=str, default=None)
    
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable Gradient Checkpointing')

    parser.add_argument('--num_workers', type=int, default=4, help='Number of subprocesses for data loading')

    args = parser.parse_args()

    main(args)