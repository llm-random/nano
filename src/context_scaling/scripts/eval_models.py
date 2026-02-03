import os
from pathlib import Path
import argparse
from tqdm.auto import tqdm
import pandas as pd
import json
from functools import partial

from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
import resolver as _

from datasets import load_from_disk
from transformers import AutoTokenizer

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed.checkpoint as dcp
from torch.utils.data import DataLoader
from torch.distributed.checkpoint.stateful import Stateful


def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def make_csv_name(template: str, cfg) -> str:
    flat_cfg = flatten_dict(OmegaConf.to_container(cfg, resolve=True))

    parts = []
    for kw in template.split(","):
        kw = kw.strip().replace("/", ".")
        for k, v in flat_cfg.items():
            if kw in k:
                kw_leaf = kw.split(".")[-1]
                parts.append(f"{kw_leaf}={v}")
                break

    return "+".join(parts) + ".csv"


def load_hydra_cfg_from_yaml(yaml_path: Path):
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Hydra yaml_config not found: {yaml_path}")

    with initialize_config_dir(config_dir=str(yaml_path.parent.resolve()), version_base=None):
        cfg = compose(config_name=yaml_path.stem)
    return cfg


def setup_distributed():
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
    print(f"✅ torch.distributed initialized (1 GPU) MASTER_PORT={os.environ.get('MASTER_PORT')}")
    return device


def collate_no_pad(batch, tokenizer, seq_len):
    texts = [ex["text"] for ex in batch]
    urls = [ex.get("url", "") for ex in batch]

    enc = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"]
    if input_ids.size(1) != seq_len:
        return None

    return {"input_ids": input_ids, "url": urls}


def batch_per_token_losses(model, input_ids, device):
    model.eval()
    input_ids = input_ids.to(device)

    out = model(input_ids)
    logits = out.logits if hasattr(out, "logits") else out

    logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]

    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).reshape(targets.shape)

    return losses.cpu(), targets.cpu()


class ModelOnly(Stateful):
    def __init__(self, model):
        self.model = model

    def state_dict(self):
        return {"model": self.model.state_dict()}

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd["model"], strict=True)


def tensors_rows_to_csv(tensors, path: str):
    rows = [t.detach().cpu() for t in tensors]
    stacked = torch.cat(rows, dim=0)
    print(f"saving CSV to {path}")
    pd.DataFrame(stacked.numpy()).to_csv(path, index=False)
    print(f"✅ CSV saved")


def eval_model(
    ckpt_dir: str,        # directory that contains step_* (or numeric step dirs)
    model_step: int,
    cfg,
    dataset_dir: str,
    out_csv: str,
    seq_len: int,
    batch_size: int,
    device: torch.device,
):
    model = instantiate(cfg.model, _convert_="all").to(device)
    model.eval()

    print(f"Model instantiated on {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    fsdp_model = FSDP(model)
    state = {"app": ModelOnly(fsdp_model)}

    ckpt_path = os.path.join(ckpt_dir, f"step_{model_step}")
    dcp.load(state, checkpoint_id=ckpt_path)
    print(f"✅ Checkpoint loaded: {ckpt_path}")

    fsdp_model.eval()

    ds = load_from_disk(dataset_dir)
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

    collate_fn = partial(collate_no_pad, tokenizer=tokenizer, seq_len=seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_losses = []
    for batch in tqdm(loader):
        if batch is None:
            continue
        with torch.no_grad():
            losses, _ = batch_per_token_losses(fsdp_model, batch["input_ids"], device)
        all_losses.append(losses)

    tensors_rows_to_csv(all_losses, path=out_csv)


def main():
    parser = argparse.ArgumentParser(description="Eval models using cached jobs.json + yaml_cache (no Neptune)")

    parser.add_argument(
        "--jobs_json",
        type=str,
        required=True,
        help="Path to jobs.json produced by the prefetch script",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to dataset loaded with load_from_disk",
    )
    parser.add_argument(
        "--out_csv_format",
        type=str,
        default="kv_heads,dff",
        help="Comma-separated keywords for CSV filename template",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=".",
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--model_step",
        type=int,
        default=320000,
        help="Checkpoint step directory name (numeric).",
    )

    args = parser.parse_args()

    device = setup_distributed()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # jobs_json contains list[{"jobID","ckpt_path","yaml_config_path"}]
    with open(args.jobs_json, "r", encoding="utf-8") as f:
        jobs = json.load(f)

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if task_id < 0 or task_id >= len(jobs):
        raise IndexError(f"SLURM_ARRAY_TASK_ID={task_id} out of range (0..{len(jobs)-1})")

    job = jobs[task_id]
    run_id = job["jobID"]
    ckpt_dir = job["ckpt_path"]
    yaml_path = Path(job["yaml_config_path"])

    print(f"Selected job: idx={task_id} run_id={run_id}")
    print(f"ckpt_dir: {ckpt_dir}")
    print(f"yaml_config_path: {yaml_path}")

    cfg = load_hydra_cfg_from_yaml(yaml_path)
    out_csv = os.path.join(out_dir, make_csv_name(args.out_csv_format, cfg))

    try:
        eval_model(
            ckpt_dir=ckpt_dir,
            model_step=args.model_step,
            cfg=cfg,
            dataset_dir=args.dataset_dir,
            out_csv=out_csv,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            device=device,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
