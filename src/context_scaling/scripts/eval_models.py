import os
import subprocess
from pathlib import Path
import argparse
from tqdm.auto import tqdm
import pandas as pd
import neptune
import shutil
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
    """Flatten a nested dict into dot-separated keys."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def make_csv_name(template: str, cfg) -> str:
    """Build CSV filename from template keywords and config values.

    Example: template="kv_heads,dff", cfg with common.kv_heads=4, common.dff=512
    Returns: "kv_heads=4+dff=512.csv"
    """
    flat_cfg = flatten_dict(OmegaConf.to_container(cfg, resolve=True))

    parts = []
    for kw in template.split(","):
        kw = kw.strip()
        kw = kw.replace("/", ".")
        for k, v in flat_cfg.items():
            if kw in k:
                kw = kw.split("/")[-1]
                parts.append(f"{kw}={v}")
                break  # take first match

    return "+".join(parts) + ".csv"


def get_neptune_table(
    tags,
    project="pmtest/llm-random",
    negative_tags=None,
    columns=None,
    print_columns=False,
):
    """
    Fetches a Neptune runs table filtered by tags and returns it as a pandas DataFrame.

    Parameters:
    - tags (list): List of tags to filter the runs.
    - negative_tags (list, optional): List of tags to exclude from the runs.
    - columns (list, optional): Additional columns to include in the runs table.
    - print_columns (bool, optional): If True, prints all available columns.

    Returns:
    - pandas.DataFrame: The runs table with the specified filters and columns.
    """

    # Initialize the Neptune project
    project = neptune.init_project(
        project=project,
        mode="read-only",
        api_token=os.environ["NEPTUNE_API_TOKEN"],
    )

    # Fetch the runs table with the specified tags and columns
    runs_table = project.fetch_runs_table(tag=tags, columns=columns).to_pandas()

    print(f"df shape: {runs_table.shape}")
    # Ensure 'sys/tags' is a list for each run
    # print(f"runs_table: {runs_table}")
    runs_table["sys/tags"] = runs_table["sys/tags"].apply(
        lambda x: x.split(",") if isinstance(x, str) else x
    )

    # Exclude runs containing any of the negative tags
    if negative_tags:
        for neg_tag in negative_tags:
            runs_table = runs_table[
                ~runs_table["sys/tags"].apply(lambda x: neg_tag in x)
            ]

    print(f"Table downloaded\nShape: {runs_table.shape}")

    if print_columns:
        print("\n=== Available columns ===")
        for col in sorted(runs_table.columns):
            print(f"  {col}")
        print("========================\n")

    return runs_table


def get_hydra_config(row):
    run_id = row["sys/id"]
    print(f"run ID: {run_id}")
    run = neptune.init_run(
        project="pmtest/llm-random",
        with_id=run_id,
        mode="read-only",
        api_token=os.environ["NEPTUNE_API_TOKEN"],
    )
    job_dir = f"{os.environ.get('SLURM_ARRAY_JOB_ID','0')}/{os.environ.get('SLURM_ARRAY_TASK_ID','0')}"
    os.makedirs(job_dir, exist_ok=True)

    cfg_file = f"{job_dir}/tmp_hydra_config.yaml"
    run["yaml_config"].download(destination=cfg_file)

    cfg_dir = str(Path(cfg_file).parent.absolute())
    cfg_name = Path(cfg_file).stem  # "tmp_hydra_config" (no .yaml)

    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name=cfg_name)

    return cfg


def setup_distributed():
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=0, world_size=1)

    print(f"✅ torch.distributed initialized (1 GPU) MASTER_PORT={os.environ.get('MASTER_PORT')}")
    return device


# def rsync_checkpoint(
#     cluster: str, remote_path: str, model_step: int, local_dir: str
# ) -> None:
#     "Rsync a checkpoint from a remote cluster to local directory."

#     cmd = [
#         "rsync", "-rlphvP",
#         "-e", "ssh -vv",
#         f"{cluster}:{remote_path}/step_{model_step}/",
#         local_dir,
#     ]
#     subprocess.run(cmd, check=True)


def collate_no_pad(batch, tokenizer, seq_len):
    texts = [ex["text"] for ex in batch]
    urls = [ex["url"] for ex in batch]

    enc = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"]  # [B, <=SEQ_LEN]

    # keep only samples that actually reached SEQ_LEN
    keep = input_ids.size(1) == seq_len
    if not keep:
        return None  # drop this batch

    return {
        "input_ids": input_ids,
        "url": urls,
    }


def batch_per_token_losses(model, input_ids, device):
    model.eval()
    input_ids = input_ids.to(device)  # [B, T]

    out = model(input_ids)
    logits = out.logits if hasattr(out, "logits") else out  # [B, T, V]

    logits = logits[:, :-1, :]  # [B, T-1, V]
    targets = input_ids[:, 1:]  # [B, T-1]

    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).reshape(
        targets.shape
    )  # [B, T-1]

    return losses.cpu(), targets.cpu()


class ModelOnly(Stateful):
    def __init__(self, model):
        self.model = model

    def state_dict(self):
        return {"model": self.model.state_dict()}

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd["model"], strict=True)


def tensors_rows_to_csv(tensors, path="tensors.csv"):
    rows = []
    for t in tensors:
        rows.append(t.detach().cpu())
    stacked = torch.cat(rows, dim=0)  # (num_tensors * N, N)
    pd.DataFrame(stacked.numpy()).to_csv(path, index=False)


def eval_model(
    ckpt_path: str,
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

    dcp.load(state, checkpoint_id=ckpt_path)

    print("✅ Checkpoint loaded")

    fsdp_model.eval()
    print("✅ Model loaded for inference (optimizer skipped)")

    ds = load_from_disk(dataset_dir)

    print(ds)
    print("Columns:", ds.column_names)
    print("Example keys:", ds[0].keys())
    print(
        "Text preview:",
        (ds[0]["text"][:200] + "...") if "text" in ds[0] else "NO 'text' COLUMN",
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

    print("Tokenizer vocab size:", len(tokenizer))

    collate_fn = partial(
        collate_no_pad,
        tokenizer=tokenizer,
        seq_len=seq_len,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    all_losses = []

    for batch in tqdm(loader):
        if batch is None:
            continue

        with torch.no_grad():
            losses, _ = batch_per_token_losses(fsdp_model, batch["input_ids"], device)
        all_losses.append(losses)

    tensors_rows_to_csv(all_losses, path=out_csv)


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-token losses and export to CSV"
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to dataset loaded with load_from_disk",
    )
    parser.add_argument(
        "--neptune_csv_path",
        type=str,
        help="path to the downloaded neptune csv file",
        required=True,
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
        help="Sequence length (e.g. 2048 or 8192)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--model_step",
        type=int,
        default=320000,
        help="for loading correct model checkpoint",
    )

    args = parser.parse_args()

    device = setup_distributed()

    df = pd.read_csv(args.neptune_csv_path)

    row = df.iloc[int(os.environ["SLURM_ARRAY_TASK_ID"])]

    cfg = get_hydra_config(row)

    ckpt_path = row["job/full_save_checkpoints_path"]
    out_csv = str(Path(args.out_dir) / make_csv_name(args.out_csv_format, cfg))
    print(
        f"""
        DEBUG eval_model params:\n
        qckpt_dir={ckpt_path},\n
        dataset_dir={args.dataset_dir},\n
        out_csv={out_csv},\n
        seq_len={args.seq_len},\n
        batch_size={args.batch_size},\n
        model_step={args.model_step},\n
        tmp_ckpt_path={args.tmp_ckpt_path},\n
        model_cluster={args.model_cluster},\n
        device={device},\n
        """
    )
    eval_model(
        ckpt_path=os.path.join(ckpt_path, args.model_step),
        cfg=cfg,
        dataset_dir=args.dataset_dir,
        out_csv=out_csv,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
    )


    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
