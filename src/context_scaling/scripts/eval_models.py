import os
import re
import subprocess
from pathlib import Path
import argparse
from tqdm.auto import tqdm
import pandas as pd
import neptune

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


def assert_unique_keyword_matches(template: str, yaml_keys: list[str]) -> None:
    """Ensure each keyword in template matches exactly one yaml key."""
    for kw in template.split(","):
        kw = kw.strip()
        matches = [k for k in yaml_keys if kw in k]
        assert (
            len(matches) == 1
        ), f"Keyword '{kw}' matched {len(matches)} yaml_keys: {matches}"


def make_csv_name(template: str, yaml_overrides: list[str]) -> str:
    """Build CSV filename from template keywords and yaml_overrides values.

    Example: template="kv_h,dff", overrides=["common.kv_heads=4", "common.dff=512"]
    Returns: "kv_h=4-dff=512.csv"
    """
    overrides_dict = {}
    for ov in yaml_overrides:
        key, val = ov.split("=", 1)
        overrides_dict[key.split(".")[-1]] = val  # use last part of key

    parts = []
    for kw in template.split(","):
        kw = kw.strip()
        matches = [k for k in overrides_dict if kw in k]
        if not matches:
            raise ValueError(
                f"Keyword '{kw}' not found in yaml_overrides: {yaml_overrides}"
            )
        parts.append(f"{kw}={overrides_dict[matches[0]]}")

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


def setup_distributed():
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    dist.init_process_group(
        backend="nccl",
        rank=0,
        world_size=1,
    )

    print("✅ torch.distributed initialized (1 GPU)")
    return device


def rsync_checkpoint(cluster: str, remote_path: str, local_dir: str) -> None:
    "Rsync a checkpoint from a remote cluster to local directory."

    cmd = ["rsync", "-rlphvP", f"{cluster}:{remote_path}", local_dir]
    subprocess.run(cmd, check=True)


def eval_model(
    ckpt_dir: str,
    exp_config: str,
    yaml_overrides: list,
    dataset_dir: str,
    out_csv: str,
    seq_len: int,
    batch_size: int,
):
    device = setup_distributed()

    config_dir = str(Path.cwd() / "configs")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name=exp_config,
            overrides=yaml_overrides,
        )

    model = instantiate(cfg.model, _convert_="all").to(device)
    model.eval()

    print(f"Model instantiated on {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    fsdp_model = FSDP(model)

    class ModelOnly(Stateful):
        def __init__(self, model):
            self.model = model

        def state_dict(self):
            return {"model": self.model.state_dict()}

        def load_state_dict(self, sd):
            self.model.load_state_dict(sd["model"], strict=True)

    state = {"app": ModelOnly(fsdp_model)}

    # rsync from helios
    tmp_ckpt_path = "/storage_nvme_4/nano/models/from_helios/rsync_from_python"
    os.makedirs(tmp_ckpt_path, exist_ok=True)
    rsync_checkpoint(
        "helios",
        ckpt_dir,
        tmp_ckpt_path,
    )

    dcp.load(state, checkpoint_id=tmp_ckpt_path)

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

    def collate_no_pad(batch):
        texts = [ex["text"] for ex in batch]
        urls = [ex["url"] for ex in batch]
        timestamps = [ex["timestamp"] for ex in batch]

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
            "timestamp": timestamps,
        }

    @torch.no_grad()
    def batch_per_token_losses(model, input_ids):
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

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_no_pad,
    )

    model.eval()

    all_losses = []

    for batch in tqdm(loader):
        if batch is None:
            continue

        losses, _ = batch_per_token_losses(model, batch["input_ids"])
        all_losses.append(losses)

    def tensors_rows_to_csv(tensors, path="tensors.csv"):
        rows = []
        for t in tensors:
            rows.append(t.detach().cpu())
        stacked = torch.cat(rows, dim=0)  # (num_tensors * N, N)
        pd.DataFrame(stacked.numpy()).to_csv(path, index=False)

    tensors_rows_to_csv(all_losses, path=out_csv)


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-token losses and export to CSV"
    )

    parser.add_argument(
        "--tags",
        nargs="+",
        required=True,
        help="Neptune tags to INCLUDE (space-separated)",
    )
    parser.add_argument(
        "--grid_params",
        nargs="+",
        required=True,
        help="params to overwrite in hydra config",
    )
    parser.add_argument(
        "--exp_config",
        type=str,
        required=True,
        help="path to yaml config, for model initialization",
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
        help="Sequence length (e.g. 2048 or 8192)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )

    args = parser.parse_args()

    yaml_keys = [
        re.sub(r"^job_config/", "", p).replace("/", ".") for p in args.grid_params
    ]
    assert_unique_keyword_matches(args.out_csv_format, yaml_keys)

    df = get_neptune_table(tags=args.tags)

    for _, row in df.iterrows():
        yaml_overrides = [f"{k}={row[p]}" for k, p in zip(yaml_keys, args.grid_params)]

        ckpt_path = row["job/full_save_checkpoints_path"]
        out_csv = str(
            Path(args.out_dir) / make_csv_name(args.out_csv_format, yaml_overrides)
        )
        print(
            f"""
            DEBUG eval_model params: ckpt_dir={ckpt_path},\n
            exp_config={args.exp_config},\n
            yaml_overrides={yaml_overrides},\n
            dataset_dir={args.dataset_dir},\n
            out_csv={out_csv},\n
            seq_len={args.seq_len},\n
            batch_size={args.batch_size}
            """
        )
        eval_model(
            ckpt_dir=ckpt_path,
            exp_config=args.exp_config,
            yaml_overrides=yaml_overrides,
            dataset_dir=args.dataset_dir,
            out_csv=out_csv,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
