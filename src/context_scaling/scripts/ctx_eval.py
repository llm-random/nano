"""Per-task context-scaling eval. Runs on cluster compute node via sbatch array.

Picks job[SLURM_ARRAY_TASK_ID] from jobs.json (written by run_ctx_eval.py's
local setup), loads checkpoint, computes per-token loss, uploads batch-mean
per-position loss to the source wandb run.
"""
import json
import os
import sys
from functools import partial
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from datasets import load_from_disk
from hydra.utils import instantiate
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# repo root on sys.path so `_target_: src.core.model.LLM` resolves under hydra
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from eval_models import (
    ModelOnly,
    batch_per_token_losses,
    collate_no_pad,
    load_cfg_from_yaml,
    setup_distributed,
)
from wandb_utils import WANDB_PROJECT, upload_mean_loss_to_wandb


@hydra.main(version_base=None, config_path="../../../configs", config_name="ctx_eval")
def main(cfg):
    eval_cfg = cfg.eval
    out_dir = Path(eval_cfg.out_dir)
    device = setup_distributed()

    with open(out_dir / "jobs.json", "r", encoding="utf-8") as f:
        jobs = json.load(f)

    # jobs.json holds 1 row per run with model_step as a list; flatten to per-task
    flat = [{**r, "model_step": s} for r in jobs for s in r["model_step"]]

    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    job = flat[task_id]
    run_id = job["jobID"]
    ckpt_dir = job["ckpt_path"]
    yaml_path = Path(job["yaml_config_path"])
    seq_len = job["seq_len"]
    model_step = job["model_step"]
    wandb_project = job.get("wandb_project", WANDB_PROJECT)

    print(f"task_id={task_id} run_id={run_id} step={model_step} seq_len={seq_len}")
    run_cfg = load_cfg_from_yaml(yaml_path)

    try:
        model = instantiate(run_cfg.model, _convert_="all").to(device)
        model.eval()
        fsdp_model = FSDP(model)
        state = {"app": ModelOnly(fsdp_model)}
        ckpt_full = os.path.join(ckpt_dir, f"step_{model_step}")
        dcp.load(state, checkpoint_id=ckpt_full)
        fsdp_model.eval()

        ds = load_from_disk(eval_cfg.dataset_dir)
        tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        collate_fn = partial(collate_no_pad, tokenizer=tokenizer, seq_len=seq_len)
        loader = DataLoader(
            ds, batch_size=eval_cfg.batch_size, shuffle=False, collate_fn=collate_fn
        )

        all_losses = []
        for batch in tqdm(loader):
            if batch is None:
                continue
            with torch.no_grad():
                losses, _ = batch_per_token_losses(
                    fsdp_model, batch["input_ids"], device
                )
            all_losses.append(losses)

        stacked = torch.cat([t.detach().cpu() for t in all_losses], dim=0)
        upload_mean_loss_to_wandb(
            run_id=run_id,
            project=wandb_project,
            mean_losses=stacked.mean(dim=0),
            model_step=model_step,
            eval_seq_len=seq_len,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
