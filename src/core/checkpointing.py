import os
import torch
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    set_optimizer_state_dict,
    get_model_state_dict,
    get_optimizer_state_dict,
    StateDictOptions,
)

import logging
from torch.distributed.fsdp import FSDPModule

from src.core.conversion_to_hf import save_to_llama_3_hf

logger = logging.getLogger(__name__)

MODEL_SD_FILENAME = "__model_state_dict.pt"
OPTIMIZER_SD_FILENAME = "__optimizer_state_dict.pt"
SCHEDULER_SD_FILENAME = "__scheduler_state_dict.pt"
TRAINING_SD_FILENAME = "__training_state_dict.pt"

def get_full_checkpoint_path(path):
    slurm_array_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    slurm_job_id = os.getenv("SLURM_JOB_ID")

    if slurm_array_task_id and slurm_job_id:
        return f"{path}/{slurm_job_id}/{slurm_array_task_id}"
    else:
        return f"{path}"

def step_checkpoint_path(path, step):
    full_config_path = get_full_checkpoint_path(path)
    return f"{full_config_path}/step_{step}"


def load_training_state(checkpoint_path):
    training_state_path = f"{checkpoint_path}/{TRAINING_SD_FILENAME}"
    if os.path.exists(training_state_path):
        return torch.load(training_state_path)
    else:
        logger.warning(
            f"Training state file '{training_state_path}' not found. "
            "Starting training from scratch."
        )
        return {"next_step": 0, "run_id": None, "processed_tokens": 0}


def _find_latest_checkpoint(path: str) -> str:
    files = [os.path.join(path, f) for f in os.listdir(path)]
    if not files:
        logger.info(f"No checkpoints in '{path}'")
        return

    return max(files, key=os.path.getmtime)


def load_optimizer(model, optimizer, checkpoint_path):
    optimizer_path = f"{checkpoint_path}/{OPTIMIZER_SD_FILENAME}" 
    if os.path.exists(optimizer_path):
        state_dict = torch.load(optimizer_path, mmap=True, weights_only=True, map_location="cpu")
        if isinstance(model, FSDPModule):
            set_optimizer_state_dict(
                model=model,
                optimizers=optimizer,
                optim_state_dict=state_dict,
                options=StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                ),
            )   
        else:
            optimizer.load_state_dict(state_dict)
    else:
        logger.warning(f"Optimizer state dict is missing in '{checkpoint_path}'")

def load_scheduler(scheduler, checkpoint_path):
    scheduler_path = f"{checkpoint_path}/{SCHEDULER_SD_FILENAME}" 
    if os.path.exists(scheduler_path):
        state_dict = torch.load(scheduler_path, mmap=True, weights_only=True, map_location="cpu")
        scheduler.load_state_dict(state_dict)
    else:
        logger.warning(f"Sheduler state dict is missing in '{scheduler_path}'")

def load_model_state_dict(checkpoint_path):
    model_path = f"{checkpoint_path}/{MODEL_SD_FILENAME}" 
    return torch.load(
        model_path, mmap=True, weights_only=True, map_location="cpu"
    )

def save_checkpoint(checkpoint_path: str, model, optimizer=None, scheduler=None, training_state=None, convert_to_hf: bool = False):
    os.makedirs(checkpoint_path, exist_ok=True)
    model_state_dict = None
    optimizer_state_dict = None

    if isinstance(model, FSDPModule):
        model_state_dict = get_model_state_dict(
            model,             
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
            )
        )

        if optimizer is not None:
            optimizer_state_dict = get_optimizer_state_dict(
                model,
                optimizer,
                options=StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                )
            )

    else:
        model_state_dict = model.state_dict()
        if optimizer is not None:
            optimizer_state_dict = optimizer.state_dict()

    if os.environ["RANK"] == "0":
        if model_state_dict is not None:
            if convert_to_hf:
                    dmodel, dff, n_att_heads, n_kvatt_heads, head_dim, nlayers = model.encoder.get_model_dimensions()

                    save_to_llama_3_hf(
                        model_state_dict, 
                        save_dir = f"{checkpoint_path}/hf_ckpt", 
                        dmodel = dmodel, 
                        dff = dff, 
                        n_att_heads = n_att_heads, 
                        n_kvatt_heads = n_kvatt_heads, 
                        head_dim = head_dim,
                        nlayers = nlayers, 
                    ) 
            else:   
                torch.save(model_state_dict, f"{checkpoint_path}/{MODEL_SD_FILENAME}")
                logger.info(f"Saved model checkpoint in '{checkpoint_path}'")

        if optimizer_state_dict is not None:
            torch.save(optimizer_state_dict, f"{checkpoint_path}/{OPTIMIZER_SD_FILENAME}")
            logger.info(f"Saved optimizer checkpoint in '{checkpoint_path}'")

        if scheduler is not None:
            state_dict = scheduler.state_dict()
            torch.save(state_dict, f"{checkpoint_path}/{SCHEDULER_SD_FILENAME}")
            logger.info(f"Saved scheduler checkpoint in '{checkpoint_path}'")

        if training_state is not None:
            torch.save(training_state, f"{checkpoint_path}/{TRAINING_SD_FILENAME}")
            logger.info(f"Saved training state checkpoint in '{checkpoint_path}'")
