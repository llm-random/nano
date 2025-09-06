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


logger = logging.getLogger(__name__)


MODEL_SD_FILENAME = "__model_state_dict.pt"
OPTIMIZER_SD_FILENAME = "__optimizer_state_dict.pt"
SCHEDULER_SD_FILENAME = "__scheduler_state_dict.pt"
TRAINING_SD_FILENAME = "__training_state_dict.pt"

def step_checkpoint_path(path, step):
    full_config_path = get_full_checkpoint_path(path)
    return f"{full_config_path}/step_{step}"

def get_full_checkpoint_path(path):
    slurm_array_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    slurm_job_id = os.getenv("SLURM_JOB_ID")

    if slurm_array_task_id and slurm_job_id:
        return f"{path}/{slurm_job_id}/{slurm_array_task_id}"
    else:
        return f"{path}"
        
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

def load_model_state(model, state_dict):
    if not isinstance(model, FSDPModule):
        print("Loading non-FSDP model state dict.")
        model.load_state_dict(state_dict, assign=True)
        print("Loaded non-FSDP model state dict.")
    else:
        print("Loading FSDP model state dict.")
        set_model_state_dict(
            model, 
            state_dict,                 
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
            )
        )
        print("Loaded FSDP model state dict.")

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

def save_checkpoint(checkpoint_path: str, model, optimizer=None, scheduler=None, training_state=None, apply_functions=[]):
    os.makedirs(checkpoint_path, exist_ok=True)
    if isinstance(model, FSDPModule):
        state_dict = get_model_state_dict(
            model,             
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
            )
        )
        for fn in apply_functions:
            fn(model)

        if os.environ["RANK"] == "0":
            torch.save(state_dict, f"{checkpoint_path}/{MODEL_SD_FILENAME}")
            logger.info(f"Saved model checkpoint in '{checkpoint_path}'")

        if optimizer is not None:
            state_dict = get_optimizer_state_dict(
                model,
                optimizer,
                options=StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                )
            )
            if os.environ["RANK"] == "0":
                torch.save(state_dict, f"{checkpoint_path}/{OPTIMIZER_SD_FILENAME}")
                logger.info(f"Saved optimizer checkpoint in '{checkpoint_path}'")

    else:
        state_dict = model.state_dict()
        torch.save(state_dict, f"{checkpoint_path}/{MODEL_SD_FILENAME}")
        logger.info(f"Saved model checkpoint in '{checkpoint_path}'")
        if optimizer is not None:
            state_dict = optimizer.state_dict()
            torch.save(state_dict, f"{checkpoint_path}/{OPTIMIZER_SD_FILENAME}")
            logger.info(f"Saved optimizer checkpoint in '{checkpoint_path}'")

    if os.environ["RANK"] == "0":
        if scheduler is not None:
            state_dict = scheduler.state_dict()
            torch.save(state_dict, f"{checkpoint_path}/{SCHEDULER_SD_FILENAME}")
            logger.info(f"Saved scheduler checkpoint in '{checkpoint_path}'")
            
        if training_state is not None:
            torch.save(training_state, f"{checkpoint_path}/{TRAINING_SD_FILENAME}")
            logger.info(f"Saved training state checkpoint in '{checkpoint_path}'")
