import os
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
import torch.distributed.checkpoint as dcp
from torch.nn.parallel import DistributedDataParallel as DDP

from src.core.metric_loggers import NeptuneLogger
import logging

logger = logging.getLogger(__name__)

class TrainingState(Stateful):
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(
            self.model, self.optimizer
        )
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict,
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )
        self.scheduler.load_state_dict(state_dict["scheduler"])



def step_checkpoint_path(path, step):
    full_config_path = get_full_checkpoint_path(path)
    return f"{full_config_path}/step_{step}"


def save_training_state(
    save_config,
    step,
    processed_tokens,
    metric_logger=None,
):
    run_id = (
        metric_logger.run["sys/id"].fetch()
        if type(metric_logger) is NeptuneLogger
        else None
    )

    path = step_checkpoint_path(save_config.path, step)
    torch.save(
        {"next_step": step + 1, "run_id": run_id, "processed_tokens": processed_tokens},
        f"{path}/{save_config.training_state_filename}",
    )

    logger.info(
        f"Saved training state in '{save_config.path}/{save_config.training_state_filename}'"
    )


def get_full_checkpoint_path(path):
    slurm_array_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    return (
        f"{path}/{slurm_array_task_id}"
        if slurm_array_task_id is not None
        else path
    )


def load_training_state(load_config):
    training_start_config = {"next_step": 0, "run_id": None, "processed_tokens": 0}

    load_path = load_config.path
    if load_path is None or load_config.get("training_state_filename") is None:
        logger.warning(
            "Save path training_state_filename  is not set. Starting training from scratch."
        )
        return training_start_config
    load_path = get_full_checkpoint_path(
        load_path
    )
    os.makedirs(load_path, exist_ok=True)

    training_state_path = (
        f"{load_path}/{load_config.training_state_filename}"
    )
    if os.path.isfile(training_state_path):
        return torch.load(training_state_path)
    else:
        logger.warning(
            f"Training state file '{training_state_path}' not found. "
            "Starting training from scratch."
        )

    return training_start_config


def _find_latest_checkpoint(path: str) -> str:
    files = [os.path.join(path, f) for f in os.listdir(path)]
    if not files:
        logger.info(f"No checkpoints in '{path}'")
        return

    return max(files, key=os.path.getmtime)


def load_checkpoint_from_file(load_config, model, optimizer, scheduler): #dev TODO remove or refactor for checkpoint manager*
    checkpoint_path = load_config.path
    if checkpoint_path is None:
        return 

    if checkpoint_path is not None:
        if isinstance(model, FSDP):
            # Sharded load
            state_dict = {"app": TrainingState(model, optimizer, scheduler)}
            dcp.load(state_dict=state_dict, checkpoint_id=checkpoint_path)
            logger.debug(f"Loaded sharded checkpoint from '{checkpoint_path}'")
        else:
            # Non-sharded load
            checkpoint_model = (
                f"{checkpoint_path}/{load_config.model_checkpoint_filename}"
            )
            checkpoint = torch.load(checkpoint_model)
            if type(model) is DDP:
                logger.info(f"Loading DDP model from '{checkpoint_path}'")
                model.module.load_state_dict(checkpoint["model"])
            else:
                logger.info(f"Loading non-DDP model from '{checkpoint_path}'")
                model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optim"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            logger.info(f"Loaded non-sharded sheduler from '{checkpoint_path}'")
            logger.debug(f"Loaded non-sharded checkpoint from '{checkpoint_path}'")
