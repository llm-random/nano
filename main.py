import os
import hydra
import yaml
from grid_generator.generate_configs import create_grid_config
from grid_generator.sbatch_builder import generate_sbatch_script
import resolver as _  # I should be able to ignore this line by linter, but ~ things like # ignore did not work
import logging
from omegaconf import OmegaConf

import os
import torch
import torch.distributed as dist
import logging
from hydra.utils import instantiate
from torch.nn.parallel import DistributedDataParallel as DDP
import logging

from src.core.checkpointing import load_checkpoint, load_training_state
from src.core.metric_loggers import NeptuneLogger, get_metric_logger
from src.core.model import Residual, wrap_model

logger = logging.getLogger(__name__)


def dump_grid_configs(configs_grid, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    class CustomDumper(yaml.SafeDumper):
        def write_line_break(self, data=None):
            super().write_line_break(data)
            if len(self.indents) == 1:  # Check if we're at the root level
                super().write_line_break()

    for idx, (cfg_dict, overrides_list) in enumerate(configs_grid):
        cfg_dict["overrides"] = overrides_list
        cfg_dict["_run_"] = True

        out_path = os.path.join(output_folder, f"config_{idx}.yaml")
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg_dict, f, Dumper=CustomDumper, sort_keys=True)


def upload_config_file(metric_logger):
    slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    file_path = f"generated_configs/config_{slurm_array_task_id}.yaml"
    if slurm_array_task_id is not None and os.path.exists(file_path):
        metric_logger.run["yaml_config"].upload(
            f"generated_configs/config_{slurm_array_task_id}.yaml"
        )

def check_env_vars():
    assert int(os.environ["RANK"]) < int(os.environ["WORLD_SIZE"])


def setup_enviroment():
    if "WORLD_SIZE" not in os.environ:
        logger.warning("WORLD_SIZE is not set, setting it to 1")
        os.environ["WORLD_SIZE"] = "1"

    if "RANK" not in os.environ:
        if "SLURM_PROCID" in os.environ:
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
        else:
            logger.warning("RANK is not set, setting it to 0")
            os.environ["RANK"] = "0"

    if "LOCAL_RANK" not in os.environ:
        if "SLURM_LOCALID" in os.environ:
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        else:
            logger.warning("LOCAL_RANK is not set, setting it to 0")
            os.environ["LOCAL_RANK"] = "0"

    if "MASTER_ADDR" not in os.environ:
        default_master_addr = "localhost"
        logger.warning(f"MASTER_ADDR is not set, setting it to {default_master_addr}")
        os.environ["MASTER_ADDR"] = default_master_addr

    if "MASTER_PORT" not in os.environ:
        default_master_port = "12355"
        logger.warning(f"MASTER_PORT is not set, setting it to {default_master_port}")
        os.environ["MASTER_PORT"] = default_master_port

    check_env_vars()


def distributed_setup():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if torch.cuda.is_available():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
    else:
        logger.warning("CUDA is not available. Running on CPU and 'gloo' backend.")
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def run(cfg, metric_logger=None):
    setup_enviroment()

    if "distributed" in cfg.trainer and cfg.trainer.distributed is not None:
        distributed_setup()

    training_state = load_training_state(cfg.trainer.checkpoint)

    if metric_logger is None:
        metric_logger = get_metric_logger(
            metric_logger_config=instantiate(cfg.infrastructure.metric_logger, _convert_="all"),
            neptune_run_id=training_state["run_id"],
        )

    if isinstance(metric_logger, NeptuneLogger):
        metric_logger.run["job_config"] = cfg
        upload_config_file(metric_logger)

    torch.manual_seed(cfg.trainer.train_dataloader.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = instantiate(cfg.model, _convert_="all").to(device)

    # Residual layers needs metric_logger for logging update norms
    for _, module in model.named_modules():
        if isinstance(module, Residual):
            module.set_metric_logger(metric_logger)

    if "distributed" in cfg.trainer and cfg.trainer.distributed is not None:
        if torch.cuda.is_available():
            model = wrap_model(model, cfg.trainer.distributed.fsdp)
        else:
            logger.info("FSDP is not supported with CPU. Running DDP instead")
            model = DDP(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.trainer.learning_rate,
        weight_decay=cfg.trainer.weight_decay,
    )

    scheduler = instantiate(cfg.trainer.scheduler)(optimizer=optimizer, n_steps=cfg.trainer.n_steps)

    load_checkpoint(cfg.trainer.checkpoint, model, optimizer, scheduler)
    trainer = instantiate(cfg.trainer)
    trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        metric_logger=metric_logger,
    ).train()

    cleanup()


@hydra.main(version_base=None, config_path="configs", config_name="exp")
def main(config):

    if config.get("_run_"):
        run(config)
        return

    configs_grid = create_grid_config(config)
    dump_grid_configs(configs_grid, config.infrastructure.generated_configs_path)

    modules_to_add = config.infrastructure.get("modules_to_add", None)
    generate_sbatch_script(
        config.infrastructure.slurm, config.infrastructure.generated_configs_path, len(configs_grid), config.infrastructure.venv_path, modules_to_add
    )

    if config.get("_debug_"):
        training_config, overrides = configs_grid[0]
        training_config["overrides"] = overrides
        training_config = OmegaConf.create(training_config)
        run(training_config)


if __name__ == "__main__":
    main()
