import os
import hydra
import yaml
from src.core.conversion_from_finalized_pc import load_finalized_pc_checkpoint
from src.core.distributed_training import setup_distributed_training
from src.core.conversion_from_llmrandom import load_llmrandom_checkpoint
from src.core.llama import copy_llama_model_weights_from_HF
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
import logging
from neptune.integrations.python_logger import NeptuneHandler
from src.core.checkpointing import load_checkpoint_from_file, load_training_state, get_full_checkpoint_path
from src.core.metric_loggers import NeptuneLogger, get_metric_logger
from src.core.model import Residual
import platform

logger = logging.getLogger(__name__)
logger.propagate = False
ch = logging.StreamHandler()
formatter = logging.Formatter(
    fmt=f"[%(levelname)s][host:{platform.node()}][local_rank:{os.environ.get('LOCAL_RANK')}] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
ch.setFormatter(formatter)
logger.addHandler(ch)

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
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=torch.device(f"cuda:{local_rank}"))
        torch.cuda.set_device(local_rank)
    else:
        logger.warning("CUDA is not available. Running on CPU and 'gloo' backend.")
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def log_environs(metric_logger):
    scrap_keys = [
        "SLURM_MEM_PER_GPU", 
        "SLURM_JOB_USER", 
        "SLURM_TASKS_PER_NODE", 
        "SLURM_JOB_UID", 
        "SLURM_TASK_PID", 
        "CONDA_EXE", 
        "SLURM_ARRAY_TASK_STEP", 
        "TMUX", 
        "SLURM_JOB_GPUS", 
        "SLURM_LOCALID", 
        "SLURM_SUBMIT_DIR", 
        "HOSTNAME", 
        "SLURMD_NODENAME",
        "SLURM_JOB_START_TIME", 
        "SLURM_CLUSTER_NAME", 
        "SLURM_JOB_END_TIME", 
        "SLURM_CPUS_ON_NODE", 
        "SLURM_JOB_CPUS_PER_NODE", 
        "SLURM_GPUS_ON_NODE", 
        "LOGNAME", 
        "USER",
        "SLURM_NODELIST",
        "SLURM_JOB_PARTITION", 
        "SLURM_JOB_ACCOUNT",
        "SLURM_NPROCS",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_ID",
        "SLURM_JOBID", 
        "SLURM_CONF", 
        "SLURM_ARRAY_TASK_COUNT", 
        "PATH", 
        "SLURM_ARRAY_JOB_ID", 
        "SLURM_JOB_NAME", 
        "SLURM_JOB_GID", 
        "CUDA_MODULE_LOADING", 
        "RANK", 
        "LOCAL_RANK", 
        "CUDA_DEVICE_ORDER",
        "SLURM_TOPOLOGY_ADDR",
        "HOME",
    ]

    environs = os.environ
    for environ_key in scrap_keys:
        metric_logger.run[f"job/{environ_key}"] = str(environs.get(environ_key))

def run(cfg: OmegaConf, metric_logger=None):
    setup_enviroment()

    if "distributed" in cfg.trainer and cfg.trainer.distributed is not None:
        distributed_setup()

    training_state = load_training_state(cfg.trainer.checkpoint.load)

    if metric_logger is None:
        metric_logger = get_metric_logger(
            metric_logger_config=instantiate(cfg.infrastructure.metric_logger, _convert_="all"),
            neptune_run_id=training_state["run_id"],
        )
        npt_handler = NeptuneHandler(run=metric_logger.run)
        logger.addHandler(npt_handler)


    if isinstance(metric_logger, NeptuneLogger) and (training_state["run_id"] is None or cfg.infrastructure.metric_logger.new_neptune_job):
        metric_logger.run["job_config"] = cfg
        upload_config_file(metric_logger)
        log_environs(metric_logger)
        metric_logger.run[f"job/full_save_checkpoints_path"] = get_full_checkpoint_path(cfg.trainer.checkpoint.save.path)
        
    torch.manual_seed(cfg.trainer.train_dataloader.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Creating model...")
    model = instantiate(cfg.model, _convert_="all").to(device)
    logger.info(f"Model {model.__class__.__name__} created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    # Residual layers needs metric_logger for logging update norms
    for _, module in model.named_modules():
        if isinstance(module, Residual):
            module.set_metric_logger(metric_logger)

    if cfg.trainer.checkpoint.load.type == "huggingface":
        copy_llama_model_weights_from_HF(model, cfg.trainer.checkpoint.load.path)
        if cfg.get("apply_functions", None):
            for fn in instantiate(cfg.apply_functions):
                fn(model)
        model = setup_distributed_training(model, cfg.trainer.distributed)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.trainer.learning_rate,
            weight_decay=cfg.trainer.weight_decay,
        )
        scheduler = instantiate(cfg.trainer.scheduler)(optimizer=optimizer, n_steps=cfg.trainer.n_steps)
    elif cfg.trainer.checkpoint.load.type == "llm-random":
        load_llmrandom_checkpoint(cfg.trainer.checkpoint.load, model)
        if cfg.get("apply_functions", None):
            for fn in instantiate(cfg.apply_functions):
                fn(model)
        model = setup_distributed_training(model, cfg.trainer.distributed)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.trainer.learning_rate,
            weight_decay=cfg.trainer.weight_decay,
        )
        scheduler = instantiate(cfg.trainer.scheduler)(optimizer=optimizer, n_steps=cfg.trainer.n_steps)
    elif cfg.trainer.checkpoint.load.type == "finalized_pc":
        load_finalized_pc_checkpoint(model, cfg.trainer.checkpoint.load)
        if cfg.get("apply_functions", None):
            for fn in instantiate(cfg.apply_functions):
                fn(model)
        model = setup_distributed_training(model, cfg.trainer.distributed)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.trainer.learning_rate,
            weight_decay=cfg.trainer.weight_decay,
        )
        scheduler = instantiate(cfg.trainer.scheduler)(optimizer=optimizer, n_steps=cfg.trainer.n_steps)
    elif cfg.trainer.checkpoint.load.type == "nano":
        if cfg.get("apply_functions", None):
            for fn in instantiate(cfg.apply_functions):
                fn(model)
        model = setup_distributed_training(model, cfg.trainer.distributed)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.trainer.learning_rate,
            weight_decay=cfg.trainer.weight_decay,
        )
        scheduler = instantiate(cfg.trainer.scheduler)(optimizer=optimizer, n_steps=cfg.trainer.n_steps)
        
        load_checkpoint_from_file(cfg.trainer.checkpoint.load, model, optimizer, scheduler)
        if cfg.trainer.checkpoint.load.only_weights:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.trainer.learning_rate,
                weight_decay=cfg.trainer.weight_decay,
            )
            scheduler = instantiate(cfg.trainer.scheduler)(optimizer=optimizer, n_steps=cfg.trainer.n_steps)
    else:
        raise Exception(f"Not recognized load checkpoint format: {cfg.trainer.checkpoint.load.type}")
    
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
def main(config: OmegaConf):
    run(config)

if __name__ == "__main__":
    main()
