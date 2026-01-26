import torch
from src.core.checkpointing import get_full_checkpoint_path, load_training_state
from src.core.metric_loggers import NeptuneLogger, get_metric_logger
from src.core.utils import solve_config_lr
from main import log_environs, upload_config_file
from src.core.distributed_training import setup_fsdp2_model
from torch.distributed.tensor import distribute_tensor, DTensor
from hydra.utils import instantiate
import logging
import platform
import os
from neptune.integrations.python_logger import NeptuneHandler
import torch.distributed.checkpoint as dcp

logger = logging.getLogger(__name__)
logger.propagate = False
ch = logging.StreamHandler()
formatter = logging.Formatter(
    fmt=f"[%(levelname)s][host:{platform.node()}][local_rank:{os.environ.get('LOCAL_RANK')}] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
ch.setFormatter(formatter)
logger.addHandler(ch)


def init_pc_attributes(cfg, metric_logger):

    training_state = load_training_state(cfg.trainer.checkpoint.load)

    if metric_logger is None:
        metric_logger = get_metric_logger(
            metric_logger_config=instantiate(
                cfg.infrastructure.metric_logger, _convert_="all"
            ),
            neptune_run_id=training_state["run_id"],
        )

        # Other loggers do not have `run` method
        if isinstance(metric_logger, NeptuneLogger):
            npt_handler = NeptuneHandler(run=metric_logger.run)
            logger.addHandler(npt_handler)

    learning_rate, exp_lr = solve_config_lr(cfg.trainer.learning_rate)

    if isinstance(metric_logger, NeptuneLogger) and (
        training_state["run_id"] is None
        or cfg.infrastructure.metric_logger.new_neptune_job
    ):
        metric_logger.run["job_config"] = cfg
        upload_config_file(metric_logger)
        log_environs(metric_logger)
        metric_logger.run[f"job/full_save_checkpoints_path"] = get_full_checkpoint_path(
            cfg.trainer.checkpoint.save.path
        )
        metric_logger.run["learning_rate"] = learning_rate
        metric_logger.run["exp_lr"] = exp_lr

    torch.manual_seed(cfg.trainer.train_dataloader.dataset.seed)

    model = create_model(
        cfg.model,
        cfg.projected_compression,
        cfg.projected_compression.source_model_for_distillation,
    )
    if cfg.projected_compression.separate_block_optimizers:
        target_model_optimize_params = get_target_model_optimize_params(model)

        target_model_optimizer = torch.optim.AdamW(
            target_model_optimize_params,
            lr=learning_rate,
            weight_decay=cfg.trainer.weight_decay,
        )
        scheduler_fn = instantiate(cfg.trainer.scheduler)
        target_model_scheduler = scheduler_fn(
            optimizer=target_model_optimizer, n_steps=cfg.trainer.n_steps
        )

        optimizer = [target_model_optimizer]
        scheduler = [target_model_scheduler]
        for block in model.projections.blocks:
            block_optimizer = torch.optim.AdamW(
                block.parameters(),
                lr=learning_rate,
                weight_decay=cfg.trainer.weight_decay,
            )
            optimizer.append(block_optimizer)
            scheduler.append(
                scheduler_fn(optimizer=block_optimizer, n_steps=cfg.trainer.n_steps)
            )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=cfg.trainer.weight_decay,
        )
        scheduler = instantiate(cfg.trainer.scheduler)(
            optimizer=optimizer, n_steps=cfg.trainer.n_steps
        )

    if cfg.trainer.checkpoint.load.path is not None:
        load_checkpoint(model, optimizer, scheduler, cfg.trainer.checkpoint.load.path)

    return model, optimizer, scheduler, training_state, metric_logger


def load_checkpoint(model, optimizer, scheduler, checkpoint_folder):
    dcp.load(model.state_dict(), checkpoint_id=f"{checkpoint_folder}/model")

    if type(optimizer) is not list:
        dcp.load(optimizer.state_dict(), checkpoint_id=f"{checkpoint_folder}/optimizer")
    else:
        dcp.load(
            optimizer[0].state_dict(), checkpoint_id=f"{checkpoint_folder}/optimizer"
        )
        for idx, block_optimizer in enumerate(optimizer[1:]):
            dcp.load(
                block_optimizer.state_dict(),
                checkpoint_id=f"{checkpoint_folder}/block_optimizer_{idx}",
            )
    # TODO: missing scheduler loader
    logger.info(
        f"Loaded sharded model checkpoint checkpoint from '{checkpoint_folder}'!"
    )


def get_target_model_optimize_params(model):
    params = []
    for block in model.target_model.encoder.blocks:
        params.extend(block.attention_layer.norm.parameters())
        params.extend(block.ff_layer.norm.parameters())

    params.extend(model.target_model.head.norm.parameters())
    params.extend(model.projections.head.parameters())
    params.append(model.projections.embedding)
    params.extend(model.projections.auxiliary_embedding_weights.parameters())
    return params


def create_model(cfg_model, cfg_projected_compression, source_model_for_distillation):
    with torch.device("meta"):
        model = instantiate(
            cfg_model,
            path_to_importances=cfg_projected_compression.path_to_importances,
            adjust_grad_norm=cfg_projected_compression.adjust_grad_norm,
            _convert_="all",
        )

    # Only layer norms from target_model are used
    if not source_model_for_distillation:
        for block in model.source_model.encoder.blocks:
            block.attention_layer.norm = None
            block.ff_layer.norm = None
        model.source_model.head.norm = None
    # embedding from source_model is used
    model.target_model.embedding = None

    model = setup_fsdp2_model(model, cfg_projected_compression)

    # Initializing model.source_model
    source_sd = torch.load(
        cfg_projected_compression.source_model_path,
        mmap=True,
        weights_only=True,
        map_location="cpu",
    )

    # We have set source_layer to None, so we do not want to
    source_norms = {}
    for k in list(source_sd.keys()):
        if "norm" in k:
            if source_model_for_distillation:
                source_norms[k] = source_sd[k].clone().detach()
            else:
                source_norms[k] = source_sd.pop(k)

    sharded_sd = get_sharded_sd(model.source_model.state_dict(), source_sd)
    model.source_model.load_state_dict(sharded_sd, strict=False, assign=True)

    # It is used in forward step, but we do not need gradient
    model.source_model.embedding.weight.requires_grad = False

    # Initializing model.target_model
    model.target_model.to_empty(device="cuda")

    if cfg_projected_compression.init_norms_with_ones:
        ones = torch.ones(model.target_model.head.norm.weight.shape, device="cuda")

        sharded_tensor = distribute_tensor(
            ones,
            model.target_model.head.norm.weight.device_mesh,
            model.target_model.head.norm.weight.placements,
        )
        model.target_model.head.norm.weight.data.copy_(sharded_tensor)

        for block in model.target_model.encoder.blocks:
            block.attention_layer.norm.weight.data.copy_(sharded_tensor)
            block.ff_layer.norm.weight.data.copy_(sharded_tensor)
            block.attention_layer.layer.rope.register_freqs()

        if source_model_for_distillation:
            for block in model.source_model.encoder.blocks:
                block.attention_layer.layer.rope.register_freqs()
    else:
        dmodel_topk_indices, dff_topk_indices = get_topk_indices(
            cfg_projected_compression.path_to_importances,
            model.projections.target_dmodel,
            model.projections.target_dff,
        )
        dmodel_topk_indices = dmodel_topk_indices.detach().cpu()
        dff_topk_indices = [indices.detach().cpu() for indices in dff_topk_indices]
        weight = source_norms["head.norm.weight"][dmodel_topk_indices]
        sharded_tensor = distribute_tensor(
            weight,
            model.target_model.head.norm.weight.device_mesh,
            model.target_model.head.norm.weight.placements,
        )
        model.target_model.head.norm.weight.data.copy_(sharded_tensor)

        for i, block in enumerate(model.target_model.encoder.blocks):
            weight = source_norms[f"encoder.blocks.{i}.attention_layer.norm.weight"][
                dmodel_topk_indices
            ]
            sharded_tensor = distribute_tensor(
                weight,
                model.target_model.head.norm.weight.device_mesh,
                model.target_model.head.norm.weight.placements,
            )
            block.attention_layer.norm.weight.data.copy_(sharded_tensor)
            weight = source_norms[f"encoder.blocks.{i}.ff_layer.norm.weight"][
                dmodel_topk_indices
            ]
            sharded_tensor = distribute_tensor(
                weight,
                model.target_model.head.norm.weight.device_mesh,
                model.target_model.head.norm.weight.placements,
            )
            block.ff_layer.norm.weight.data.copy_(sharded_tensor)
            block.attention_layer.layer.rope.register_freqs()

        if source_model_for_distillation:
            for i, block in enumerate(model.source_model.encoder.blocks):
                block.attention_layer.layer.rope.register_freqs()

    # Initializing model.projections
    model.projections.to_empty(device="cuda")
    model.projections.init_projection_weights(
        cfg_projected_compression.path_to_importances
    )

    return model


def get_sharded_sd(target_sd, source_sd):
    sharded_sd = {}
    for param_name, full_tensor in source_sd.items():
        sharded_meta_param = target_sd.get(param_name)
        sharded_tensor = distribute_tensor(
            full_tensor,
            sharded_meta_param.device_mesh,
            sharded_meta_param.placements,
        )
        sharded_sd[param_name] = torch.nn.Parameter(sharded_tensor)
    return sharded_sd


def get_topk_indices(dimensions_importances_path, target_dmodel, target_dff):

    dimensions_importances = torch.load(dimensions_importances_path)
    dmodel_importances = dimensions_importances["dmodel_importances"]
    dff_importances = dimensions_importances["dff_importances"]

    dmodel_indices = torch.topk(
        dmodel_importances, dim=0, largest=True, k=target_dmodel
    ).indices
    dff_indices = []
    for i in range(len(dff_importances)):
        dff_top_indices_current = torch.topk(
            dff_importances[i], dim=0, largest=True, k=target_dff
        ).indices
        dff_indices.append(dff_top_indices_current)

    return dmodel_indices, dff_indices
