import torch
import torch.nn as nn
from src.core.checkpointing import (
    get_full_checkpoint_path,
    load_training_state,
    step_checkpoint_path,
)
from src.core.metric_loggers import WandbLogger, get_metric_logger
from src.core.utils import solve_config_lr
from src.core.distributed_training import setup_fsdp2_model
from torch.distributed.tensor import distribute_tensor, DTensor
from hydra.utils import instantiate
import logging
import platform
import os
import torch.distributed.checkpoint as dcp

logger = logging.getLogger(__name__)


def _build_checkpoint_path(cfg):
    """Return the full expected checkpoint path including step dir and /hf suffix if applicable."""
    base = get_full_checkpoint_path(cfg.trainer.checkpoint.save.path)
    save_type = cfg.trainer.checkpoint.save.type
    n_steps = cfg.trainer.n_steps
    if n_steps is not None and save_type in ("hf_only", "nano_and_hf"):
        return f"{base}/step_{n_steps - 1}/hf"
    elif n_steps is not None:
        return f"{base}/step_{n_steps - 1}"
    return base


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
        # Reverted instantiate here. Passing raw OmegaConf dict so dot notation works inside get_metric_logger.
        metric_logger = get_metric_logger(
            metric_logger_config=cfg.infrastructure.metric_logger,
            tracker_run_id=training_state["run_id"],
            full_config=cfg,
        )

    learning_rate, exp_lr = solve_config_lr(cfg.trainer.learning_rate)

    if isinstance(metric_logger, WandbLogger) and (
        training_state["run_id"] is None
        or cfg.infrastructure.metric_logger.new_wandb_job
    ):
        if metric_logger.run is not None:
            metric_logger.run.log(
                {
                    "learning_rate": learning_rate,
                    "exp_lr": exp_lr,
                    "full_save_checkpoints_path": _build_checkpoint_path(cfg),
                }
            )

    torch.manual_seed(cfg.trainer.train_dataloader.dataset.seed)

    model = create_model(
        cfg.model,
        cfg.projected_compression,
        cfg.projected_compression.source_model_for_distillation,
    )

    if cfg.projected_compression.separate_block_optimizers:
        target_model_optimize_params = get_target_model_optimize_params(model)

        cpu_offload = cfg.projected_compression.get("cpu_offload_projections", False)
        target_model_optimizer = torch.optim.AdamW(
            target_model_optimize_params,
            lr=learning_rate,
            weight_decay=cfg.trainer.weight_decay,
            # When cpu_offload is active, head/embedding projections are plain GPU tensors
            # while target_model norms are FSDP2 DTensors.  _foreach ops cannot mix the two,
            # so fall back to per-param scalar ops (params here are tiny, no perf cost).
            foreach=not cpu_offload,
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
        attn = block.attention_layer.layer
        if getattr(attn, "q_norm", None) is not None:  # Qwen QK-norm
            params.extend(attn.q_norm.parameters())
            params.extend(attn.k_norm.parameters())

    params.extend(model.target_model.head.norm.parameters())
    params.extend(model.projections.head.parameters())
    params.append(model.projections.embedding)
    params.extend(model.projections.auxiliary_embedding_weights.parameters())
    return params


def create_model(cfg_model, cfg_projected_compression, source_model_for_distillation):
    cpu_offload_projections = cfg_projected_compression.get(
        "cpu_offload_projections", False
    )

    with torch.device("meta"):
        model = instantiate(
            cfg_model,
            path_to_importances=cfg_projected_compression.path_to_importances,
            adjust_grad_norm=cfg_projected_compression.adjust_grad_norm,
            cpu_offload_projections=cpu_offload_projections,
            block_gpu_compute=cfg_projected_compression.get("block_gpu_compute", True),
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

    if cpu_offload_projections:
        # Temporarily detach projections and source_model from the model before FSDP2
        # so that only target_model gets sharded. Projections and source weights will
        # live as plain CPU tensors — no VRAM used for them.
        projections_module = model._modules.pop("projections")
        source_model_module = model._modules.pop("source_model")

    model = setup_fsdp2_model(model, cfg_projected_compression)

    if cpu_offload_projections:
        # Re-attach as plain (non-FSDP2) submodules.
        model.projections = projections_module
        model.source_model = source_model_module

    # Initializing model.source_model
    source_sd = torch.load(
        cfg_projected_compression.source_model_path,
        mmap=True,
        weights_only=True,
        map_location="cpu",
    )

    # We have set source_layer to None, so we do not want to
    source_norms = {}
    qk_norms = {}
    for k in list(source_sd.keys()):
        if "q_norm" in k or "k_norm" in k:
            # Qwen QK-norm: dhead-sized, not compressed. Clone but leave in source_sd
            # so the frozen source QwenAttention loads them (no leftover meta params).
            qk_norms[k] = source_sd[k].clone().detach()
        elif "norm" in k:
            if source_model_for_distillation:
                source_norms[k] = source_sd[k].clone().detach()
            else:
                source_norms[k] = source_sd.pop(k)

    if cpu_offload_projections:
        # Load source model directly as plain CPU tensors — no FSDP2 sharding.
        model.source_model.to_empty(device="cpu")
        model.source_model.load_state_dict(source_sd, strict=False, assign=True)
    else:
        sharded_sd = get_sharded_sd(model.source_model.state_dict(), source_sd)
        model.source_model.load_state_dict(sharded_sd, strict=False, assign=True)

    # Source model weights are frozen — they provide fixed basis for projections.
    for param in model.source_model.parameters():
        param.requires_grad = False

    if cpu_offload_projections:
        # Embedding and head projections run on GPU — only encoder blocks are CPU-offloaded.
        # Move the corresponding source weights to GPU so the GPU path has everything it needs.
        model.source_model.embedding = model.source_model.embedding.to("cuda")
        model.source_model.head = model.source_model.head.to("cuda")

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

    # Qwen QK-norm: copied source -> target verbatim (dhead unchanged, no top-k), trainable.
    for i, block in enumerate(model.target_model.encoder.blocks):
        attn = block.attention_layer.layer
        if getattr(attn, "q_norm", None) is None:
            break  # non-Qwen attention (e.g. Llama) has no QK-norm
        for norm_name in ("q_norm", "k_norm"):
            src = qk_norms[
                f"encoder.blocks.{i}.attention_layer.layer.{norm_name}.weight"
            ]
            tgt = getattr(attn, norm_name).weight
            if hasattr(tgt, "device_mesh"):
                tgt.data.copy_(distribute_tensor(src, tgt.device_mesh, tgt.placements))
            else:
                tgt.data.copy_(src.to(tgt.device))

    # Initializing model.projections.
    # In the FSDP2 case all projections params (including CompressibleBlock weights) are
    # meta DTensors at this point.  Calling to_empty("cuda") allocates them as FULL-SIZE
    # plain CUDA tensors, losing FSDP2 sharding — each GPU would hold all proj params
    # (e.g. 24 GB for 8B 10p) and their optimizer state (~48 GB) → OOM.
    # Fix: use the same pattern as source_model init — build CPU init tensors and call
    # load_state_dict(assign=True) while params are still meta DTensors.  assign=True
    # replaces each meta DTensor with a properly FSDP2-sharded CUDA DTensor in one step,
    # with no intermediate full-size allocation.
    if not cpu_offload_projections and torch.distributed.is_initialized():
        meta_sd = model.projections.state_dict()
        dmodel_topk_indices, dff_topk_indices = get_topk_indices(
            cfg_projected_compression.path_to_importances,
            model.projections.target_dmodel,
            model.projections.target_dff,
        )
        dmodel_topk_indices = dmodel_topk_indices.detach().cpu()
        dff_topk_indices = [idx.detach().cpu() for idx in dff_topk_indices]
        _init_projections_sharded_one_by_one(
            model.projections, meta_sd, dmodel_topk_indices, dff_topk_indices
        )
        logger.info("Initialized projections as FSDP2-sharded CUDA DTensors.")
    else:
        model.projections.to_empty(device="cuda")
        model.projections.init_projection_weights(
            cfg_projected_compression.path_to_importances
        )

    return model


def _init_projections_sharded_one_by_one(
    projections_module, meta_sd, dmodel_topk_indices, dff_topk_indices
):
    """Initialize projections one param at a time to avoid CPU RAM OOM.

    Building the full CPU state dict for all projections at once allocates ~77GB CPU RAM
    for 8B 50% compression. With 4 processes per node that is ~308GB — exceeding node RAM.
    This function processes one param at a time so peak CPU RAM per process stays ~2GB.

    Replicates Projections.init_projection_weights logic:
    - "embedding": identity-like row selection, tensor[arange(target_dmodel), dmodel_topk] = 1
    - "*projection_in_weight": tensor[topk, arange(result_dim)] = 1
    - "*projection_out_weight": tensor[arange(result_dim), topk] = 1
    - everything else (auxiliary weights, auxiliary_embedding_weights.weight): zeros
    """
    target_dmodel = dmodel_topk_indices.shape[0]
    target_dff = dff_topk_indices[0].shape[0]

    for param_name, meta_param in meta_sd.items():
        shape = meta_param.shape  # DTensor exposes the global (full) tensor shape
        tensor = torch.zeros(shape, dtype=torch.float32)

        if param_name == "embedding":
            # shape: (target_dmodel, base_dmodel); tensor[arange(target_dmodel), dmodel_topk] = 1
            tensor[torch.arange(shape[0]), dmodel_topk_indices] = 1

        elif ".projection_in_weight" in param_name:
            # shape: (base_dim, result_dim); tensor[topk, arange(result_dim)] = 1
            result_dim = shape[1]
            if result_dim == target_dff and param_name.startswith("blocks."):
                block_idx = int(param_name.split(".")[1])
                topk = dff_topk_indices[block_idx]
            else:
                topk = dmodel_topk_indices
            tensor[topk, torch.arange(result_dim)] = 1

        elif ".projection_out_weight" in param_name:
            # shape: (result_dim, base_dim); tensor[arange(result_dim), topk] = 1
            result_dim = shape[0]
            if result_dim == target_dff and param_name.startswith("blocks."):
                block_idx = int(param_name.split(".")[1])
                topk = dff_topk_indices[block_idx]
            else:
                topk = dmodel_topk_indices
            tensor[torch.arange(result_dim), topk] = 1

        # auxiliary_weight and auxiliary_embedding_weights.weight stay all-zeros

        sharded_tensor = distribute_tensor(
            tensor,
            meta_param.device_mesh,
            meta_param.placements,
        )
        projections_module.load_state_dict(
            {param_name: torch.nn.Parameter(sharded_tensor)},
            strict=False,
            assign=True,
        )
        del tensor


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
