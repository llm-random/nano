from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
import importlib
import torch
import os
import logging
import sys
from torch.distributed.device_mesh import init_device_mesh

logger = logging.getLogger(__name__)


def get_classes_from_dotted_path(paths):
    return [dynamic_import(path) for path in paths]


def dynamic_import(dotted_path):
    module_path, _, obj_name = dotted_path.rpartition(".")
    if not module_path or not obj_name:
        raise ValueError(f"Invalid path: {dotted_path}")
    module = importlib.import_module(module_path)

    return getattr(module, obj_name)


def setup_fsdp1_model(model, fsdp_config):
    classes_to_wrap = get_classes_from_dotted_path(fsdp_config.modules_to_wrap)
    logger.info(f"[FSDP1] Wrapping model with classes: {classes_to_wrap}")

    ignore_mixed_precision_classes = get_classes_from_dotted_path(
        fsdp_config.mixed_precision.ignored_classes
    )
    logger.info(
        f"[FSDP1] Ignoring mixed precision for classes: {ignore_mixed_precision_classes}"
    )

    mixed_precision_dtype = getattr(
        sys.modules["torch"], fsdp_config.mixed_precision.dtype
    )
    logger.info(f"[FSDP1] Using mixed precision dtype: {mixed_precision_dtype}")

    wrapped_model = FSDP(
        model,
        device_id=int(os.environ["RANK"]),
        mixed_precision=MixedPrecision(
            param_dtype=mixed_precision_dtype,
            cast_forward_inputs=True,
            _module_classes_to_ignore=ignore_mixed_precision_classes,
        ),
        auto_wrap_policy=ModuleWrapPolicy(classes_to_wrap),
        use_orig_params=True,
    )
    return wrapped_model


def setup_fsdp2_model(model, fsdp_config, ignored_params=None):
    modules_to_shard = get_classes_from_dotted_path(fsdp_config.modules_to_shard)
    logger.debug(f"[FSDP2] Sharding model with classes: {modules_to_shard}")
    device_mesh = init_device_mesh("cuda", (torch.distributed.get_world_size(),))

    fsdp2_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    }
    # O29. Optional, defaults to FSDP2's own behaviour so no existing run moves.
    #
    # With the default (True) a sharded module frees its all-gathered parameters
    # at the end of forward and gathers them AGAIN in backward. False keeps them,
    # trading memory for one all-gather per module per micro-step.
    #
    # Whether that is worth it depends on something that changed underneath this
    # code: the 4-GPU profile (job 32843) puts `ncclDevKernel_AllGather` at
    # 219.2 ms/step, 113.2 of it in forward and 106.0 in backward, and the step
    # is only ~177 ms away from being pure compute. While attention was slow the
    # all-gather hid behind it and this flag was worth nothing; now that it does
    # not, the backward half is a candidate.
    #
    # It costs memory, which at h20/micro-batch 16 (63.5 GiB of 80) is the
    # binding constraint -- hence a knob rather than a new default.
    reshard = fsdp_config.get("reshard_after_forward", None)
    if reshard is not None:
        fsdp2_kwargs["reshard_after_forward"] = bool(reshard)
        logger.info(f"[FSDP2] reshard_after_forward={bool(reshard)}")
    if ignored_params:
        # O1. Parameters another sharding scheme already owns. It has to be
        # passed to EVERY fully_shard call that could reach them -- the
        # per-block ones as well as the root -- or the inner call claims them
        # and the root never sees them to ignore. Two schemes claiming the same
        # parameter is not an error FSDP2 reports; it is a wrong gradient.
        fsdp2_kwargs["ignored_params"] = set(ignored_params)
        logger.info(
            f"[FSDP2] ignoring {len(fsdp2_kwargs['ignored_params'])} parameters "
            "owned by another sharding scheme"
        )

    for module in model.modules():
        if isinstance(module, tuple(modules_to_shard)):
            fully_shard(module, mesh=device_mesh, **fsdp2_kwargs)

    fully_shard(model, mesh=device_mesh, **fsdp2_kwargs)
    logger.info(f"Sharding done.")
    return model


def setup_ddp_model(model, ddp_config, ignored_params=None):
    """Replicated backbone, DistributedDataParallel.

    The point of this path is what it does NOT do. FSDP2 all-gathers every
    parameter at the start of each module's forward and again in its backward,
    once per micro-step; DDP keeps them resident and communicates only
    gradients, once per optimizer step when `no_sync` covers the accumulation
    loop. For AME that matters because the parameters worth sharding -- the
    2.576B of Q/K banks -- are not FSDP2's to shard any more: the
    memory-parallel controller owns them and column-shards them itself. What is
    left for FSDP2 is a 587M backbone, small enough to replicate, and sharding
    it buys ~1.8 GB per rank in exchange for a per-micro-step all-gather.

    This is the shape pkmf trained their memory models in, and the shape in
    which memory parallelism has a baseline that is not already sharding
    everything (documents/notes/ame_optimization_plan.md section 3b).

    `ignored_params` is DDP's own: parameters it must not touch, which is
    exactly the bank set. DDP would otherwise register an all-reduce hook on
    2.576B parameters whose `.grad` is permanently None.
    """
    device_id = int(os.environ["LOCAL_RANK"])
    options = ddp_config if isinstance(ddp_config, dict) else dict(ddp_config or {})
    ignored = set(ignored_params or ())
    if ignored:
        names = [n for n, p in model.named_parameters() if p in ignored]
        # The public API takes NAMES, and takes them before construction.
        DDP._set_params_and_buffers_to_ignore_for_model(model, names)
        logger.info(f"[DDP] ignoring {len(names)} parameters owned by another scheme")
    wrapped = DDP(
        model,
        device_ids=[device_id],
        # The banks produce no gradient of their own, and with the backbone
        # alone there is nothing unused left to find; skip the extra traversal.
        find_unused_parameters=bool(options.get("find_unused_parameters", False)),
        gradient_as_bucket_view=bool(options.get("gradient_as_bucket_view", True)),
        bucket_cap_mb=int(options.get("bucket_cap_mb", 25)),
    )
    logger.info("[DDP] backbone replicated")
    return wrapped


def setup_distributed_training(model, distributed_config, ignored_params=None):
    if distributed_config is not None:
        if torch.cuda.is_available():
            if distributed_config.get("ddp"):
                model = setup_ddp_model(
                    model, distributed_config.ddp, ignored_params=ignored_params
                )
            elif distributed_config.get("fsdp2"):
                model = setup_fsdp2_model(
                    model, distributed_config.fsdp2, ignored_params=ignored_params
                )
            elif distributed_config.get("fsdp"):
                if ignored_params:
                    raise ValueError(
                        "memory-parallel banks are only implemented against "
                        "FSDP2; FSDP1 has no ignored_params equivalent here"
                    )
                model = setup_fsdp1_model(model, distributed_config.fsdp)
            else:
                raise ValueError(f"Unknown distributed config.")
        else:
            logger.info(
                "CUDA is not available. Skipping distributed training - using unwrapped model for single-process CPU training."
            )

    return model
