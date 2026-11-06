import os
import hydra
import yaml
from src.core.utils import solve_config_lr
from src.core.distributed_training import setup_distributed_training
import resolver as _  # noqa: F401  side-effect import: registers OmegaConf resolvers
import logging
from typing import cast
from omegaconf import DictConfig

import torch
import torch.distributed as dist
from hydra.utils import instantiate
from src.core.checkpointing import (
    load_checkpoint_from_file,
    load_training_state,
    get_full_checkpoint_path,
)
from src.core.memory_parallel import (
    MemoryParallelController,
    get_controller,
)
from src.core.metric_loggers import WandbLogger, get_metric_logger
from src.core.model import Residual, parameter_groups

# Importing src.definitions registers `base_schema` in Hydra's ConfigStore, which
# every runnable config's defaults list references -- it must happen before compose.
from src.definitions import RootSchema, check_point, resolve_n_steps
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


CHECKPOINTING_DISABLED = "<checkpointing disabled>"


def _build_save_path(cfg):
    if cfg.trainer.checkpoint.save.path is None:
        # Checkpointing is off -- _should_save_final_checkpoint guards on exactly this,
        # so nothing is written. Interpolating a path anyway produced
        # "None/<jobid>/<task>/step_N" in the run record: a directory that does not
        # exist and that someone will eventually go looking for.
        return CHECKPOINTING_DISABLED
    base = get_full_checkpoint_path(cfg.trainer.checkpoint.save.path)
    n_steps = cfg.trainer.n_steps
    if n_steps is not None:
        return f"{base}/step_{n_steps - 1}"
    return base


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


def check_env_vars():
    assert int(os.environ["RANK"]) < int(os.environ["WORLD_SIZE"])


def isolate_triton_cache():
    """Give every rank its own Triton compile cache, on node-local storage.

    torch.compile hands each rank's generated kernels to Triton, which caches
    them under TRITON_CACHE_DIR -- defaulting to `$HOME/.triton/cache`. On a
    cluster that default is wrong twice over, and it fails *during compilation*,
    before step 1, so a job burns its allocation and produces nothing.

    Every rank writes the same cache keys at the same moment, and Triton
    publishes each entry with `os.replace(temp, final)`. That is atomic on a
    local filesystem and is not on NFS, so ranks tear each other's entries
    apart. Observed on lem, 2026-11-04, jobs 6236712 and 6236713 -- three
    different symptoms of the one race, none of which names its cause:

        FileNotFoundError [Errno 2]   another rank's rename removed the temp dir
        OSError [Errno 26]            "Text file busy" -- replacing a .so that a
                                      rank already has mapped
        ImportError ... Permission denied
                                      opening a half-written .so

    Worse, `$HOME` is shared across *jobs*, so two concurrent runs race as well,
    which is exactly how those two jobs died on separate nodes.

    So: node-local (TMPDIR, which Slurm points at per-job local disk; /tmp
    otherwise) and per-rank. Set before anything imports or invokes the
    compiler. Honours an explicit TRITON_CACHE_DIR if the operator set one.
    """
    if os.environ.get("TRITON_CACHE_DIR"):
        return
    base = os.environ.get("TMPDIR") or "/tmp"
    job = os.environ.get("SLURM_JOB_ID", "local")
    rank = os.environ.get("RANK", "0")
    cache_dir = os.path.join(base, f"triton-cache-{job}-rank{rank}")
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError as exc:
        # Never fail a run over a cache directory; the default path may still
        # work on a single-rank job, which is the only case it is safe in.
        logger.warning(
            f"Could not create {cache_dir} ({exc}); leaving TRITON_CACHE_DIR unset."
        )
        return
    os.environ["TRITON_CACHE_DIR"] = cache_dir
    logger.info(f"TRITON_CACHE_DIR={cache_dir}")


def setup_enviroment():
    if "WORLD_SIZE" not in os.environ:
        logger.warning("WORLD_SIZE is not set, setting it to 1")
        os.environ["WORLD_SIZE"] = "1"

    if "LOCAL_WORLD_SIZE" not in os.environ:
        logger.warning("LOCAL_WORLD_SIZE is not set, setting it to 1")
        os.environ["LOCAL_WORLD_SIZE"] = "1"

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
    # After RANK is known and before any compilation happens.
    isolate_triton_cache()


def distributed_setup():
    if not torch.cuda.is_available():
        logger.warning(
            "CUDA is not available. Skipping distributed setup - running single-process training on CPU."
        )
        return

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    torch.cuda.set_device(local_rank)


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device


def build_memory_parallel_controller(cfg, model):
    """Tiers 2+3. None unless `trainer.mem_parallel_size` asks for it.

    The compute dtype has to match what the rest of the stack will run in, and
    nothing downstream casts an embedding: FSDP2's MixedPrecisionPolicy casts
    the parameters it owns and `torch.autocast` casts nn.Linear, but an
    nn.Embedding gather returns the bank's own dtype either way. So under FSDP2
    the banks are held in bf16 (what the policy would have produced) and
    everywhere else in whatever dtype the model was built in -- which keeps the
    single-GPU and CPU paths bit-comparable against a run without this flag.
    """
    size = cfg.trainer.get("mem_parallel_size", None)
    if not size:
        return None

    distributed = cfg.trainer.get("distributed", None)
    # bf16 whenever the surrounding stack computes in bf16, by either route:
    # FSDP2's MixedPrecisionPolicy, or torch.autocast under DDP / single GPU.
    # It matters because nothing downstream casts an embedding -- autocast casts
    # nn.Linear and leaves a gather alone -- so an fp32 bank under a bf16 stack
    # gathers twice the bytes and then gets cast anyway.
    bf16_stack = torch.cuda.is_available() and (
        bool(distributed is not None and distributed.get("fsdp2"))
        or bool(cfg.trainer.get("autocast_bf16", False))
    )
    reference = next(model.parameters())
    compute_dtype = torch.bfloat16 if bf16_stack else reference.dtype
    controller = MemoryParallelController(
        model, mem_parallel_size=size, compute_dtype=compute_dtype
    )
    logger.info(
        f"Memory-parallel banks enabled: mem_parallel_size={size}, "
        f"compute dtype {compute_dtype}"
    )
    return controller


def get_model_optimizer_scheduler(cfg, model, learning_rate):
    if cfg.get("apply_functions", None):
        for fn in instantiate(cfg.apply_functions):
            res = fn(model)
            if res == False:
                logger.info("Initialization failed, exiting...")
                return None, None, None
    # Before every compile and before fully_shard: this swaps each q_bank /
    # k_bank nn.Embedding for a ShardedBank, and a module swapped in after a
    # block was compiled would be traced on the next call anyway -- but the
    # ordering that is actually load-bearing is against fully_shard, which must
    # see the parameters it is being told to ignore.
    banks = build_memory_parallel_controller(cfg, model)

    if cfg.get("compile_blocks", False):
        # Regional compile, applied before fully_shard wraps the blocks -- the
        # pairing FSDP2 supports. The compiled unit is each block's attention
        # and feed-forward module, NOT the whole TransformerBlock: Residual's
        # forward feeds its per-block log_name string to the metric logger, so
        # a block-level compile guards on that string, needs one specialization
        # per block, and dynamo's recompile_limit (8) silently dropped half an
        # h16 to eager (measured: compile gained -0.2%). The inner modules are
        # string-free, so one graph serves every block. Top-level key
        # (+compile_blocks=true) rather than a trainer key, because every
        # cfg.trainer entry is passed to Trainer's constructor.
        #
        # Kept as-is so the runs recorded in documents/notes/ reproduce from
        # their logged commands; compile_blocks_full below is the wider unit
        # that the log_name fix made possible.
        for block in model.encoder.blocks:
            block.attention_layer.layer.compile()
            block.ff_layer.layer.compile()
        logger.info(
            f"torch.compile applied to attention+ff of {len(model.encoder.blocks)} blocks"
        )
    if cfg.get("compile_blocks_full", False):
        # Same regional strategy, one level wider: the compiled unit is the
        # whole TransformerBlock, so each block's two Residual wrappers -- the
        # RMSNorm, the `out + x` add and the bf16<->fp32 boundary casts -- are
        # traced with the attention and ff they surround instead of running
        # eager between two compiled regions.
        #
        # That glue is where the time is. Under compile_blocks the dominant
        # eager kernels fire 51 times per micro-step at h20 (25 blocks x 2
        # Residual wrappers + the head norm), measured at 3.7 s/step of
        # elementwise plus 0.5 s of norm pieces on one GPU
        # (documents/notes/multigpu_efficiency.md).
        #
        # The reason compile_blocks stops short no longer applies: Residual's
        # per-block log_name used to be traced and guarded on, forcing one
        # recompile per block until recompile_limit silently dropped blocks to
        # eager, and that logging now sits behind @torch.compiler.disable.
        # The compiled unit is also exactly the module FSDP2 shards, which is
        # the pairing FSDP2 documents. Watch for recompile warnings anyway --
        # this failure mode is silent (documents/notes/multigpu_compile_scope_plan.md).
        for block in model.encoder.blocks:
            block.compile()
        logger.info(
            f"torch.compile applied to {len(model.encoder.blocks)} whole blocks"
        )
    if cfg.get("compile_model", False):
        # Whole-model compile, nanoGPT-style: one inductor graph containing the
        # blocks AND the glue between them (norms, residual adds, autocast
        # casts), which regional compile leaves eager -- the kernel profile
        # measured that glue at ~2.9 s/step at h20. Residual's metric logging
        # is confined to a compiler-disabled helper, so its per-block log_name
        # string never enters tracing (the trap that killed block-level
        # compile). Mutually exclusive with compile_blocks by intent.
        model.compile()
        logger.info("torch.compile applied to the whole model")
    ignored = banks.ignored_parameters() if banks is not None else None
    model = setup_distributed_training(model, cfg.trainer.distributed, ignored)
    optimizer = torch.optim.AdamW(
        # Groups rather than a flat parameters(): a layer may declare scales
        # that decoupled decay must not touch (src.core.model.parameter_groups).
        # A model that declares none yields one group holding exactly
        # model.parameters(), so this is a no-op for every existing config.
        parameter_groups(model, cfg.trainer.weight_decay, exclude=ignored),
        lr=learning_rate,
        # The fused kernel exists only for CUDA tensors; CPU runs keep the
        # default path, which also keeps the seeded CPU gates bit-identical.
        fused=torch.cuda.is_available(),
    )
    if banks is not None:
        # wd=0 by construction: AMEAttention names both banks in
        # no_weight_decay_parameters, so the group they would have landed in
        # above is the wd=0 one. `fused` is deliberately NOT passed -- this is a
        # single step over up to 2.576B parameters and the foreach/fused paths
        # are the ones pkmf hit an OOM on. Measure before changing it.
        banks.build_optimizer(lr=learning_rate, weight_decay=0.0)
    scheduler = instantiate(cfg.trainer.scheduler)(
        optimizer=optimizer, n_steps=cfg.trainer.n_steps
    )

    return model, optimizer, scheduler


def initialize_training_components(cfg: RootSchema, metric_logger=None):
    training_state = load_training_state(cfg.trainer.checkpoint.load)

    if metric_logger is None:
        metric_logger = get_metric_logger(
            metric_logger_config=cfg.infrastructure.metric_logger,
            tracker_run_id=training_state["run_id"],
            # cast: the runtime object is a DictConfig; RootSchema is Pylance sugar.
            full_config=cast(DictConfig, cfg),
        )

    # cast: validate() has already rejected a null/negative learning_rate.
    learning_rate, exp_lr = solve_config_lr(cast(float, cfg.trainer.learning_rate))

    if isinstance(metric_logger, WandbLogger) and (
        training_state["run_id"] is None
        or cfg.infrastructure.metric_logger.new_wandb_job
    ):
        # config, not log: every one of these is a run *input*, fixed before step 0
        # and never updated again. run.log() files a value in the history, where
        # wandb can only render it as a one-point chart and the runs table cannot
        # group, filter or sort on it. config.update() puts them beside common.* and
        # trainer.*, which is where a hyperparameter belongs.
        if metric_logger.run is not None:
            metric_logger.run.config.update(
                {
                    "learning_rate": learning_rate,
                    "exp_lr": exp_lr,
                    "full_save_checkpoints_path": _build_save_path(cfg),
                    # The training data is sharded per rank, so a run is reproducible
                    # at a fixed (seed, world_size) and not across world sizes -- two
                    # runs of the same config on different GPU counts train on
                    # different data. The seed is already in the config; this puts the
                    # other half of the pair in the run record beside it.
                    "world_size": int(os.environ.get("WORLD_SIZE", 1)),
                },
                # a resumed run may already carry these keys
                allow_val_change=True,
            )

    torch.manual_seed(cfg.trainer.train_dataloader.dataset.seed)

    device = get_device()

    logger.info(f"Creating model...")
    model = instantiate(cfg.model, _convert_="all").to(device)
    logger.info(
        f"Model {model.__class__.__name__} created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )
    # Residual layers needs metric_logger for logging update norms
    for _, module in model.named_modules():
        if isinstance(module, Residual):
            module.set_metric_logger(metric_logger)

    if cfg.trainer.checkpoint.load.type == "nano":
        # TODO! if you want to apply function on loaded model it does NOT work now, it applies function on newly inintialized model than it loads model weights
        model, optimizer, scheduler = get_model_optimizer_scheduler(
            cfg, model, learning_rate
        )
        load_checkpoint_from_file(
            cfg.trainer.checkpoint.load, model, optimizer, scheduler
        )
        if cfg.trainer.checkpoint.load.only_weights:
            optimizer = torch.optim.AdamW(
                # Same grouping as the fresh-start path above. This branch
                # rebuilds the optimizer from scratch after a weights-only
                # load, so without it a resume would silently put decay back
                # on the parameters the model asked to keep out of it -- or,
                # with a memory-parallel controller in play, put 2.576B of bank
                # parameters into two optimizers at once.
                parameter_groups(
                    model,
                    cfg.trainer.weight_decay,
                    exclude=(
                        get_controller(model).ignored_parameters()
                        if get_controller(model) is not None
                        else None
                    ),
                ),
                lr=learning_rate,
                fused=torch.cuda.is_available(),
            )
            scheduler = instantiate(cfg.trainer.scheduler)(
                optimizer=optimizer, n_steps=cfg.trainer.n_steps
            )
    else:
        raise Exception(
            f"Not recognized load checkpoint format: {cfg.trainer.checkpoint.load.type}"
        )

    return model, optimizer, scheduler, training_state, metric_logger


def run(cfg: RootSchema, metric_logger=None):
    # First, before anything reads n_steps: a length given in tokens becomes
    # steps here, on the cluster too -- the generated configs keep n_tokens_b
    # as written, and run_exp.py has already checked that it resolves.
    resolve_n_steps(cfg)
    if cfg.trainer.get("n_tokens_b") is not None:
        tokens_per_step = cfg.common.batch_size * cfg.common.sequence_length
        logger.info(
            f"trainer.n_tokens_b={cfg.trainer.n_tokens_b} -> trainer.n_steps="
            f"{cfg.trainer.n_steps} ({cfg.trainer.n_steps * tokens_per_step:,} "
            f"tokens at {tokens_per_step:,} per step, rounded down)"
        )
    setup_enviroment()

    # The runtime object is a DictConfig (the RootSchema annotation is Pylance
    # sugar); `in` also guards configs that predate the schema'd key.
    trainer_node = cast(DictConfig, cfg.trainer)
    if "distributed" in trainer_node and trainer_node.distributed is not None:
        distributed_setup()

    initialize_fn = (
        instantiate(cfg.init_model_opt_sched_fn, _convert_="all")
        if hasattr(cfg, "init_model_opt_sched_fn")
        else initialize_training_components
    )

    model, optimizer, scheduler, training_state, metric_logger = initialize_fn(
        cfg, metric_logger
    )

    if model is not None:
        logger.info(f"Model initialized")

        trainer = instantiate(cfg.trainer)

        trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
            metric_logger=metric_logger,
        ).train()

        # TODO
        # finetuning

    cleanup()


@hydra.main(version_base=None, config_path="configs", config_name="exp")
def main(config: RootSchema):
    # Re-checks keys/types against the schema (Hydra's `+key=` force-add slips
    # past struct validation at compose) and runs validate()'s range checks.
    check_point(config)
    run(config)


if __name__ == "__main__":
    main()
