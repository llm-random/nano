"""The repo's config schema: typed `common` and `trainer`, validated at submit time.

Built per documents/notes/typed_config_goal.md. Three jobs:

1. IDE support: entry points annotate `cfg: RootSchema`, so Pylance autocompletes
   and typo-checks `cfg.common.*` / `cfg.trainer.*` in code.
2. Compose-time strictness: `base_schema` is registered in Hydra's ConfigStore on
   import and sits first in every runnable config's defaults list. The stored node
   is strict under `common` and `trainer` only (a mistyped key or wrong-typed value
   there fails on the laptop, not on the cluster) and open everywhere else -- the
   model tree, datasets, infrastructure and ad-hoc `+key=...` additions outside the
   schema'd sections behave exactly as before.
3. Value sanity: `validate(cfg)` holds the range checks the file's original
   pydantic types (PositiveInt and friends) intended. OmegaConf does not run
   `__post_init__` on merged values, so the checks live here, called explicitly
   after composition (main.py) and on every expanded grid point (run_exp.py).

Grid sweeps: `^key: [...]` keys cannot live under a schema'd section of a
schema-backed config -- composition rejects them as unknown keys. A sweep config
using carets under `common`/`trainer` must leave `- base_schema` out of its
defaults list; it loses nothing, because run_exp.py checks every *expanded* grid
point against this schema (`check_point`) before dumping, which is where caret
typos surface anyway.

Policy (accepted in the goal doc): new trainer knobs are declared here, with a
default, BEFORE first use. Once declared, plain `trainer.X=...` works on the
command line; the `+trainer.X=...` spelling is retired for schema'd sections.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf


@dataclass
class AttentionSchema:
    type: str = MISSING
    # Optional per-layer override, e.g. {type: ame, layers: every_4th} -- see
    # resolve_layer_types in src/core/attention.py, which validates its keys.
    mix: Any = None


@dataclass
class CommonSchema:
    """The shared model-shape knobs every config sets (union over all runnable
    configs at 6874efe7). All required: a config that drops one fails the
    submit-time missing-keys check instead of crashing on the cluster."""

    h: int = MISSING
    n_blocks: int = MISSING
    dmodel: int = MISSING
    dff: int = MISSING
    dhead: int = MISSING
    q_heads: int = MISSING
    kv_heads: int = MISSING
    vocab_size: int = MISSING
    sequence_length: int = MISSING
    batch_size: int = MISSING
    attention: AttentionSchema = field(default_factory=AttentionSchema)


@dataclass
class SchedulerSchema:
    _target_: str = MISSING
    _partial_: bool = MISSING
    warmup_steps: int = MISSING
    final_lr_fraction: float = MISSING


@dataclass
class CheckpointSaveSchema:
    type: str = MISSING
    interval: int = MISSING
    path: Optional[str] = MISSING
    model_checkpoint_filename: str = MISSING
    training_state_filename: str = MISSING


@dataclass
class CheckpointLoadSchema:
    type: str = MISSING
    path: Optional[str] = MISSING
    model_checkpoint_filename: str = MISSING
    training_state_filename: Optional[str] = MISSING
    only_weights: bool = MISSING


@dataclass
class CheckpointSchema:
    save: CheckpointSaveSchema = field(default_factory=CheckpointSaveSchema)
    load: CheckpointLoadSchema = field(default_factory=CheckpointLoadSchema)


@dataclass
class DistributedSchema:
    """setup_distributed_training dispatches on exactly these two keys; their
    contents are framework config (FSDP wrapping policy), left untyped."""

    fsdp2: Any = None
    fsdp: Any = None


@dataclass
class TrainerSchema:
    """Mirror of the yaml-supplied attributes of src.core.trainer.Trainer.
    tests/test_typed_config.py keeps the two from drifting. The runtime-injected
    attributes (model, optimizer, training_state, metric_logger) are not config
    keys and are deliberately absent."""

    _target_: str = MISSING
    _partial_: bool = MISSING
    n_steps: Optional[int] = MISSING
    learning_rate: Optional[float] = MISSING
    weight_decay: float = MISSING
    gradient_accumulation_steps: int = MISSING
    gradient_clipping: Optional[float] = MISSING
    eval_interval: int = MISSING
    n_eval_steps: int = MISSING
    scheduler: SchedulerSchema = field(default_factory=SchedulerSchema)
    checkpoint: CheckpointSchema = field(default_factory=CheckpointSchema)
    distributed: Optional[DistributedSchema] = MISSING
    # Dataloaders are dataset-world `_target_` trees (out of scope this round);
    # eval_dataloader is null in the audit configs that switch eval off.
    train_dataloader: Any = MISSING
    eval_dataloader: Any = MISSING
    # Opt-in instrumentation knobs, declared so `trainer.X=true` composes.
    autocast_bf16: bool = False
    profile_timers: bool = False
    kernel_profile: bool = False
    compile_loss: bool = False
    compile_ce: bool = False


@dataclass
class RootSchema:
    """Top level of every runnable config. Only `common` and `trainer` are
    strict; the rest are typed Any so Pylance knows they exist but composition
    stays permissive (the registered node reopens the root -- see register())."""

    common: CommonSchema = field(default_factory=CommonSchema)
    trainer: TrainerSchema = field(default_factory=TrainerSchema)
    model: Any = MISSING
    infrastructure: Any = MISSING
    cluster_switch: Any = MISSING
    # Annotation-only extension hook: run() swaps its initializer when a config
    # provides this (hasattr-guarded). Deliberately absent from the stored
    # schema node (_base_schema_node), so no composed config gains the key --
    # a None reaching run() through the guard would crash the initializer swap.
    init_model_opt_sched_fn: Any = None


def _base_schema_node() -> DictConfig:
    """The node stored as `base_schema`: strict exactly where scoped.

    A plain-dict root holding structured `common`/`trainer` children. Typed
    nodes reject unknown keys through their object_type, with no struct *flag*
    set anywhere -- flags inherit downward, so flag-based strictness on
    `trainer` would also close its untyped subtrees (train_dataloader,
    distributed.fsdp2), which merge freely today and must stay open. With no
    flags, the root stays open for today's idioms (`_run_: true` from
    _cluster/local.yaml, ad-hoc `+compile_model=true`, the model tree), while
    unknown keys under common/trainer still fail composition.
    """
    return OmegaConf.create(
        {
            "common": OmegaConf.structured(CommonSchema),
            "trainer": OmegaConf.structured(TrainerSchema),
        }
    )


def register() -> None:
    ConfigStore.instance().store(name="base_schema", node=_base_schema_node())


# Executed on import: main.py, run_exp.py and conftest.py import this module, so
# the schema is registered before any of them composes a config.
register()


def check_point(point: "dict | DictConfig | RootSchema") -> DictConfig:
    """Merge one expanded grid point against the schema and range-check it.

    run_exp.py calls this on every point create_grid_config produced, before
    dumping -- caret keys are expanded by then, so a `^trainer.n_stepz` typo in
    a sweep fails here, on the laptop, with the offending key in the error.
    """
    if isinstance(point, DictConfig):
        # Merging a node that already carries the schema's object_type skips
        # per-key validation, which would let a `+trainer.bogus=1` force-add
        # through. A plain container is re-validated key by key.
        point = OmegaConf.to_container(point, resolve=False)
    merged = OmegaConf.merge(_base_schema_node(), OmegaConf.create(point))
    validate(merged)
    return merged


def validate(cfg) -> None:
    """Range checks for type-valid but nonsensical values.

    Duck-typed on purpose: the cluster-side run composes the generated flat
    config without the schema, and main.py validates that too.
    """
    problems: List[str] = []

    def check(ok: bool, message: str) -> None:
        if not ok:
            problems.append(message)

    c = cfg.common
    for name in (
        "h",
        "n_blocks",
        "dmodel",
        "dff",
        "dhead",
        "q_heads",
        "kv_heads",
        "vocab_size",
        "sequence_length",
        "batch_size",
    ):
        value = getattr(c, name)
        check(
            value is not None and value > 0, f"common.{name} must be > 0, got {value!r}"
        )
    if c.q_heads and c.kv_heads:
        check(
            c.q_heads % c.kv_heads == 0,
            f"common.q_heads={c.q_heads} must be a multiple of common.kv_heads="
            f"{c.kv_heads} (grouped-query attention repeats kv heads an integer "
            "number of times)",
        )

    t = cfg.trainer
    check(
        t.learning_rate is not None and t.learning_rate > 0,
        f"trainer.learning_rate must be set and > 0, got {t.learning_rate!r} "
        "(the n = 2**-n convention still means a positive number)",
    )
    check(
        t.weight_decay >= 0,
        f"trainer.weight_decay must be >= 0, got {t.weight_decay!r}",
    )
    check(
        t.gradient_accumulation_steps >= 1,
        f"trainer.gradient_accumulation_steps must be >= 1, got {t.gradient_accumulation_steps!r}",
    )
    check(
        t.gradient_clipping is None or t.gradient_clipping > 0,
        f"trainer.gradient_clipping must be null or > 0, got {t.gradient_clipping!r}",
    )
    check(
        t.eval_interval >= 0,
        f"trainer.eval_interval must be >= 0, got {t.eval_interval!r}",
    )
    check(
        t.n_eval_steps >= 0,
        f"trainer.n_eval_steps must be >= 0, got {t.n_eval_steps!r}",
    )

    if t.n_steps is not None:
        # 0 is legitimate: finalize/conversion passes build a scheduler they
        # never step (see the assert in src/core/schedulers.py).
        check(
            t.n_steps >= 0, f"trainer.n_steps must be null or >= 0, got {t.n_steps!r}"
        )
        warmup = t.scheduler.warmup_steps
        check(
            warmup >= 0, f"trainer.scheduler.warmup_steps must be >= 0, got {warmup!r}"
        )
        # Mirrors the assert in get_cosine_scheduler_with_warmup: the cosine
        # phase needs T_max >= 1, i.e. n_steps > warmup_steps + 1.
        check(
            t.n_steps == 0 or t.n_steps > warmup + 1,
            f"trainer.n_steps={t.n_steps} must exceed scheduler.warmup_steps+1="
            f"{warmup + 1}, or the scheduler divides by zero on step 1",
        )

    save_interval = t.checkpoint.save.interval
    check(
        save_interval >= -1,
        f"trainer.checkpoint.save.interval must be >= -1 (-1 means final-only), got {save_interval!r}",
    )

    if problems:
        raise ValueError(
            "config failed validation (src/definitions.py validate):\n  "
            + "\n  ".join(problems)
        )
