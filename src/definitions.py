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
from omegaconf import MISSING, DictConfig, OmegaConf, open_dict


# trainer.bank_lr_ratio when a config leaves it out (a sweep config composed
# without base_schema). TrainerSchema and src.core.trainer.Trainer carry the
# same literal; tests/test_typed_config.py holds the pair together.
DEFAULT_BANK_LR_RATIO = 4.0


@dataclass
class AttentionSchema:
    type: str = MISSING
    # Optional per-layer override, e.g. {type: ame, layers: every_4th} -- see
    # resolve_layer_types in src/core/attention.py, which validates its keys.
    mix: Any = None
    # Optional per-type options block, applied wherever that type is built.
    # Left untyped: build_attention checks the keys against the type's own
    # option list and names the offender, which a dataclass per type would
    # duplicate. One field per type that has options.
    ame: Any = None
    # {rope: false} is NoPE; {qk_norm: true} adds AME's QK-norm
    # (_GQA_OPTIONS in src/core/attention.py)
    gqa: Any = None
    # AME's QK-norm on every layer of every type: null keeps each type's
    # default (mha/mqa/gqa off, ame on), true/false sets it everywhere. A
    # type's own option of the same name may repeat it but not contradict it
    # (build error).
    qk_norm: Optional[bool] = None


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
    """setup_distributed_training dispatches on exactly these keys; their
    contents are framework config (FSDP wrapping policy, DDP bucket options),
    left untyped.

    `ddp` is checked first: it replicates the backbone instead of sharding it,
    which is the right trade once src.core.memory_parallel owns the 2.576B of
    Q/K banks and all that is left to shard is a 587M backbone. A config that
    sets it should null `fsdp2` too, so the file says what it means.
    """

    fsdp2: Any = None
    fsdp: Any = None
    ddp: Any = None


@dataclass
class TrainerSchema:
    """Mirror of the yaml-supplied attributes of src.core.trainer.Trainer.
    tests/test_typed_config.py keeps the two from drifting. The runtime-injected
    attributes (model, optimizer, training_state, metric_logger) are not config
    keys and are deliberately absent."""

    _target_: str = MISSING
    _partial_: bool = MISSING
    n_steps: Optional[int] = MISSING
    # Training length in BILLIONS of tokens, the alternative to n_steps: set
    # one, leave the other null. resolve_n_steps() turns it into whole steps
    # of common.batch_size * common.sequence_length tokens, rounded down,
    # before anything reads n_steps.
    n_tokens_b: Optional[float] = None
    learning_rate: Optional[float] = MISSING
    # The AME memory banks (q_bank/k_bank) train at this multiple of
    # learning_rate, in their own param group or in the memory-parallel
    # controller's optimizer, for the whole schedule. No effect on a model
    # without banks; 1.0 is the old single-lr behavior.
    bank_lr_ratio: float = 4.0  # == DEFAULT_BANK_LR_RATIO
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
    # O11, fused-linear-CE. `fused_ce_chunk_size` is in rows
    # (micro_batch * seq flattened); None means one chunk, which is the memory
    # of the unfused path.
    fused_linear_ce: bool = False
    fused_ce_chunk_size: Optional[int] = None
    # O9: reduce gradients once per optimizer step instead of once per
    # micro-step. FSDP2 only, and it costs memory -- see the long note on
    # Trainer.skip_grad_sync_during_accumulation before enabling it.
    skip_grad_sync_during_accumulation: bool = False
    # Tiers 2+3: ranks per copy of the AME Q/K banks. null/1 keeps the whole
    # bank on every rank (Tier 2 alone, which does not fit at h20); N > 1
    # column-shards every bank across contiguous blocks of N ranks. Must divide
    # world_size, and the model must have q_bank/k_bank embeddings.
    mem_parallel_size: Optional[int] = None


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
    # On the merged copy: the caller's point stays as written, n_tokens_b and
    # all, which is what run_exp.py dumps and the cluster-side run resolves
    # again (main.run). Before validate, whose n_steps checks need the steps.
    resolve_n_steps(merged)
    validate(merged)
    return merged


def steps_for_tokens(n_tokens_b: float, batch_size: int, sequence_length: int) -> int:
    """Whole optimizer steps in `n_tokens_b` billion tokens, rounded down.

    One step is `batch_size * sequence_length` tokens: common.batch_size is the
    GLOBAL batch (get_dataloader hands each rank batch_size // world_size), so
    the count does not depend on the GPU count or gradient accumulation. These
    are corpus tokens, the convention of the hand-written `n_steps: 7630 # 8B
    tokens / 1,048,576 tokens per step` configs. The trainer's processed_tokens
    (the W&B token axis) counts predicted tokens instead, sequence_length - 1
    per sequence, so it ends 1/sequence_length below the budget (0.05% at 2048).

    The token count is rounded to a whole number first, so a budget that is an
    exact multiple of the step survives float noise: 1.001 * 10**9 is
    1000999999.9999999 in floating point, and flooring that would lose a step.
    """
    tokens_per_step = batch_size * sequence_length
    return round(n_tokens_b * 10**9) // tokens_per_step


def resolve_n_steps(cfg) -> None:
    """Fold `trainer.n_tokens_b` into `trainer.n_steps`, in place.

    The training length is given in steps or in billions of tokens, never
    both. A config that sets n_tokens_b must leave n_steps null; this fills it
    in with steps_for_tokens(), so everything downstream -- the scheduler, the
    warmup interpolation `${trainer.n_steps}`, checkpoint names, the training
    loop -- keeps reading steps and nothing else has to know about tokens.

    Runs in two places: on check_point's merged copy (run_exp.py, every grid
    point, before anything is submitted; main.py at startup) and in place in
    main.run, before anything reads n_steps. Once per config object: after it,
    both keys are set, which is exactly what it rejects on input.
    """
    t = cfg.trainer
    n_tokens_b = t.get("n_tokens_b")
    if n_tokens_b is None:
        return
    if t.get("n_steps") is not None:
        raise ValueError(
            f"trainer.n_tokens_b={n_tokens_b} and trainer.n_steps={t.n_steps} are "
            "both set: the training length is given one way or the other. Leave "
            "n_steps empty when giving tokens (`n_steps: null` in yaml, "
            "`trainer.n_steps=null` on the command line)."
        )
    if n_tokens_b <= 0:
        raise ValueError(
            f"trainer.n_tokens_b must be > 0 (billions of tokens), got {n_tokens_b!r}"
        )
    batch_size, sequence_length = cfg.common.batch_size, cfg.common.sequence_length
    if not (batch_size and sequence_length and batch_size > 0 and sequence_length > 0):
        raise ValueError(
            "trainer.n_tokens_b needs common.batch_size and common.sequence_length "
            f"> 0 to count steps, got {batch_size!r} and {sequence_length!r}"
        )
    n_steps = steps_for_tokens(n_tokens_b, batch_size, sequence_length)
    if n_steps < 1:
        raise ValueError(
            f"trainer.n_tokens_b={n_tokens_b} is less than one optimizer step of "
            f"common.batch_size * common.sequence_length = "
            f"{batch_size * sequence_length:,} tokens"
        )
    with open_dict(t):
        t.n_steps = n_steps


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
    for name in ("h", "n_blocks", "dmodel", "dff", "dhead", "q_heads", "kv_heads",
                 "vocab_size", "sequence_length", "batch_size"):
        value = getattr(c, name)
        check(value is not None and value > 0, f"common.{name} must be > 0, got {value!r}")
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
    check(t.weight_decay >= 0, f"trainer.weight_decay must be >= 0, got {t.weight_decay!r}")
    bank_lr_ratio = getattr(t, "bank_lr_ratio", DEFAULT_BANK_LR_RATIO)
    check(
        bank_lr_ratio is not None and bank_lr_ratio > 0,
        f"trainer.bank_lr_ratio must be > 0, got {bank_lr_ratio!r}",
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
        t.mem_parallel_size is None or t.mem_parallel_size >= 1,
        f"trainer.mem_parallel_size must be null or >= 1, got {t.mem_parallel_size!r}",
    )
    check(
        t.fused_ce_chunk_size is None or t.fused_ce_chunk_size >= 1,
        f"trainer.fused_ce_chunk_size must be null or >= 1, got "
        f"{t.fused_ce_chunk_size!r}",
    )
    check(
        not (t.fused_linear_ce and t.compile_ce),
        "trainer.fused_linear_ce and trainer.compile_ce both set: compile_ce "
        "compiles a cross-entropy over logits that fused_linear_ce never "
        "builds, so the two cannot both apply.",
    )
    check(t.eval_interval >= 0, f"trainer.eval_interval must be >= 0, got {t.eval_interval!r}")
    check(t.n_eval_steps >= 0, f"trainer.n_eval_steps must be >= 0, got {t.n_eval_steps!r}")

    if t.n_steps is not None:
        # 0 is legitimate: finalize/conversion passes build a scheduler they
        # never step (see the assert in src/core/schedulers.py).
        check(t.n_steps >= 0, f"trainer.n_steps must be null or >= 0, got {t.n_steps!r}")
        warmup = t.scheduler.warmup_steps
        check(warmup >= 0, f"trainer.scheduler.warmup_steps must be >= 0, got {warmup!r}")
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
