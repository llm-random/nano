"""Tier 1 of the baseline audit: CPU unit tests for the framework itself.

See documents/notes/main_baseline_audit.md. These cover src/core and the grid
generator -- the parts a new, non-PC project inherits wholesale from main -- so they
matter most for main as a branch point. No GPU, no data, no cluster.
"""

import itertools
import math

import pytest
import torch
from omegaconf import OmegaConf

from grid_generator.generate_configs import create_grid_config
from src.core.schedulers import get_cosine_scheduler_with_warmup
from src.core.utils import solve_config_lr


# --------------------------------------------------------------------------------------
# the learning-rate convention


@pytest.mark.parametrize("exponent", [5, 10, 11, 12, 13, 20])
def test_integer_learning_rate_means_two_to_the_minus_n(exponent):
    """`learning_rate: 13` in a config means 2**-13. Everything is tuned on this."""
    lr, exp_lr = solve_config_lr(exponent)
    assert lr == 2**-exponent
    assert exp_lr == exponent, (
        "the exponent must be reported, not dropped: it is what identifies a run's LR "
        "in wandb, and run provenance depends on it"
    )


def test_fractional_learning_rate_is_passed_through_with_its_exponent():
    lr, exp_lr = solve_config_lr(2**-13)
    assert lr == 2**-13
    assert exp_lr == pytest.approx(13.0)

    lr, exp_lr = solve_config_lr(5e-05)
    assert lr == 5e-05
    assert exp_lr == pytest.approx(-math.log2(5e-05))


@pytest.mark.parametrize("exponent", [5, 11, 13])
def test_the_two_learning_rate_spellings_agree(exponent):
    """Writing the exponent and writing the rate itself must describe the same run.

    These are two branches of one function and used to disagree: the integer branch
    returned exp_lr=None, so a run configured the normal way logged no exponent at all.
    """
    by_exponent = solve_config_lr(exponent)
    by_value = solve_config_lr(2**-exponent)
    assert by_exponent[0] == by_value[0]
    assert by_exponent[1] == pytest.approx(by_value[1])


# --------------------------------------------------------------------------------------
# scheduler


def _scheduler(n_steps, warmup_steps, base_lr=1.0, final_lr_fraction=0.1):
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=base_lr)
    scheduler = get_cosine_scheduler_with_warmup(
        optimizer,
        warmup_steps=warmup_steps,
        n_steps=n_steps,
        final_lr_fraction=final_lr_fraction,
    )
    return optimizer, scheduler


def test_cosine_schedule_warms_up_then_decays_to_the_floor():
    n_steps, warmup = 100, 10
    optimizer, scheduler = _scheduler(
        n_steps, warmup, base_lr=1.0, final_lr_fraction=0.1
    )

    seen = []
    for _ in range(n_steps):
        seen.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert seen[0] < seen[warmup // 2] < seen[warmup], "warmup is not increasing"
    assert seen[warmup] == pytest.approx(1.0, rel=1e-6), "peak is not the base LR"
    tail = seen[warmup + 1 :]
    assert all(
        later <= earlier + 1e-12 for earlier, later in zip(tail, tail[1:])
    ), "the cosine phase is not monotonically decreasing"
    assert seen[-1] == pytest.approx(
        0.1, abs=5e-3
    ), "did not decay to final_lr_fraction"
    assert all(lr > 0 for lr in seen), "learning rate went non-positive"


def test_scheduler_rejects_an_n_steps_that_leaves_no_cosine_phase():
    """T_max=0 would otherwise surface as a bare ZeroDivisionError on the first step.

    Easy to hit when shrinking n_steps for a smoke test and forgetting warmup_steps.
    """
    with pytest.raises(AssertionError, match="warmup_steps"):
        _scheduler(n_steps=10, warmup_steps=10)


def test_zero_steps_is_allowed():
    """The conversion and importances configs build a scheduler they never step."""
    _scheduler(n_steps=0, warmup_steps=10)


# --------------------------------------------------------------------------------------
# grid expansion -- the run_exp.py / main.py contract


def test_caret_key_expands_to_one_config_per_value():
    """run_exp.py expands `^`-prefixed keys into a grid; main.py cannot consume them."""
    cfg = OmegaConf.create({"trainer": {"^n_steps": [10, 20, 30], "lr": 13}})
    grid = create_grid_config(cfg)
    assert len(grid) == 3
    assert sorted(point["trainer"]["n_steps"] for point, _ in grid) == [10, 20, 30]
    for point, _ in grid:
        assert "^n_steps" not in point["trainer"], "the caret key survived expansion"
        assert point["trainer"]["lr"] == 13, "a plain key was lost during expansion"


def test_two_caret_keys_give_the_cartesian_product():
    cfg = OmegaConf.create({"a": {"^x": [1, 2]}, "b": {"^y": [10, 20, 30]}})
    grid = create_grid_config(cfg)
    assert len(grid) == 6
    combinations = {(point["a"]["x"], point["b"]["y"]) for point, _ in grid}
    assert combinations == {(x, y) for x in (1, 2) for y in (10, 20, 30)}


def test_config_without_caret_keys_is_a_single_point():
    cfg = OmegaConf.create({"trainer": {"n_steps": 10, "lr": 13}})
    grid = create_grid_config(cfg)
    assert len(grid) == 1
    assert grid[0][0]["trainer"]["n_steps"] == 10


def test_caret_key_whose_value_is_not_a_list_is_rejected():
    """A `^` key holding a scalar is a typo, and silently running one config would
    quietly drop the rest of an intended sweep."""
    cfg = OmegaConf.create({"trainer": {"^n_steps": 10}})
    with pytest.raises(Exception, match="not a list"):
        create_grid_config(cfg)


# --------------------------------------------------------------------------------------
# dataloader determinism
#
# These need the local dummy dataset, which is gitignored, so they are skipped where it
# is absent. They matter more than they look: the resume bug fixed in this branch was a
# data-ordering bug, and the property that makes the fix correct -- that the stream is a
# deterministic function of the seed -- is exactly what is asserted here.

import os  # noqa: E402

import resolver  # noqa: F401,E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

needs_local_data = pytest.mark.skipif(
    not (
        os.path.isdir(os.path.join(REPO_ROOT, "data"))
        and os.path.isdir(os.path.join(REPO_ROOT, "data_eval"))
    ),
    reason="local dummy dataset not present (data/ and data_eval/ are gitignored)",
)


def _train_batches(n, seed=None):
    for key, value in dict(
        WORLD_SIZE="1", LOCAL_WORLD_SIZE="1", RANK="0", LOCAL_RANK="0"
    ).items():
        os.environ.setdefault(key, value)
    overrides = ["infrastructure.metric_logger.type=stdout"]
    if seed is not None:
        overrides.append(f"trainer.train_dataloader.dataset.seed={seed}")
    with initialize_config_dir(
        version_base=None, config_dir=os.path.join(REPO_ROOT, "configs")
    ):
        cfg = compose(config_name="tiny_local", overrides=overrides)
    loader = instantiate(cfg.trainer.train_dataloader, _convert_="all")
    out = []
    for i, batch in enumerate(loader):
        if i >= n:
            break
        out.append(batch.clone())
    return out


@needs_local_data
def test_the_batch_stream_is_a_function_of_the_seed():
    """Two loaders built with the same seed must yield the same batches."""
    first = _train_batches(4, seed=123)
    second = _train_batches(4, seed=123)
    assert len(first) == len(second) == 4
    for i, (a, b) in enumerate(zip(first, second)):
        assert torch.equal(
            a, b
        ), f"batch {i} differs between two identically seeded loaders"


@needs_local_data
def test_a_different_seed_gives_a_different_stream():
    """Otherwise the seed is not doing anything and 'deterministic by seed' is vacuous."""
    assert not all(
        torch.equal(a, b)
        for a, b in zip(_train_batches(4, seed=123), _train_batches(4, seed=999))
    ), "changing the seed did not change the batches"


@needs_local_data
def test_restarting_a_loader_replays_from_the_beginning():
    """The property the resume fix is built on.

    AbstractDataset.__iter__ reseeds, so a fresh iterator starts the stream over. That
    is why Trainer._batches_from_start_step has to skip start_step batches by hand: if
    this ever became false -- a stateful or sharded loader, say -- skipping would be
    wrong and resume would silently corrupt the data order again.
    """
    assert torch.equal(_train_batches(1, seed=123)[0], _train_batches(1, seed=123)[0])


# ---------------------------------------------------------------------------
# Rank sharding.
#
# A 1-GPU and a 2-GPU run of the same config do not see the same data, and that
# surprised me enough to chase it: a 2-GPU parity smoketest diverged from the
# 1-GPU run by up to 0.75 in loss. The mechanism is in _load_dataset. With
# world_size_independent=False -- what every shipped config sets -- the HF
# dataset is split by split_dataset_by_node at load time, so each rank reads a
# different third of the 64 shards and AbstractDataset.__iter__ hands back the
# packer unsliced. With True, every rank loads the whole dataset and __iter__
# strides it by rank instead. Both shard; they just shard at different layers.
#
# So cross-world-size loss equality is not a property this framework has, and a
# test asserting it would be wrong. What must hold is that ranks never train on
# the same samples -- silent duplication would quietly scale the effective batch
# down and is exactly the kind of bug that hides behind a plausible loss curve.
# That is what these two assert, one per branch.


def _rank_stream(n, world_size, rank, world_size_independent, monkeypatch):
    from src.core.datasets import C4Dataset, gpt2_tokenize_fn

    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    monkeypatch.setenv("RANK", str(rank))
    dataset = C4Dataset(
        sequence_length=16,
        tokenize_fn=gpt2_tokenize_fn(),
        path=os.path.join(REPO_ROOT, "data"),
        split="train",
        seed=123,
        shuffle=True,
        world_size_independent=world_size_independent,
    )
    return [tuple(sequence) for sequence in itertools.islice(iter(dataset), n)]


@needs_local_data
@pytest.mark.parametrize("world_size_independent", [False, True])
def test_ranks_never_see_the_same_sample(world_size_independent, monkeypatch):
    """Two ranks of one run must train on disjoint data, by either sharding route."""
    first = _rank_stream(12, 2, 0, world_size_independent, monkeypatch)
    second = _rank_stream(12, 2, 1, world_size_independent, monkeypatch)
    assert first, "rank 0 produced no samples"
    shared = set(first) & set(second)
    assert not shared, (
        f"{len(shared)} of 12 sequences are on both ranks with "
        f"world_size_independent={world_size_independent}; ranks are duplicating data"
    )


@needs_local_data
def test_world_size_independent_partitions_one_stream(monkeypatch):
    """The True branch is a stride, so the ranks interleave back into the 1-GPU stream."""
    whole = _rank_stream(8, 1, 0, True, monkeypatch)
    first = _rank_stream(4, 2, 0, True, monkeypatch)
    second = _rank_stream(4, 2, 1, True, monkeypatch)
    interleaved = [sequence for pair in zip(first, second) for sequence in pair]
    assert (
        interleaved == whole
    ), "world_size_independent=True did not partition the single-rank stream"


# ---------------------------------------------------------------------------
# Checkpoint logging.
#
# save_training_state writes to step_<n>/ but used to log save_config.path, the
# parent -- a path with no file at it. Resume takes load.path and looks for the
# training state at exactly that path, so anyone following the log would point
# it at the parent, get "Training state file not found. Starting training from
# scratch", and silently restart at step 0. That is the same trap as F14, one
# layer out: the loud part is right and the part you act on is wrong.


def test_the_logged_training_state_path_is_where_the_file_lands(tmp_path, caplog):
    """Whatever the log names has to be a file that exists."""
    from omegaconf import OmegaConf

    from src.core.checkpointing import save_training_state, step_checkpoint_path

    save_config = OmegaConf.create(
        {
            "path": str(tmp_path),
            "training_state_filename": "__training_state_filename.pt",
        }
    )
    step = 5
    os.makedirs(step_checkpoint_path(str(tmp_path), step), exist_ok=True)

    with caplog.at_level("INFO", logger="src.core.checkpointing"):
        save_training_state(save_config, step=step, processed_tokens=123)

    logged = [m for m in caplog.messages if "Saved training state in" in m]
    assert len(logged) == 1, f"expected one save line, got {logged}"
    path = logged[0].split("'")[1]
    assert os.path.isfile(path), (
        f"log points at '{path}', which does not exist -- a resume following this "
        f"log would restart from step 0"
    )


# ---------------------------------------------------------------------------
# The fineweb holdout.
#
# fineweb ships only a train directory on every cluster -- no validation split
# was ever downloaded -- so eval was pointed at the training data and scored the
# batches the model was training on, in the same order, flattering in exactly
# the direction nobody checks. There is no second directory to point at, so the
# fix carves one: the last holdout_fraction of rows goes to eval and the rest to
# train, both computed from the same row count. These assert the two slices are
# genuinely disjoint rather than merely configured to be.


def _holdout_stream(n, holdout, monkeypatch, rows=60, max_samples=None):
    from src.core.datasets import FineWebEduDataset, gpt2_tokenize_fn

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    dataset = FineWebEduDataset(
        sequence_length=16,
        tokenize_fn=gpt2_tokenize_fn(),
        path=os.path.join(REPO_ROOT, "data"),
        seed=123,
        shuffle=True,
        world_size_independent=True,
        holdout=holdout,
        holdout_rows=rows,
        max_samples=max_samples,
    )
    return [tuple(sequence) for sequence in itertools.islice(iter(dataset), n)]


@needs_local_data
def test_the_holdout_slices_do_not_overlap(monkeypatch):
    """What train sees and what eval sees must be disjoint."""
    train = _holdout_stream(20, "train", monkeypatch)
    evaluation = _holdout_stream(20, "eval", monkeypatch)
    assert train and evaluation
    shared = set(train) & set(evaluation)
    assert not shared, f"{len(shared)} sequences appear in both train and eval holdouts"


@needs_local_data
def test_without_a_holdout_eval_sees_the_training_data(monkeypatch):
    """The bug this guards against, stated as a fact rather than a worry.

    With holdout=None both dataloaders read every row with the same seed, so they
    produce the same stream -- which is what made the reported eval loss meaningless.
    """
    first = _holdout_stream(8, None, monkeypatch)
    second = _holdout_stream(8, None, monkeypatch)
    assert first == second, "two unheld-out streams should be identical, by seed"


@needs_local_data
def test_holdout_rejects_a_nonsense_setting(monkeypatch):
    with pytest.raises(ValueError, match="holdout must be"):
        _holdout_stream(1, "validation", monkeypatch)
    with pytest.raises(ValueError, match="holdout_rows must be"):
        _holdout_stream(1, "eval", monkeypatch, rows=0)
    with pytest.raises(ValueError, match="leaves no training rows"):
        _holdout_stream(1, "eval", monkeypatch, rows=10**9)


# ---------------------------------------------------------------------------
# get_metric_logger's config argument.
#
# Three call sites build it differently. main.py and initialization.py pass the
# config node; convert_memeff_to_hf passed instantiate(..., _convert_="all"),
# which is a plain dict, and get_metric_logger reads .type by attribute -- so
# both mem-eff -> HF conversion configs died with AttributeError the moment the
# earlier TypeError was fixed. Two bugs stacked in the same four lines, neither
# reachable without a trained 1B checkpoint to run the config against.


@pytest.mark.parametrize("as_plain_dict", [False, True])
def test_metric_logger_accepts_either_config_form(as_plain_dict):
    """A dict and a DictConfig describe the same config; both must work."""
    from src.core.metric_loggers import get_metric_logger

    config = {
        "type": "stdout",
        "name": "t",
        "tags": [],
        "new_wandb_job": True,
        "heavy_metrics_calculation_interval": 100,
    }
    if not as_plain_dict:
        config = OmegaConf.create(config)
    logger_obj = get_metric_logger(metric_logger_config=config)
    assert logger_obj is not None


# ---------------------------------------------------------------------------
# A fixed eval set.
#
# An eval number is only comparable across runs if it is measured on the same
# samples. Two things used to break that. The stream was endless, so "the eval
# set" was however much of it got read; and Trainer.eval reused one iterator
# created at startup, so each evaluation scored whatever came next rather than
# the same data -- eval loss moved partly because the data moved. max_samples
# makes the set finite, and eval re-creates the iterator so it replays.


@needs_local_data
def test_max_samples_makes_the_stream_finite(monkeypatch):
    """Without a cap the packer never ends, so the eval 'set' has no size."""
    capped = _holdout_stream(100, "eval", monkeypatch, max_samples=7)
    assert len(capped) == 7, f"expected the stream to stop at 7, got {len(capped)}"


@needs_local_data
def test_the_eval_set_is_the_same_samples_every_time(monkeypatch):
    """Re-iterating must replay, not continue."""
    first = _holdout_stream(5, "eval", monkeypatch, max_samples=5)
    second = _holdout_stream(5, "eval", monkeypatch, max_samples=5)
    assert first == second, "the eval set changed between evaluations"


# ---------------------------------------------------------------------------
# A dataloader may not hand the model a longer sequence than it was built for.


def test_rope_cannot_score_a_longer_sequence_than_it_was_built_for():
    """LlamaRoPE slices its buffers; it never extends them.

    __init__ builds sin/cos with `length` rows from the model's seq_len, and forward
    takes `self.cos[: x.shape[-2], :]`. Ask for more rows than exist and the slice
    returns every row it has -- silently the wrong shape -- and the broadcast fails.

    This is why the fineweb eval dataloader cannot pin its own sequence_length. Pinning
    2048 while a run trains at 1024 put exactly this error between the last training
    step and the final checkpoint.
    """
    from src.core.llama import LlamaRoPE

    rope = LlamaRoPE(dhead=8, length=16)
    rope(torch.zeros(1, 2, 16, 8))  # at the built length: fine

    with pytest.raises(RuntimeError, match="must match"):
        rope(torch.zeros(1, 2, 32, 8))


# ---------------------------------------------------------------------------
# `max_samples` is a budget for the whole job, not for each rank.
#
# The cap used to go on after the per-rank stride, so every rank took max_samples
# samples out of its own substream and the job scored max_samples * world_size. The
# eval set exists precisely to be independent of the GPU count, so that turned the one
# number meant to be comparable between runs into one that moved with the launch. At
# world_size 1 -- which is where it was tested -- the two are identical.


def _capped_stream(monkeypatch, world_size, rank, max_samples, n=None):
    from src.core.datasets import FineWebEduDataset, gpt2_tokenize_fn

    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    monkeypatch.setenv("RANK", str(rank))
    dataset = FineWebEduDataset(
        sequence_length=16,
        tokenize_fn=gpt2_tokenize_fn(),
        path=os.path.join(REPO_ROOT, "data"),
        seed=123,
        shuffle=True,
        world_size_independent=True,
        holdout="eval",
        holdout_rows=60,
        max_samples=max_samples,
    )
    stream = iter(dataset)
    if n is not None:
        stream = itertools.islice(stream, n)
    return [tuple(sequence) for sequence in stream]


@needs_local_data
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_the_eval_set_is_the_same_whatever_the_gpu_count(monkeypatch, world_size):
    """The union over ranks must be one fixed set of exactly max_samples samples."""
    union = []
    for rank in range(world_size):
        union.extend(_capped_stream(monkeypatch, world_size, rank, max_samples=12))

    single = _capped_stream(monkeypatch, 1, 0, max_samples=12)

    assert len(single) == 12, f"the cap itself is wrong: {len(single)}"
    assert len(union) == 12, (
        f"world_size={world_size} scored {len(union)} samples, not 12 -- the cap is "
        f"being applied per rank"
    )
    assert sorted(union) == sorted(single), "the ranks do not cover the same samples"


@needs_local_data
def test_the_ranks_do_not_score_the_same_sample_twice(monkeypatch):
    """Striding after the cap still has to partition, not duplicate."""
    per_rank = [
        _capped_stream(monkeypatch, 4, rank, max_samples=12) for rank in range(4)
    ]
    flat = [sequence for rank in per_rank for sequence in rank]
    assert len(flat) == len(set(flat)), "a sample is scored by more than one rank"


@needs_local_data
def test_a_capped_stream_refuses_a_sharded_dataloader(monkeypatch):
    """world_size_independent=False hands each rank a different shard to cap."""
    from src.core.datasets import FineWebEduDataset, gpt2_tokenize_fn

    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "0")
    dataset = FineWebEduDataset(
        sequence_length=16,
        tokenize_fn=gpt2_tokenize_fn(),
        path=os.path.join(REPO_ROOT, "data"),
        seed=123,
        shuffle=True,
        world_size_independent=False,
        holdout="eval",
        holdout_rows=60,
        max_samples=12,
    )
    with pytest.raises(ValueError, match="world_size_independent"):
        iter(dataset)


def test_a_capped_stream_refuses_multiple_dataloader_workers(monkeypatch):
    """Each worker runs __iter__ and applies its own cap, so the job reads N * workers.

    to_iterable_dataset shards by worker, so the workers do not hand back duplicate
    rows -- which is what makes this a silent multiplication rather than an obvious
    one. Needs no data: the check runs before anything is read.
    """
    from src.core.datasets import AbstractDataset

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    dataset = AbstractDataset(
        sequence_length=16,
        tokenize_fn=lambda x: x,
        world_size_independent=True,
        max_samples=8192,
    )

    class TwoWorkers:
        num_workers = 2

    monkeypatch.setattr(
        torch.utils.data, "get_worker_info", lambda: TwoWorkers(), raising=True
    )
    with pytest.raises(ValueError, match="num_workers"):
        dataset._check_the_cap_can_be_honoured()


def test_an_uncapped_stream_is_fine_with_workers_and_sharding(monkeypatch):
    """The guards must only fire when there is a cap to honour."""
    from src.core.datasets import AbstractDataset

    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "0")
    dataset = AbstractDataset(
        sequence_length=16,
        tokenize_fn=lambda x: x,
        world_size_independent=False,
        max_samples=None,
    )

    class TwoWorkers:
        num_workers = 2

    monkeypatch.setattr(
        torch.utils.data, "get_worker_info", lambda: TwoWorkers(), raising=True
    )
    dataset._check_the_cap_can_be_honoured()  # must not raise
