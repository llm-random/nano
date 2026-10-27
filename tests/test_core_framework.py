"""Tier 1 of the baseline audit: CPU unit tests for the framework itself.

See documents/notes/main_baseline_audit.md. These cover src/core and the grid
generator -- the parts a new, non-PC project inherits wholesale from main -- so they
matter most for main as a branch point. No GPU, no data, no cluster.
"""

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
