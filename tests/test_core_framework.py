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
