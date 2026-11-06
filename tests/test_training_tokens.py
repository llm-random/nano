"""trainer.n_tokens_b: the training length in billions of tokens.

A config gives n_steps or n_tokens_b, never both. src.definitions.resolve_n_steps
turns tokens into whole optimizer steps of common.batch_size * common.sequence_length
tokens (the global batch), rounded down. It runs at submit time on every grid point
(check_point, which run_exp.py calls before anything is queued) and again at startup
in main.run, so the rest of the code only ever sees n_steps.

tiny_local: batch_size 2 x sequence_length 16 = 32 tokens per step, so 0.0001 B =
100,000 tokens = 3125 steps exactly, and its warmup interpolation gives 31.
"""

import os
import warnings

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import resolver  # noqa: F401  (registers the repo's custom OmegaConf resolvers)
from grid_generator.generate_configs import create_grid_config
from src.definitions import check_point, resolve_n_steps, steps_for_tokens

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(REPO_ROOT, "configs")

TOKENS = ["trainer.n_steps=null", "trainer.n_tokens_b=0.0001"]


def _compose(overrides=()):
    warnings.filterwarnings("ignore")
    with initialize_config_dir(version_base=None, config_dir=CONFIGS_DIR):
        return compose(config_name="tiny_local", overrides=list(overrides))


def _point(overrides=()):
    """A grid point as run_exp.py hands it to check_point: a plain, unresolved dict."""
    return OmegaConf.to_container(_compose(overrides), resolve=False)


def test_steps_round_down_to_whole_global_batches():
    # h16 Stage 1b/2b: 512 x 2048 = 1,048,576 tokens per step. The hand-written
    # configs say 7630 for 8B ("rounded up"); the knob rounds down.
    assert steps_for_tokens(8, 512, 2048) == 7629
    assert steps_for_tokens(4, 512, 2048) == 3814
    assert steps_for_tokens(1.048576, 512, 2048) == 1000  # an exact multiple


def test_float_noise_does_not_cost_a_step():
    # 1.001 * 10**9 is 1000999999.9999999 in floating point; flooring that
    # directly would give 1000999 steps of 1000 tokens instead of 1001000.
    assert steps_for_tokens(1.001, 1000, 1) == 1_001_000


def test_tokens_become_steps_and_the_warmup_interpolation_follows():
    cfg = _compose(TOKENS)  # Hydra's struct-mode config, as main() receives it
    resolve_n_steps(cfg)
    assert cfg.trainer.n_steps == 3125
    assert cfg.trainer.scheduler.warmup_steps == 31  # int(3125 * 0.01)


def test_steps_and_tokens_together_fail_before_submission():
    """tiny_local sets n_steps: 201, so adding tokens alone sets both."""
    with pytest.raises(ValueError, match="both set"):
        check_point(_point(["trainer.n_tokens_b=0.0001"]))


def test_both_set_fails_on_the_grid_path_too():
    plain = _point(["trainer.n_tokens_b=0.0001"])
    grid = create_grid_config(OmegaConf.create(plain))
    with pytest.raises(ValueError, match="trainer.n_steps=null"):
        for point, _overrides in grid:
            check_point(point)


def test_check_point_resolves_a_copy_and_leaves_the_point_as_written():
    """run_exp.py dumps the point it checked; the cluster-side main.run resolves
    the budget again, so the dumped config must still say tokens, not steps."""
    point = _point(TOKENS)
    checked = check_point(point)
    assert checked.trainer.n_steps == 3125
    assert point["trainer"]["n_steps"] is None
    assert point["trainer"]["n_tokens_b"] == 0.0001


def test_a_token_grid_resolves_point_by_point():
    plain = _point(["trainer.n_steps=null"])
    del plain["trainer"]["n_tokens_b"]
    plain["trainer"]["^n_tokens_b"] = [0.0001, 0.0002]
    grid = create_grid_config(OmegaConf.create(plain))
    assert sorted(check_point(p).trainer.n_steps for p, _ in grid) == [3125, 6250]


def test_the_global_batch_sets_the_step_count():
    """Same budget, twice the batch: half the steps, rounded down (1562.5)."""
    plain = _point(TOKENS)
    del plain["common"]["batch_size"]
    plain["common"]["^batch_size"] = [2, 4]
    grid = create_grid_config(OmegaConf.create(plain))
    assert sorted(check_point(p).trainer.n_steps for p, _ in grid) == [1562, 3125]


def test_a_budget_under_one_step_is_rejected():
    with pytest.raises(ValueError, match="less than one optimizer step"):
        check_point(_point(["trainer.n_steps=null", "trainer.n_tokens_b=1e-9"]))


@pytest.mark.parametrize("budget", ["0", "-1"])
def test_a_non_positive_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="must be > 0"):
        check_point(_point(["trainer.n_steps=null", f"trainer.n_tokens_b={budget}"]))


def test_main_run_resolves_tokens_before_anything_reads_n_steps(monkeypatch, tmp_path):
    """The cluster runs main.py on a dumped config that still says n_tokens_b;
    run() must turn it into steps before the model, optimizer and scheduler are
    built. The initializer is replaced by a probe that returns no model, which
    makes run() skip training."""
    import main

    seen = {}

    def probe(cfg, metric_logger=None):
        seen["n_steps"] = cfg.trainer.n_steps
        seen["warmup_steps"] = cfg.trainer.scheduler.warmup_steps
        return None, None, None, None, None

    monkeypatch.setattr(main, "initialize_training_components", probe)
    # setup_enviroment() fills these in when they are missing; set them so the
    # test leaves the process environment as it found it.
    for name, value in {
        "WORLD_SIZE": "1",
        "LOCAL_WORLD_SIZE": "1",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "12355",
        "TRITON_CACHE_DIR": str(tmp_path),
    }.items():
        monkeypatch.setenv(name, value)

    main.run(_compose([*TOKENS, "trainer.distributed=null"]))
    assert seen == {"n_steps": 3125, "warmup_steps": 31}
