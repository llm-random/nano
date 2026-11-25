"""trainer.n_tokens_b: the training length in billions of tokens.

A config gives n_steps or n_tokens_b, never both. src.definitions.resolve_n_steps
turns tokens into whole optimizer steps of common.batch_size *
(common.sequence_length - 1) tokens (the tokens a global batch trains on), rounded
up. It runs at submit time on every grid point (check_point, which run_exp.py calls
before anything is queued) and again at startup in main.run, so the rest of the code
only ever sees n_steps.

tiny_local: batch_size 2 x (sequence_length 16 - 1) = 30 tokens per step, so
0.0001 B = 100,000 tokens = 3334 steps (3333.3 rounded up), and its warmup
interpolation gives 33.
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


def test_steps_are_the_hand_written_counts():
    # 512 rows of sequence_length 2049 train on 512 x 2048 = 1,048,576 tokens a
    # step. The stage configs' n_steps rounded up (8e9 / 1,048,576 = 7629.4), and
    # so does the knob (documents/notes/review_issues.md, N4).
    assert steps_for_tokens(4, 512, 2049) == 3815
    assert steps_for_tokens(8, 512, 2049) == 7630
    assert steps_for_tokens(16, 512, 2049) == 15259
    assert steps_for_tokens(1.048576, 512, 2049) == 1000  # an exact multiple


def test_float_noise_does_not_cost_or_add_a_step():
    # Steps of 1000 x (2 - 1) = 1000 tokens. 1.001 * 10**9 is
    # 1000999999.9999999 in floating point and 0.067 * 10**9 is
    # 67000000.00000001; rounding the latter up directly would add a step.
    assert steps_for_tokens(1.001, 1000, 2) == 1_001_000
    assert steps_for_tokens(0.067, 1000, 2) == 67_000


def test_tokens_become_steps_and_the_warmup_interpolation_follows():
    cfg = _compose(TOKENS)  # Hydra's struct-mode config, as main() receives it
    resolve_n_steps(cfg)
    assert cfg.trainer.n_steps == 3334
    assert cfg.trainer.scheduler.warmup_steps == 33  # int(3334 * 0.01)


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
    assert checked.trainer.n_steps == 3334
    assert point["trainer"]["n_steps"] is None
    assert point["trainer"]["n_tokens_b"] == 0.0001


def test_run_exp_checks_missing_keys_on_a_token_budget():
    """run_exp.py checks for ??? before the grid expands. missing_keys resolves
    interpolations on its way, and the warmup's ${trainer.n_steps} is None until
    the budget is resolved, so the check resolves a copy and leaves the config
    as written."""
    from run_exp import check_missing_keys

    cfg = _compose(TOKENS)
    check_missing_keys(cfg)  # raised InterpolationResolutionError before the fix
    assert cfg.trainer.n_steps is None
    assert cfg.trainer.n_tokens_b == 0.0001


def test_run_exp_still_reports_a_missing_key_with_a_token_budget():
    from run_exp import check_missing_keys

    cfg = _compose([*TOKENS, "trainer.learning_rate=???"])
    with pytest.raises(RuntimeError, match="learning_rate"):
        check_missing_keys(cfg)


def _token_sweep(key, values):
    """tiny_local with a token budget and `^key: values` in place of `key`."""
    plain = _point(TOKENS)
    section = "trainer" if key == "n_tokens_b" else "common"
    del plain[section][key]
    plain[section][f"^{key}"] = values
    return OmegaConf.create(plain)


@pytest.mark.parametrize(
    "key, values", [("n_tokens_b", [0.0001, 0.0002]), ("batch_size", [2, 4])]
)
def test_run_exp_checks_missing_keys_on_a_token_sweep(key, values):
    """A sweep over the budget, or over what a step holds, has no single budget
    or batch to resolve before the grid expands; the check runs on every point
    (documents/notes/review_issues.md, N5). Both raised before."""
    from run_exp import check_missing_keys

    cfg = _token_sweep(key, values)
    check_missing_keys(cfg)
    assert f"^{key}" in (cfg.trainer if key == "n_tokens_b" else cfg.common)


def test_run_exp_reports_a_missing_key_inside_a_token_sweep():
    from run_exp import check_missing_keys

    cfg = _token_sweep("n_tokens_b", [0.0001, 0.0002])
    cfg.trainer.learning_rate = "???"
    with pytest.raises(RuntimeError, match="learning_rate"):
        check_missing_keys(cfg)


def test_a_token_grid_resolves_point_by_point():
    plain = _point(["trainer.n_steps=null"])
    del plain["trainer"]["n_tokens_b"]
    plain["trainer"]["^n_tokens_b"] = [0.0001, 0.0002]
    grid = create_grid_config(OmegaConf.create(plain))
    assert sorted(check_point(p).trainer.n_steps for p, _ in grid) == [3334, 6667]


def test_the_global_batch_sets_the_step_count():
    """Same budget, twice the batch: half the steps, rounded up (1666.7)."""
    plain = _point(TOKENS)
    del plain["common"]["batch_size"]
    plain["common"]["^batch_size"] = [2, 4]
    grid = create_grid_config(OmegaConf.create(plain))
    assert sorted(check_point(p).trainer.n_steps for p, _ in grid) == [1667, 3334]


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
    assert seen == {"n_steps": 3334, "warmup_steps": 33}
