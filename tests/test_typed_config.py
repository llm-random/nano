"""The typed-config layer (src/definitions.py, per documents/notes/typed_config_goal.md).

Three things are locked in here:

1. The schema and the Trainer class cannot drift apart silently.
2. A mistyped key, a wrong-typed value, or a bogus grid key fails on the laptop
   with a readable error -- at compose where Hydra enforces it, at the
   check_point schema merge where it does not (`+key=` force-adds slip past
   struct validation at compose).
3. Every config that was green before the schema is still green with it.
"""

import os
import warnings
from dataclasses import fields as dataclass_fields

import attr
import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError

import resolver  # noqa: F401  (registers the repo's custom OmegaConf resolvers)
import src.definitions as definitions
from grid_generator.generate_configs import create_grid_config
from src.core.trainer import Trainer
from src.definitions import TrainerSchema, check_point, validate

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(REPO_ROOT, "configs")

# Trainer attributes that main.py passes at call time rather than the yaml:
# trainer = instantiate(cfg.trainer); trainer(model=..., optimizer=..., ...).
# `scheduler` is deliberately NOT here: it is both a yaml node (the factory
# config) and a call-time argument, so it belongs to the schema too.
RUNTIME_INJECTED = {"model", "optimizer", "training_state", "metric_logger"}


def _compose(name, overrides=()):
    warnings.filterwarnings("ignore")
    with initialize_config_dir(version_base=None, config_dir=CONFIGS_DIR):
        return compose(config_name=name, overrides=list(overrides))


def test_trainer_schema_mirrors_the_trainer_class():
    """Every yaml-supplied Trainer attribute has a schema field and vice versa.

    Add a knob to one without the other and this names the odd one out --
    the declare-the-knob-first policy, enforced.
    """
    schema_fields = {f.name for f in dataclass_fields(TrainerSchema)}
    schema_fields -= {"_target_", "_partial_"}
    config_supplied = {a.name for a in attr.fields(Trainer)} - RUNTIME_INJECTED
    assert schema_fields == config_supplied, (
        f"schema-only: {sorted(schema_fields - config_supplied)}, "
        f"trainer-only: {sorted(config_supplied - schema_fields)}"
    )


def test_typo_key_override_fails_composition():
    with pytest.raises(ConfigCompositionException):
        _compose("tiny_local", ["trainer.n_stepz=5"])


def test_wrong_typed_value_fails_composition():
    with pytest.raises(ConfigCompositionException):
        _compose("tiny_local", ["trainer.n_steps=abc"])


def test_undeclared_plus_add_fails_check_point():
    """Hydra's `+` force-adds past struct validation, so composition alone lets
    `+trainer.bogus=1` through; the check_point layer (run by main.py and on
    every grid point in run_exp.py) is what rejects it."""
    cfg = _compose("tiny_local", ["+trainer.bogus=1"])
    with pytest.raises(ConfigKeyError, match="bogus"):
        check_point(cfg)


def test_declared_knob_composes_without_plus():
    cfg = _compose("tiny_local", ["trainer.compile_loss=true"])
    assert cfg.trainer.compile_loss is True


def test_bogus_grid_key_fails_generation():
    plain = OmegaConf.to_container(_compose("tiny_local"), resolve=False)
    plain["trainer"]["^n_stepz"] = [5, 6]
    grid = create_grid_config(OmegaConf.create(plain))
    with pytest.raises(ConfigKeyError, match="n_stepz"):
        for point, _overrides in grid:
            check_point(point)


def test_legit_grid_expands_and_passes_check_point():
    plain = OmegaConf.to_container(_compose("tiny_local"), resolve=False)
    plain["trainer"]["^learning_rate"] = [1e-3, 5e-4]
    del plain["trainer"]["learning_rate"]
    grid = create_grid_config(OmegaConf.create(plain))
    assert len(grid) == 2
    for point, _overrides in grid:
        check_point(point)


def test_range_violation_fails_validate():
    cfg = _compose("tiny_local", ["trainer.learning_rate=-1"])
    with pytest.raises(ValueError, match="learning_rate"):
        validate(cfg)


def test_warmup_exceeding_n_steps_fails_validate():
    cfg = _compose("tiny_local", ["trainer.n_steps=5", "trainer.scheduler.warmup_steps=10"])
    with pytest.raises(ValueError, match="warmup"):
        validate(cfg)


@pytest.mark.parametrize(
    "name", ["tiny_local", "exp", "smoketest_fineweb_final_eval_local"]
)
def test_schema_backed_top_level_configs_compose_and_validate(name):
    """The top-level runnable configs are outside the hygiene test's glob
    (configs/*/*.yaml), so their schema-backed composition is pinned here.
    exp.yaml leaves submit-time values as `???`/null on purpose, so only the
    fully-specified ones go through validate."""
    cfg = _compose(name)
    assert cfg.trainer._target_ == "src.core.trainer.Trainer"
    if name != "exp":
        validate(cfg)


def test_base_schema_registration_is_idempotent():
    definitions.register()
    cfg = _compose("tiny_local")
    assert cfg.common.h == 4


def test_n_steps_zero_passes_validate():
    """A finalize/conversion pass is configured with n_steps: 0 and builds a
    scheduler it never steps (src/core/schedulers.py says so explicitly);
    validate() must not kill it at startup."""
    cfg = _compose("tiny_local", ["trainer.n_steps=0"])
    validate(cfg)


def test_attention_mix_composes_and_passes_check_point():
    """The per-layer attention mix (configs/_model/h_balanced/base.yaml idiom
    `attention: {type: gqa, mix: {type: ame, layers: every_4th}}`) is a live
    knob; the schema must admit it through both the compose and grid paths."""
    cfg = _compose(
        "tiny_local", ["common.attention.mix={type: ame, layers: every_4th}"]
    )
    assert cfg.common.attention.mix.type == "ame"
    check_point(cfg)
    point = OmegaConf.to_container(cfg, resolve=False)
    check_point(point)


def test_expand_config_script_composes_a_schema_backed_config(tmp_path):
    """scripts/expand_config.py is the only way to run a `^`-grid config
    locally; it composes runnable configs, so it must register base_schema
    (it was the one composer the rewire missed on the first pass)."""
    import subprocess
    import sys

    out = tmp_path / "expanded.yaml"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/expand_config.py",
            "--config-path",
            "configs",
            "--config-name",
            "tiny_local",
            "-o",
            str(out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert out.exists()
