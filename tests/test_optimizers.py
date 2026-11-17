"""src/core/optimizers.py: parameter roles, param groups and their checkpoints."""

import os
import warnings

import pytest
import torch
import torch.distributed.checkpoint as dcp
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

import resolver  # noqa: F401  (registers the repo's custom OmegaConf resolvers)
from src.core.checkpointing import TrainingState, load_checkpoint_from_file
from src.core.optimizers import ROLES, build_optimizer_and_scheduler, param_roles
from src.core.schedulers import get_cosine_scheduler_with_warmup
from test_distributed import _shard, cpu_process_group  # noqa: F401  (fixture)

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
MODEL_FILE = "model.pt"


def _cfg(optimizer):
    warnings.filterwarnings("ignore")
    with initialize_config_dir(version_base=None, config_dir=CONFIGS_DIR):
        return compose("tiny_local", [f"_trainer/_optimizer@optimizer={optimizer}"])


def _build(optimizer, seed=0, sharded=False):
    """The h4 tiny_local model, optimizer and scheduler, as main.py builds them."""
    cfg = _cfg(optimizer)
    torch.manual_seed(seed)
    model = instantiate(cfg.model, _convert_="all")
    if sharded:
        model = _shard(model)
    return (model, *build_optimizer_and_scheduler(cfg, model, 1e-3))


def _step(model, optimizer, scheduler):
    torch.manual_seed(1)
    tokens = torch.randint(0, 50304, (2, 16), device=next(model.parameters()).device)
    model(tokens).float().square().mean().backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()


def _save(folder, model, optimizer, scheduler, sharded):
    """As Trainer.save_checkpoint writes a nano checkpoint."""
    if sharded:
        dcp.save(
            {"app": TrainingState(model, optimizer, scheduler)}, checkpoint_id=folder
        )
    else:
        state = {"model": model.state_dict(), "optim": optimizer.state_dict()}
        torch.save(
            {**state, "scheduler": scheduler.state_dict()}, f"{folder}/{MODEL_FILE}"
        )


def _load(folder, model, optimizer, scheduler, only_weights):
    load_config = OmegaConf.create(
        {
            "path": folder,
            "model_checkpoint_filename": MODEL_FILE,
            "only_weights": only_weights,
        }
    )
    load_checkpoint_from_file(load_config, model, optimizer, scheduler)


def _params(model):
    return [
        p.full_tensor() if hasattr(p, "full_tensor") else p for p in model.parameters()
    ]


def _hyperparameters(optimizer):
    return [
        {k: v for k, v in g.items() if k != "params"} for g in optimizer.param_groups
    ]


def test_every_h4_parameter_gets_exactly_one_role():
    model, _, _ = _build("adamw")
    roles = param_roles(model)
    assert {r: len(ps) for r, ps in roles.items()} == {
        # 2 RMSNorm and 2 QK-norm gains per block, plus the head's RMSNorm
        "vector": 17,
        "embedding": 1,
        "head": 1,
        "hidden": 28,
    }
    ids = [id(p) for ps in roles.values() for p in ps]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    assert set(ids) == {id(p) for p in model.parameters()}
    assert roles["embedding"][0] is model.embedding.weight
    assert roles["head"][0] is model.head.linear.weight


def test_adamw_is_one_group():
    _, optimizer, _ = _build("adamw")
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["weight_decay"] == 0.1


def test_adamw_groups_has_one_group_per_nonempty_role():
    _, optimizer, _ = _build("adamw_groups")
    assert [(g["role"], g["weight_decay"]) for g in optimizer.param_groups] == [
        ("vector", 0.0),
        ("embedding", 0.0),
        ("head", 0.1),
        ("hidden", 0.1),
    ]
    no_head = torch.nn.Sequential(
        torch.nn.Embedding(8, 4), torch.nn.Linear(4, 4, bias=False)
    )
    optimizer, _ = build_optimizer_and_scheduler(_cfg("adamw_groups"), no_head, 1e-3)
    assert [g["role"] for g in optimizer.param_groups] == ["embedding", "hidden"]


def test_cosine_takes_several_groups_only_with_one_lr():
    a, b = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    same = torch.optim.AdamW(
        [{"params": a.parameters()}, {"params": b.parameters()}], lr=1e-3
    )
    get_cosine_scheduler_with_warmup(
        same, warmup_steps=1, n_steps=10, final_lr_fraction=0.1
    )
    differ = torch.optim.AdamW(
        [{"params": a.parameters()}, {"params": b.parameters(), "lr": 1e-2}], lr=1e-3
    )
    with pytest.raises(AssertionError, match="one LR"):
        get_cosine_scheduler_with_warmup(
            differ, warmup_steps=1, n_steps=10, final_lr_fraction=0.1
        )


def _check_resume(folder, sharded):
    model, optimizer, scheduler = _build("adamw_groups", sharded=sharded)
    _step(model, optimizer, scheduler)
    _save(folder, model, optimizer, scheduler, sharded)
    resumed = _build("adamw_groups", seed=99, sharded=sharded)
    _load(folder, *resumed, only_weights=False)
    # Same next step from both: model, AdamW moments, group decays and LR all came back.
    _step(model, optimizer, scheduler)
    _step(*resumed)
    for a, b in zip(_params(model), _params(resumed[0])):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert _hyperparameters(resumed[1]) == _hyperparameters(optimizer)


def test_adamw_groups_checkpoint_resumes(tmp_path):
    _check_resume(str(tmp_path), sharded=False)


def test_adamw_groups_sharded_checkpoint_resumes(cpu_process_group, tmp_path):
    _check_resume(str(tmp_path), sharded=True)


def _check_only_weights(folder, sharded):
    model, optimizer, scheduler = _build("adamw", sharded=sharded)
    _step(model, optimizer, scheduler)
    _save(folder, model, optimizer, scheduler, sharded)
    grouped = _build("adamw_groups", seed=99, sharded=sharded)
    _load(folder, *grouped, only_weights=True)
    for a, b in zip(_params(model), _params(grouped[0])):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not grouped[1].state, "only_weights must start a fresh optimizer"
    assert [g["role"] for g in grouped[1].param_groups] == list(ROLES)
    _step(*grouped)


def test_only_weights_loads_an_adamw_checkpoint_into_adamw_groups(tmp_path):
    _check_only_weights(str(tmp_path), sharded=False)


def test_only_weights_loads_a_sharded_adamw_checkpoint_into_adamw_groups(
    cpu_process_group, tmp_path
):
    _check_only_weights(str(tmp_path), sharded=True)
