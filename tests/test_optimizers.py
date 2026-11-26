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
from src.core.moe import MoE
from src.core.optimizers import ROLES, build_optimizer_and_scheduler, param_roles
from src.core.schedulers import get_cosine_scheduler_with_warmup
from test_distributed import _shard, cpu_process_group  # noqa: F401  (fixture)

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
MODEL_FILE = "model.pt"


def _cfg(optimizer, overrides=()):
    warnings.filterwarnings("ignore")
    with initialize_config_dir(version_base=None, config_dir=CONFIGS_DIR):
        return compose(
            "tiny_local", [f"_trainer/_optimizer@optimizer={optimizer}", *overrides]
        )


def _build(optimizer, seed=0, sharded=False, overrides=()):
    """The h4 tiny_local model, optimizer and scheduler, as main.py builds them."""
    cfg = _cfg(optimizer, overrides)
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


def _two_groups(second_lr):
    a, b = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    return torch.optim.AdamW(
        [{"params": a.parameters()}, {"params": b.parameters(), "lr": second_lr}],
        lr=1e-3,
    )


def _lrs(optimizer, scheduler, n_steps):
    """Each group's LR at every step, then after the last one."""
    lrs = []
    for _ in range(n_steps):
        lrs.append([g["lr"] for g in optimizer.param_groups])
        optimizer.step()
        scheduler.step()
    return lrs + [[g["lr"] for g in optimizer.param_groups]]


def _stock_cosine(optimizer, warmup_steps, n_steps, final_lr_fraction):
    """The cosine schedule before per-group eta_min: torch's CosineAnnealingLR."""
    s = torch.optim.lr_scheduler
    eta_min = final_lr_fraction * optimizer.param_groups[0]["lr"]
    return s.SequentialLR(
        optimizer,
        schedulers=[
            s.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps),
            s.ConstantLR(optimizer, factor=1.0),
            s.CosineAnnealingLR(
                optimizer, T_max=n_steps - warmup_steps - 1, eta_min=eta_min
            ),
        ],
        milestones=[warmup_steps, warmup_steps + 1],
    )


def test_cosine_with_one_lr_is_bit_identical_to_torchs():
    ours, stock = _two_groups(1e-3), _two_groups(1e-3)
    scheduler = get_cosine_scheduler_with_warmup(
        ours, warmup_steps=3, n_steps=20, final_lr_fraction=0.1
    )
    reference = _stock_cosine(stock, warmup_steps=3, n_steps=20, final_lr_fraction=0.1)
    assert _lrs(ours, scheduler, 20) == _lrs(stock, reference, 20)


def test_cosine_decays_each_group_to_its_own_final_fraction():
    optimizer = _two_groups(1e-2)
    scheduler = get_cosine_scheduler_with_warmup(
        optimizer, warmup_steps=3, n_steps=20, final_lr_fraction=0.1
    )
    lrs = _lrs(optimizer, scheduler, 20)
    for first, second in lrs:
        assert second == pytest.approx(10 * first)
    assert lrs[-1] == pytest.approx([1e-4, 1e-3])


def _check_resume(folder, sharded, option="adamw_groups", overrides=()):
    model, optimizer, scheduler = _build(option, sharded=sharded, overrides=overrides)
    _step(model, optimizer, scheduler)
    _save(folder, model, optimizer, scheduler, sharded)
    resumed = _build(option, seed=99, sharded=sharded, overrides=overrides)
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


def _width(h):
    """Width h at depth 1."""
    return [f"common.h={h}", "common.n_blocks=1"]


@pytest.mark.parametrize("h, m", [(4, 1), (8, 2)])
def test_adamw_simplep_scales_only_the_hidden_and_head_lr(h, m):
    _, optimizer, _ = _build("adamw_simplep", overrides=_width(h))
    groups = [
        (g["role"], g["initial_lr"], g["weight_decay"]) for g in optimizer.param_groups
    ]
    assert groups == [
        ("vector", 1e-3, 0.0),
        ("embedding", 1e-3, 0.0),
        ("head", pytest.approx(1e-3 / m), 0.1),
        ("hidden", pytest.approx(1e-3 / m), 0.1),
    ]


def test_adamw_simplep_trains_every_moe_expert_and_the_router_as_hidden():
    moe = ["+_ff_layer@model.encoder.block_fn.ff_layer_fn=moe"]
    model, optimizer, _ = _build("adamw_simplep", overrides=_width(8) + moe)
    hidden = next(g for g in optimizer.param_groups if g["role"] == "hidden")
    in_hidden = {id(p) for p in hidden["params"]}
    moe_params = [
        p for m in model.modules() if isinstance(m, MoE) for p in m.parameters()
    ]
    assert moe_params and all(id(p) in in_hidden for p in moe_params)
    assert hidden["initial_lr"] == pytest.approx(1e-3 / 2)


def test_adamw_simplep_checkpoint_resumes(tmp_path):
    # Base h2 makes the h4 group LRs differ, so WSD's replay must restore each one.
    overrides = [
        "_trainer/_scheduler@trainer.scheduler=wsd",
        "optimizer.base_dmodel=128",
    ]
    _check_resume(
        str(tmp_path), sharded=False, option="adamw_simplep", overrides=overrides
    )
