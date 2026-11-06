"""trainer.bank_lr_ratio: AME's memory banks train at a multiple of the backbone lr.

Default 4. The banks (every module's `bank_parameters`: AME's q_bank/k_bank) get a
param group of their own at ratio x lr (src.core.model.parameter_groups), or, under a
memory-parallel controller, its optimizer is stepped at ratio x the backbone's lr
(Trainer._bank_lr). The cosine scheduler floors every group at its own lr
(src.core.schedulers.CosineAnnealingToFraction), so the ratio holds for the whole
run -- and with one lr across groups the schedule is bit-identical to torch's.
"""

import os
import warnings
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import resolver  # noqa: F401
from src.core.attention import build_attention
from src.core.model import parameter_groups
from src.core.schedulers import get_cosine_scheduler_with_warmup
from src.definitions import validate

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LR = 2**-9


def _torch_cosine_with_warmup(optimizer, warmup_steps, n_steps, final_lr_fraction):
    """The schedule as it was built before per-group floors: torch's
    CosineAnnealingLR with one scalar eta_min."""
    lr = {group["lr"] for group in optimizer.param_groups}.pop()
    sched = torch.optim.lr_scheduler
    return sched.SequentialLR(
        optimizer,
        schedulers=[
            sched.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
            ),
            sched.ConstantLR(optimizer, factor=1.0),
            sched.CosineAnnealingLR(
                optimizer,
                T_max=n_steps - warmup_steps - 1,
                eta_min=final_lr_fraction * lr,
            ),
        ],
        milestones=[warmup_steps, warmup_steps + 1],
    )


def _lr_trace(make_scheduler, ratios, n_steps, warmup_steps):
    """Every group's lr at every step, for groups at `ratio x LR`."""
    params = [torch.nn.Parameter(torch.zeros(1)) for _ in ratios]
    optimizer = torch.optim.AdamW(
        [
            {"params": [p], **({} if ratio == 1 else {"lr": ratio * LR})}
            for p, ratio in zip(params, ratios)
        ],
        lr=LR,
    )
    warnings.filterwarnings("ignore")  # SequentialLR's epoch-argument warning
    scheduler = make_scheduler(optimizer, warmup_steps, n_steps, 0.1)
    trace = []
    for _ in range(n_steps):
        trace.append([group["lr"] for group in optimizer.param_groups])
        optimizer.step()
        scheduler.step()
    return trace


@pytest.mark.parametrize("n_steps, warmup_steps", [(201, 2), (7629, 76), (30, 0)])
def test_one_lr_is_the_old_schedule_bit_for_bit(n_steps, warmup_steps):
    """Pins CosineAnnealingToFraction against torch's class, so a torch
    upgrade that changes the upstream formulas fails here, not in a run."""
    old = _lr_trace(_torch_cosine_with_warmup, [1, 1], n_steps, warmup_steps)
    new = _lr_trace(get_cosine_scheduler_with_warmup, [1, 1], n_steps, warmup_steps)
    assert new == old


@pytest.mark.parametrize("n_steps, warmup_steps", [(201, 2), (7629, 76)])
def test_a_bank_group_holds_its_ratio_for_the_whole_run(n_steps, warmup_steps):
    alone = _lr_trace(get_cosine_scheduler_with_warmup, [1], n_steps, warmup_steps)
    paired = _lr_trace(get_cosine_scheduler_with_warmup, [1, 4], n_steps, warmup_steps)
    # A power-of-two ratio is exact in binary floating point.
    assert all(bank == 4 * base for base, bank in paired)
    # And the backbone's schedule does not notice the extra group.
    assert [base for base, _ in paired] == [lr for (lr,) in alone]
    # Annealed to its own floor (0.1 x 4 LR), not the backbone's.
    assert paired[-1][1] == pytest.approx(0.1 * 4 * LR, rel=1e-3)


# ---------------------------------------------------------------------------
# parameter_groups


def _ame_and_dense(n_blocks=2):
    """One AME layer and one gqa layer side by side (vocab 128, dmodel 64)."""
    torch.manual_seed(0)
    attention = {"type": "gqa", "mix": {"type": "ame", "layers": "every_2nd"}}
    dims = dict(dmodel=64, q_heads=4, kv_heads=2, seq_len=16, vocab_size=128)
    return torch.nn.ModuleList(
        build_attention(i, attention=attention, n_blocks=n_blocks, **dims)
        for i in range(n_blocks)
    )


def _banks(model):
    return [
        p
        for module in model.modules()
        for name in getattr(module, "bank_parameters", ())
        for p in [module.get_parameter(name)]
    ]


def test_the_banks_get_a_group_of_their_own():
    model = _ame_and_dense()
    groups = parameter_groups(model, 0.1, lr=LR, bank_lr_ratio=4.0)
    assert len(groups) == 3
    bank_group = groups[-1]
    assert bank_group["lr"] == 4 * LR
    assert bank_group["weight_decay"] == 0.0  # AME keeps its banks out of decay
    assert {id(p) for p in bank_group["params"]} == {id(p) for p in _banks(model)}
    assert all("lr" not in group for group in groups[:-1])
    # Nothing dropped, nothing in two groups.
    claimed = [id(p) for group in groups for p in group["params"]]
    assert sorted(claimed) == sorted(id(p) for p in model.parameters())


def test_ratio_one_is_the_old_grouping():
    model = _ame_and_dense()
    old = parameter_groups(model, 0.1)
    new = parameter_groups(model, 0.1, lr=LR, bank_lr_ratio=1.0)
    assert len(new) == len(old) == 2
    for a, b in zip(old, new):
        assert a.keys() == b.keys()
        assert [id(p) for p in a["params"]] == [id(p) for p in b["params"]]


def test_a_model_without_banks_is_untouched():
    torch.manual_seed(0)
    dense = build_attention(
        0,
        attention={"type": "gqa"},
        n_blocks=1,
        dmodel=64,
        q_heads=4,
        kv_heads=2,
        seq_len=16,
    )
    groups = parameter_groups(dense, 0.1, lr=LR, bank_lr_ratio=4.0)
    assert len(groups) == 1 and "lr" not in groups[0]


def test_banks_a_controller_owns_get_no_group_here():
    """Under memory parallelism the controller's optimizer steps the banks and
    applies the ratio itself (Trainer._bank_lr); the backbone optimizer must
    neither see them nor grow an empty group."""
    model = _ame_and_dense()
    groups = parameter_groups(
        model, 0.1, exclude=set(_banks(model)), lr=LR, bank_lr_ratio=4.0
    )
    assert len(groups) == 2 and all("lr" not in group for group in groups)


def test_a_ratio_without_the_base_lr_is_an_error():
    with pytest.raises(ValueError, match="needs the base `lr`"):
        parameter_groups(_ame_and_dense(), 0.1, bank_lr_ratio=4.0)


def test_weight_decay_zero_does_not_duplicate_the_banks():
    model = _ame_and_dense()
    groups = parameter_groups(model, 0.0, lr=LR, bank_lr_ratio=4.0)
    claimed = [id(p) for group in groups for p in group["params"]]
    assert len(claimed) == len(set(claimed)) == len(list(model.parameters()))


# ---------------------------------------------------------------------------
# The wiring: config -> main.py -> optimizer, and the controller path's lr.


def _compose_tiny(overrides):
    warnings.filterwarnings("ignore")
    with initialize_config_dir(
        version_base=None, config_dir=os.path.join(REPO_ROOT, "configs")
    ):
        return compose(config_name="tiny_local", overrides=list(overrides))


@pytest.mark.parametrize("ratio, groups", [(None, 3), (1.0, 2), (2.5, 3)])
def test_main_builds_the_optimizer_from_the_config(ratio, groups):
    import main

    overrides = [
        "common.attention.type=ame",
        "common.vocab_size=512",  # small banks: this test only builds groups
        "trainer.distributed=null",
    ]
    if ratio is not None:
        overrides.append(f"trainer.bank_lr_ratio={ratio}")
    cfg = _compose_tiny(overrides)
    torch.manual_seed(0)
    model = instantiate(cfg.model, _convert_="all")
    _, optimizer, scheduler = main.get_model_optimizer_scheduler(cfg, model, LR)

    assert len(optimizer.param_groups) == groups
    expected = 4.0 if ratio is None else ratio  # 4 is the default
    if groups == 3:
        assert optimizer.param_groups[-1]["lr"] == pytest.approx(expected * 0.1 * LR)
        for _ in range(3):  # through warmup: the ratio holds
            optimizer.step()
            scheduler.step()
            base, bank = (
                optimizer.param_groups[0]["lr"],
                optimizer.param_groups[-1]["lr"],
            )
            assert bank == pytest.approx(expected * base, rel=1e-12)


def test_the_trainer_steps_a_controller_at_the_ratio():
    """Trainer._bank_lr is what the training loop hands the memory-parallel
    controller each step (and logs as train/bank_lr)."""
    from src.core.trainer import Trainer

    stub = SimpleNamespace(
        bank_lr_ratio=4.0,
        optimizer=SimpleNamespace(param_groups=[{"lr": 0.5}, {"lr": 9.0}]),
    )
    assert Trainer._bank_lr(stub) == 2.0


@pytest.mark.parametrize("ratio", ["0", "-1"])
def test_a_non_positive_ratio_fails_validation(ratio):
    with pytest.raises(ValueError, match="bank_lr_ratio"):
        validate(_compose_tiny([f"trainer.bank_lr_ratio={ratio}"]))
