"""The looped-depth switch (common.loop): which blocks run how often, and its guards."""

import os

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import resolver  # noqa: F401
from src.core.model import ConcatInjection, Residual
from src.core.moe import MoE

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _h4(*overrides):
    with initialize_config_dir(
        version_base=None,
        config_dir=os.path.join(REPO_ROOT, "configs", "_model", "h_balanced"),
    ):
        cfg = compose(
            config_name="h4", overrides=["common.sequence_length=16", *overrides]
        )
    return instantiate(cfg.model, _convert_="all")


def _block_calls(model):
    calls = [0] * len(model.encoder.blocks)
    for i, block in enumerate(model.encoder.blocks):
        block.register_forward_hook(lambda *_, i=i: calls.__setitem__(i, calls[i] + 1))
    model(torch.randint(0, 100, (2, 16)))
    return calls


def test_off_runs_every_block_once():
    assert _block_calls(_h4()) == [1, 1, 1, 1]


def test_the_core_runs_n_loops_times_between_prelude_and_coda():
    model = _h4(
        "common.loop.n_loops=3", "common.loop.n_prelude=1", "common.loop.n_coda=1"
    )
    assert _block_calls(model) == [1, 3, 3, 1]


class _NameRecorder:
    def __init__(self):
        self.names = []

    def accumulate_metrics(self, layer_name, **_):
        self.names.append(layer_name)


def _logged_names(model):
    recorder = _NameRecorder()
    for module in model.modules():
        if isinstance(module, Residual):
            module.set_metric_logger(recorder)
    model(torch.randint(0, 100, (2, 16)))
    return [name.removesuffix("/residual_attention") for name in recorder.names[::2]]


def test_off_logs_the_plain_block_names():
    assert _logged_names(_h4()) == [f"block[{i}]" for i in range(4)]


def test_a_looped_block_logs_each_pass_under_its_own_name():
    model = _h4(
        "common.loop.n_loops=2", "common.loop.n_prelude=1", "common.loop.n_coda=1"
    )
    assert _logged_names(model) == [
        "block[0]",
        "block[1]/pass[0]",
        "block[2]/pass[0]",
        "block[1]/pass[1]",
        "block[2]/pass[1]",
        "block[3]",
    ]


def _tiny_local_loop_moe():
    with initialize_config_dir(
        version_base=None, config_dir=os.path.join(REPO_ROOT, "configs/jm/feature")
    ):
        cfg = compose(config_name="tiny_local_loop_moe")
    torch.manual_seed(0)
    return instantiate(cfg.model, _convert_="all")


def test_a_looped_moe_layer_sums_its_aux_loss_and_averages_its_stats_over_passes():
    model = _tiny_local_loop_moe()
    calls = {}
    for moe in (m for m in model.modules() if isinstance(m, MoE)):
        moe.register_forward_hook(
            lambda m, *_: calls.setdefault(m, []).append((m.aux_loss, m.stats))
        )
    model(torch.randint(0, 100, (2, 16)))
    assert sorted(len(c) for c in calls.values()) == [1, 1, 2, 2]
    for moe, per_call in calls.items():
        assert torch.equal(moe.aux_loss, sum(aux for aux, _ in per_call))
        assert torch.equal(moe.stats, torch.stack([s for _, s in per_call]).mean(0))
    # the passes route different hidden states, so merging is not a no-op
    looped = [c for c in calls.values() if len(c) == 2]
    assert all(not torch.equal(c[0][0], c[1][0]) for c in looped)


def test_concat_injection_starts_as_the_identity_on_the_state():
    torch.manual_seed(0)
    x, prelude_out = torch.randn(2, 16, 64), torch.randn(2, 16, 64)
    assert torch.equal(ConcatInjection(64)(x, prelude_out), x)


@pytest.mark.parametrize(
    "overrides, message",
    [
        (["common.loop.n_loops=0"], "n_loops must be >= 1"),
        (["common.loop.n_prelude=2", "common.loop.n_coda=2"], "non-empty core"),
        (["common.loop.n_prelude=-1"], "non-empty core"),
        (["common.loop.injection=add"], "none or concat"),
    ],
)
def test_a_bad_loop_config_is_a_build_error(overrides, message):
    with pytest.raises(Exception, match=message):
        _h4(*overrides)
