"""The MFU accounting: measured FLOPs, peak lookup, and the logging hook.

The FLOPs count is measured from the live model, so these tests build real (tiny)
models and check the measurement against hand-derived closed forms -- exactly, not
approximately. The logging hook is tested on a stub with a fake clock, because what
matters is the arithmetic and the null-safety, not wall time.
"""

import os

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import resolver  # noqa: F401
from src.core.attention import build_attention
from src.core.mfu import model_flops_per_token, peak_flops_bf16
from src.core.trainer import Trainer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB = 50304


def _h4(seq_len):
    with initialize_config_dir(
        version_base=None,
        config_dir=os.path.join(REPO_ROOT, "configs", "_model", "h_balanced"),
    ):
        cfg = compose(config_name="h4", overrides=[f"common.sequence_length={seq_len}"])
    with torch.device("meta"):
        return instantiate(cfg.model, _convert_="all")


def test_h4_flops_match_the_closed_form():
    """P = 15*L*d^2 + V*d for the gqa H family; scores add 12*s*d per layer."""
    h, s = 4, 2048
    d, L = 64 * h, h
    model = _h4(s)
    expected = 6 * (15 * L * d**2 + VOCAB * d) + 12 * L * s * d
    assert model_flops_per_token(model, s) == expected


def test_the_embedding_lookup_is_not_counted_but_the_head_is():
    """Ablate the formula: drop the head term and the count must disagree."""
    h, s = 4, 128
    d, L = 64 * h, h
    model = _h4(s)
    without_head = 6 * (15 * L * d**2) + 12 * L * s * d
    assert model_flops_per_token(model, s) != without_head


def test_an_mqa_layer_reports_its_smaller_kv_cost():
    """The count is measured, so kv_heads=1 must shave exactly the kv width."""
    dims = dict(dmodel=256, q_heads=4, kv_heads=2, seq_len=128)
    gqa = build_attention(0, attention={"type": "gqa"}, n_blocks=1, **dims)
    mqa = build_attention(0, attention={"type": "mqa"}, n_blocks=1, **dims)
    dhead = dims["dmodel"] // dims["q_heads"]
    gqa_params = sum(p.numel() for p in gqa.parameters() if p.dim() == 2)
    mqa_params = sum(p.numel() for p in mqa.parameters() if p.dim() == 2)
    assert gqa_params - mqa_params == 2 * dims["dmodel"] * dhead * (2 - 1)
    # score flops depend on q_heads only, so they are equal
    assert model_flops_per_token(gqa, 128) - model_flops_per_token(mqa, 128) == 6 * (
        gqa_params - mqa_params
    )


@pytest.mark.parametrize(
    "name, peak",
    [
        ("NVIDIA H100 80GB HBM3", 989e12),
        ("NVIDIA H100 PCIe", 756e12),
        ("NVIDIA A100-SXM4-80GB", 312e12),
        ("NVIDIA GH200 480GB", 989e12),
    ],
)
def test_known_gpus_resolve_to_their_bf16_peak(name, peak):
    assert peak_flops_bf16(name) == peak


def test_an_unknown_gpu_gives_none_not_a_wrong_number():
    assert peak_flops_bf16("NVIDIA GTX 710M") is None


# ---------------------------------------------------------------------------
# The logging hook, on a stub with a fake clock.


class _RecordingLogger:
    def __init__(self):
        self.logged = {}

    def log(self, name, value):
        self.logged[name] = value


def _throughput_stub():
    class Stub:
        _log_throughput = Trainer._log_throughput

    stub = Stub()
    stub.metric_logger = _RecordingLogger()
    stub._prev_log_time = None
    stub._prev_log_tokens = None
    stub._flops_per_token = None
    stub._peak_flops = None
    stub._seq_len_for_flops = 16
    stub.processed_tokens = 0
    stub.model = torch.nn.Linear(2, 2)
    stub.device = torch.device("cpu")
    return stub


def test_the_first_step_logs_no_throughput():
    stub = _throughput_stub()
    stub._log_throughput()
    assert "train/step_time" not in stub.metric_logger.logged


def test_tokens_per_second_is_the_token_delta_over_the_time_delta(monkeypatch):
    import src.core.trainer as trainer_module

    clock = iter([100.0, 102.0])
    monkeypatch.setattr(trainer_module.time, "time", lambda: next(clock))
    stub = _throughput_stub()
    stub._log_throughput()
    stub.processed_tokens = 1_048_576
    stub._log_throughput()
    logged = stub.metric_logger.logged
    assert logged["train/step_time"] == pytest.approx(2.0)
    assert logged["train/tokens_per_s"] == pytest.approx(1_048_576 / 2.0)
    # on CPU there is no device to look up, so no mfu -- and no crash
    assert "train/mfu" not in logged


def test_mfu_is_tokens_per_second_times_flops_over_peak(monkeypatch):
    import src.core.trainer as trainer_module

    clock = iter([100.0, 101.0])
    monkeypatch.setattr(trainer_module.time, "time", lambda: next(clock))
    stub = _throughput_stub()
    stub._flops_per_token = 2.0e9
    stub._peak_flops = 989e12
    stub._log_throughput()
    stub.processed_tokens = 1_048_576
    stub._log_throughput()
    logged = stub.metric_logger.logged
    expected = (1_048_576 / 1.0) * 2.0e9 / 989e12
    assert logged["train/mfu"] == pytest.approx(expected)
