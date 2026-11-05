"""A diagnostic must never kill a training run.

lem, 2026-11-04, jobs 6236888 and 6236892: both 4-GPU AME runs trained fine,
then died at the first heavy-metric step. `_log_layer_instrumentation` re-runs
part of the model's forward OUTSIDE FSDP2's forward hooks, where parameters are
still sharded DTensors, and `aten.embedding` on a DTensor weight with plain
int64 ids raises:

    RuntimeError: aten.embedding.default: got mixed torch.Tensor and DTensor

The readout is a handful of scalars. The run was hours of GPU time. Losing the
observation is right; losing the run is not.
"""

import logging
import types

import pytest
import torch.nn as nn

from src.core.trainer import Trainer


class _Logger:
    """The slice of MetricLogger that _log_layer_instrumentation touches."""

    def __init__(self, heavy=True):
        self._should_log_heavy_metrics = heavy
        self.logged = {}

    def log(self, key, value):
        self.logged[key] = value


class _Exploding(nn.Module):
    calls = 0

    def instrumentation(self, token_ids=None):
        type(self).calls += 1
        raise RuntimeError(
            "aten.embedding.default: got mixed torch.Tensor and DTensor, "
            "need to convert all torch.Tensor to DTensor before calling "
            "distributed operators!"
        )


class _Healthy(nn.Module):
    def instrumentation(self, token_ids=None):
        return {"tau/mean": 11.3}


def _trainer_stub(model):
    stub = types.SimpleNamespace(
        model=model,
        metric_logger=_Logger(),
        _last_input_ids=None,
        _instrumentation_failures=set(),
    )
    return stub


def _run(stub):
    Trainer._log_layer_instrumentation(stub)


def test_a_failing_readout_does_not_propagate():
    model = nn.Sequential()
    model.add_module("boom", _Exploding())
    stub = _trainer_stub(model)
    _run(stub)  # must not raise


def test_healthy_readouts_on_other_modules_still_land():
    """One broken layer must not silence the rest."""
    model = nn.Sequential()
    model.add_module("boom", _Exploding())
    model.add_module("fine", _Healthy())
    stub = _trainer_stub(model)
    _run(stub)
    assert stub.metric_logger.logged == {"fine/tau/mean": 11.3}


def test_it_warns_once_then_stops_retrying(caplog):
    """A per-step retry would spam the log and re-pay the cost every time."""
    _Exploding.calls = 0
    model = nn.Sequential()
    model.add_module("boom", _Exploding())
    stub = _trainer_stub(model)
    with caplog.at_level(logging.WARNING):
        _run(stub)
        _run(stub)
        _run(stub)
    warnings = [r for r in caplog.records if "boom" in r.getMessage()]
    assert len(warnings) == 1, f"expected one warning, got {len(warnings)}"
    assert "disabled for the rest of this run" in warnings[0].getMessage()
    assert _Exploding.calls == 1, (
        f"readout retried {_Exploding.calls} times; it should be skipped after "
        "the first failure"
    )


def test_the_warning_names_the_module_and_the_cause():
    """A silent skip would hide a real bug; the log must be actionable."""
    # nested, so named_modules() yields the dotted path a real model has
    attention = _Exploding()
    block = nn.Sequential()
    block.add_module("attention", attention)
    blocks = nn.Sequential()
    blocks.add_module("0", block)
    model = nn.Sequential()
    model.add_module("blocks", blocks)
    stub = _trainer_stub(model)
    import logging as _l

    records = []

    class _H(_l.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = _l.getLogger("src.core.trainer")
    handler = _H()
    logger.addHandler(handler)
    try:
        _run(stub)
    finally:
        logger.removeHandler(handler)
    assert any("blocks.0.attention" in m and "DTensor" in m for m in records), records


def test_nothing_runs_when_heavy_metrics_are_off():
    _Exploding.calls = 0
    model = nn.Sequential()
    model.add_module("boom", _Exploding())
    stub = _trainer_stub(model)
    stub.metric_logger._should_log_heavy_metrics = False
    _run(stub)
    assert _Exploding.calls == 0
