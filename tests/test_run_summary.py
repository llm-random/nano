"""scripts/run_summary.py reduces a finished run's slurm log to table numbers.

Two things it must get right, because both are silent when wrong:

- **Rank and step filtering.** All ranks share one job log, so averaging without
  filtering mixes four ranks' timers. Steps before 15 are warmup/compile, and
  steps 12-13 carry the kernel profiler when it is on, which distorts them.
- **Failure detection.** A log that crashed looks exactly like a log that
  finished unless something inspects it, so the tool exits non-zero on any
  failure signature.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "run_summary.py"

GOOD = """\
[INFO] MFU accounting: NVIDIA H100 80GB HBM3 x4, 3.770 GFLOPs/token at seq_len 2047, peak 3956.0 TFLOP/s (989.0 per GPU)
[INFO] timers_ms: rank=0 step=12 data_fetch=9.0 data_h2d=9.0 forward=9000.0 backward=9000.0 clip=9.0 optim=9.0 log=9.0 step_total=99999.0
[INFO] timers_ms: rank=0 step=15 data_fetch=2.0 data_h2d=10.0 forward=900.0 backward=1800.0 clip=40.0 optim=250.0 log=2.0 step_total=3000.0
[INFO] timers_ms: rank=3 step=15 data_fetch=9.0 data_h2d=9.0 forward=9000.0 backward=9000.0 clip=9.0 optim=9.0 log=9.0 step_total=99999.0
[INFO] timers_ms: rank=0 step=16 data_fetch=2.0 data_h2d=10.0 forward=900.0 backward=1800.0 clip=40.0 optim=250.0 log=2.0 step_total=3000.0
[INFO] Peak CUDA memory: 43.70 GiB
wandb:  train/mfu 0.1111
wandb:  train/mfu 0.2914
wandb:  train/tokens_per_s 305812
"""


def _run(tmp_path, text, name="slurm-1_0.out"):
    log = tmp_path / name
    log.write_text(text)
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(log)], capture_output=True, text=True
    )


def test_it_ignores_other_ranks_and_the_profiled_steps(tmp_path):
    result = _run(tmp_path, GOOD)
    assert result.returncode == 0, result.stdout + result.stderr
    # rank 3 and step 12 both carry 9000.0 forward; if either leaked into the
    # mean the printed value would not be 900.0. Matched on the numbers rather
    # than exact column spacing, which is formatting, not behaviour.
    forward = re.search(r"forward\s+([\d.]+) ms", result.stdout)
    assert forward and float(forward.group(1)) == pytest.approx(900.0), result.stdout
    assert "mean of 2 steps" in result.stdout
    assert "99999" not in result.stdout


def test_it_reports_the_last_tracker_value_not_the_first(tmp_path):
    result = _run(tmp_path, GOOD)
    assert "MFU/GPU=29.14%" in result.stdout
    assert "11.11" not in result.stdout


def test_it_names_the_flops_denominator(tmp_path):
    """Three h20 shapes are in play at different GFLOPs/token; a run's own log
    is the only thing that settles which one it used."""
    result = _run(tmp_path, GOOD)
    assert "3.770 GFLOPs/token" in result.stdout
    assert "x4" in result.stdout


def test_section_coverage_is_reported(tmp_path):
    """The T1.3 gate is '>=90% of step time accounted for'; the tool computes it."""
    result = _run(tmp_path, GOOD)
    assert "covered by sections:" in result.stdout


@pytest.mark.parametrize(
    "signature",
    [
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 9.50 GiB",
        "Traceback (most recent call last):",
        "srun: error: h100a: task 0: Exited with exit code 1",
        "slurmstepd: error: *** JOB 32543 CANCELLED AT 2026-11-04T23:00:00 ***",
    ],
)
def test_failures_are_detected_and_exit_nonzero(tmp_path, signature):
    result = _run(tmp_path, GOOD + signature + "\n")
    assert result.returncode == 1, f"{signature!r} was not detected"
    assert "FAILURE SIGNATURES" in result.stdout


def test_a_run_without_profile_timers_still_summarises(tmp_path):
    text = "\n".join(l for l in GOOD.splitlines() if "timers_ms" not in l)
    result = _run(tmp_path, text + "\n")
    assert result.returncode == 0
    assert "no timers_ms lines" in result.stdout
    assert "MFU/GPU=29.14%" in result.stdout


def test_a_handled_warning_is_not_a_failure(tmp_path):
    """The trainer catches a broken per-layer diagnostic and carries on. Marking
    that run failed is as misleading as missing a real crash -- lem 6237108 and
    6237109 both trained 30 steps and were reported as failures."""
    handled = (
        "[2026-11-04 19:30:26][src.core.trainer][WARNING] - instrumentation for "
        "encoder.blocks.0.attention_layer.layer failed and is disabled for the rest "
        "of this run: RuntimeError: aten.embedding.default: got mixed torch.Tensor "
        "and DTensor\n"
    )
    result = _run(tmp_path, GOOD + handled)
    assert result.returncode == 0, result.stdout
    assert "FAILURE SIGNATURES" not in result.stdout
    assert "handled by the run" in result.stdout


def test_a_real_crash_alongside_a_handled_warning_still_fails(tmp_path):
    handled = (
        "[WARNING] - instrumentation for x failed and is disabled for the rest of "
        "this run: RuntimeError: boom\n"
    )
    real = "[rank0]: torch.OutOfMemoryError: CUDA out of memory.\n"
    result = _run(tmp_path, GOOD + handled + real)
    assert result.returncode == 1
    assert "FAILURE SIGNATURES" in result.stdout
