"""Tier 2 of the baseline audit, in the form that runs without a cluster.

See documents/notes/main_baseline_audit.md. These drive main.py end to end the way the
README documents a local run -- a tiny model, the dummy dataset in data/, CPU only --
and assert that a checkpoint actually lands on disk.

Unit tests cover the save-type *dispatch*; this covers the part that only a real run
exercises: the training loop, the final-checkpoint branch, and the writers themselves.
It is what caught nothing during the audit precisely because it was run, and it is
cheap insurance against a refactor of Trainer.train() that unit tests would miss.

Marked slow (about 40s each) so CI deselects it, and skipped where the local dataset is
absent -- data/ and data_eval/ are gitignored, so a fresh clone or a git worktree does
not have them.
"""

import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (
            os.path.isdir(os.path.join(REPO_ROOT, "data"))
            and os.path.isdir(os.path.join(REPO_ROOT, "data_eval"))
        ),
        reason="local dummy dataset not present (data/ and data_eval/ are gitignored)",
    ),
]

N_STEPS = 12


def _run(tmp_path, save_type):
    save_path = str(tmp_path / "ckpt")
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "ignore",
            "main.py",
            "--config-path",
            "configs",
            "--config-name",
            "tiny_local",
            "infrastructure.metric_logger.type=stdout",
            f"trainer.n_steps={N_STEPS}",
            f"trainer.checkpoint.save.type={save_type}",
            f"trainer.checkpoint.save.path={save_path}",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, (
        f"save.type={save_type} run failed:\n"
        f"{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
    )
    return save_path, result.stdout + result.stderr


def test_local_training_writes_a_nano_checkpoint(tmp_path):
    save_path, _ = _run(tmp_path, "nano")
    step_dir = os.path.join(save_path, f"step_{N_STEPS - 1}")
    assert os.path.isdir(step_dir), f"no step_{N_STEPS - 1} directory under {save_path}"
    written = os.listdir(step_dir)
    assert any("model" in name for name in written), f"no model checkpoint: {written}"
    assert any(
        "training_state" in name for name in written
    ), f"no training state, so the run could not be resumed: {written}"


def test_local_training_writes_a_huggingface_checkpoint(tmp_path):
    """The save.type the refactor of Trainer.save_final_checkpoint touched."""
    save_path, _ = _run(tmp_path, "huggingface")
    written = set(os.listdir(save_path))
    for expected in ("config.json", "model.safetensors"):
        assert expected in written, f"{expected} missing from {sorted(written)}"


def test_training_loop_reports_a_loss_for_every_step(tmp_path):
    """A silent loop that trains nothing would still exit 0 and save a checkpoint."""
    _, output = _run(tmp_path, "nano")
    losses = [
        float(line.split("loss:")[1].split()[0].rstrip(","))
        for line in output.splitlines()
        if "loss:" in line
    ]
    assert len(losses) >= N_STEPS, f"expected >= {N_STEPS} losses, got {len(losses)}"
    assert all(value == value for value in losses), "loss went NaN"
    assert all(value > 0 for value in losses), "loss is not positive"
