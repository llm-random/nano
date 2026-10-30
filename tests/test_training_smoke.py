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
import re
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


def _run(tmp_path, save_type, n_steps=N_STEPS, name="ckpt", extra=()):
    save_path = str(tmp_path / name)
    from conftest import free_port

    environment = {
        **os.environ,
        # own rendezvous port per subprocess: the parent pytest process may have
        # leaked a MASTER_PORT into the environment, and on a shared GPU node a
        # fixed port collides across runs
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(free_port()),
    }
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
            f"trainer.n_steps={n_steps}",
            f"trainer.checkpoint.save.type={save_type}",
            f"trainer.checkpoint.save.path={save_path}",
            *extra,
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, (
        f"save.type={save_type} run failed:\n"
        f"{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
    )
    return save_path, result.stdout + result.stderr


def _losses_by_step(output):
    """step -> train loss, parsed from the stdout metric logger."""
    found = {}
    for line in output.splitlines():
        match = re.search(r"step:(\d+) .*train/loss: ([0-9.]+)", line)
        if match:
            found[int(match.group(1))] = match.group(2)
    return found


def test_local_training_writes_a_nano_checkpoint(tmp_path):
    save_path, _ = _run(tmp_path, "nano")
    step_dir = os.path.join(save_path, f"step_{N_STEPS - 1}")
    assert os.path.isdir(step_dir), f"no step_{N_STEPS - 1} directory under {save_path}"
    written = os.listdir(step_dir)
    assert any("model" in name for name in written), f"no model checkpoint: {written}"
    assert any(
        "training_state" in name for name in written
    ), f"no training state, so the run could not be resumed: {written}"


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
    # A fresh model's first loss must be ~ln(vocab_size): wrong init, a vocab
    # mismatch or a head-wiring bug all land here before any GPU time is spent.
    import math

    expected = math.log(50304)
    assert abs(losses[0] - expected) < 0.15, (
        f"first loss {losses[0]:.3f} is far from ln(50304)={expected:.3f}; "
        "suspect init, vocab or head wiring"
    )


RESUME_STEPS = 20
RESUME_AT = 10


def test_resuming_reproduces_the_uninterrupted_run(tmp_path):
    """A resumed run must continue, not start the data over.

    The datasets reseed in __iter__ and train() takes one batch per step, so a resumed
    run used to get a fresh iterator at batch 0 and re-train on data it had already
    seen, while the step counter carried on from the checkpoint. It failed in the worst
    possible direction: the loss after a resume came out *lower*, because the model was
    being shown examples it had already learned, so it read as a good run rather than a
    broken one. Measured at 10.17 against 12.39 on this config.

    n_steps is held equal across both runs on purpose. warmup_steps and the cosine
    T_max are derived from n_steps, so a shorter first run would train under a
    different learning-rate schedule and the comparison would be meaningless.
    """
    uninterrupted, output_a = _run(
        tmp_path,
        "nano",
        n_steps=RESUME_STEPS,
        name="straight",
        extra=[f"trainer.checkpoint.save.interval={RESUME_AT}"],
    )
    checkpoint = os.path.join(uninterrupted, f"step_{RESUME_AT}")
    assert os.path.isdir(checkpoint), f"no mid-run checkpoint at {checkpoint}"

    _, output_b = _run(
        tmp_path,
        "nano",
        n_steps=RESUME_STEPS,
        name="resumed",
        extra=[
            "trainer.checkpoint.load.type=nano",
            f"trainer.checkpoint.load.path={checkpoint}",
            "trainer.checkpoint.load.training_state_filename="
            "__training_state_filename.pt",
            "trainer.checkpoint.load.only_weights=false",
        ],
    )

    straight = _losses_by_step(output_a)
    resumed = _losses_by_step(output_b)

    compared = [s for s in range(RESUME_AT + 1, RESUME_STEPS) if s in straight]
    assert len(compared) >= 5, f"too few steps to compare: {compared}"
    assert set(compared) <= set(resumed), (
        f"resumed run is missing steps {sorted(set(compared) - set(resumed))} -- "
        "it did not pick up where the checkpoint left off"
    )

    mismatched = {
        s: (straight[s], resumed[s]) for s in compared if straight[s] != resumed[s]
    }
    assert not mismatched, (
        "resumed run diverged from the uninterrupted one "
        f"(step: straight vs resumed): {mismatched}"
    )
