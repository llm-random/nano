"""scripts/sync_cluster_env.sh must keep working, and must not touch anything on a dry run.

The script is how a cluster environment gets the repo's lock -- the step that was missing
when entropy and lem ended up without pytest despite pixi.toml declaring it. It is a shell
script, so nothing else in the suite would notice if it broke.
"""

import os
import shutil
import subprocess

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "sync_cluster_env.sh")


def _fake_pixi_home(tmp_path, *, differing: bool):
    target = tmp_path / "pixi"
    target.mkdir()
    shutil.copy(os.path.join(REPO_ROOT, "pixi.toml"), target / "pixi.toml")
    lock = open(os.path.join(REPO_ROOT, "pixi.lock")).read()
    if differing:
        lock = "\n".join(lock.splitlines()[:50]) + "\n"
    (target / "pixi.lock").write_text(lock)
    return target


def _run(*args):
    return subprocess.run(
        ["bash", SCRIPT, *args], capture_output=True, text=True, cwd=REPO_ROOT
    )


def test_dry_run_reports_a_difference_and_changes_nothing(tmp_path):
    target = _fake_pixi_home(tmp_path, differing=True)
    before = (target / "pixi.lock").read_text()

    result = _run(str(target))

    assert result.returncode == 0, result.stderr
    assert "DIFFERS" in result.stdout, result.stdout
    assert "Dry run" in result.stdout
    assert (target / "pixi.lock").read_text() == before, "a dry run modified the target"
    assert not list(target.glob("*.bak-*")), "a dry run wrote backups"


def test_identical_environment_is_a_no_op(tmp_path):
    target = _fake_pixi_home(tmp_path, differing=False)
    result = _run(str(target))
    assert result.returncode == 0, result.stderr
    assert "Nothing to do" in result.stdout, result.stdout


def test_missing_target_is_rejected(tmp_path):
    result = _run(str(tmp_path / "does_not_exist"))
    assert result.returncode != 0, "a nonexistent PIXI_HOME was accepted"
    assert "not a directory" in result.stderr


def test_no_argument_is_rejected(monkeypatch):
    monkeypatch.delenv("PIXI_HOME", raising=False)
    result = _run()
    assert result.returncode != 0
    assert "usage:" in result.stderr
