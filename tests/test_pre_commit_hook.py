"""The pre-commit hook has to actually run.

It guards the one thing that breaks clusters silently: a pixi.lock bumped to v7, which
helios's pixi 0.67.1 cannot read. Nothing exercised the hook, so it was free to rot --
and it did. Adding requires-pixi to pixi.toml broke it, because the hook called
`pixi run`, which that bound refuses on a modern pixi. Every commit would have failed
for anyone with the hook installed.
"""

import os
import shutil
import subprocess

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOK = os.path.join(REPO_ROOT, ".githooks", "pre-commit")


def _run_hook(cwd):
    return subprocess.run(["sh", HOOK], capture_output=True, text=True, cwd=cwd)


def test_the_hook_is_executable_shell():
    assert os.path.isfile(HOOK)
    check = subprocess.run(["sh", "-n", HOOK], capture_output=True, text=True)
    assert check.returncode == 0, f"the hook is not valid shell: {check.stderr}"


def test_the_hook_does_not_route_through_pixi_run():
    """requires-pixi refuses `pixi run` on a modern pixi, which would fail every commit."""
    code = [
        line
        for line in open(HOOK).read().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert not [line for line in code if "pixi run" in line], (
        "the hook calls `pixi run`, which pixi.toml's requires-pixi bound refuses on the "
        "pixi most people have -- every commit would fail"
    )


def test_the_hook_rejects_a_lock_that_is_not_v6(tmp_path):
    """The thing it exists for: a v7 lock must not get committed."""
    work = tmp_path / "repo"
    work.mkdir()
    (work / "pixi.lock").write_text("version: 7\nenvironments: {}\n")

    result = _run_hook(str(work))

    assert result.returncode != 0, "a v7 lock was accepted"
    assert "not version 6" in result.stdout + result.stderr


def test_the_hook_accepts_a_v6_lock(tmp_path):
    work = tmp_path / "repo"
    work.mkdir()
    (work / "pixi.lock").write_text("version: 6\nenvironments: {}\n")
    # no pixi env in the temp dir, so the formatting check should bow out cleanly
    result = _run_hook(str(work))
    assert result.returncode == 0, result.stdout + result.stderr
