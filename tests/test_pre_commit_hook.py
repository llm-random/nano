"""The pre-commit hook has to actually run.

It guards the one thing that breaks clusters silently: a pixi.lock bumped to v7, which
helios's pixi 0.67.1 cannot read. Nothing exercised the hook, so it was free to rot --
and it did. Adding requires-pixi to pixi.toml broke it, because the hook called
`pixi run`, which that bound refuses on a modern pixi. Every commit would have failed
for anyone with the hook installed.
"""

import os
import re
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


# ---------------------------------------------------------------------------
# Installing the hook must not need pixi.
#
# The only documented route was `pixi run install_hooks`. pixi.toml declares
# requires-pixi to stop a too-new pixi rewriting pixi.lock to v7, and that bound refuses
# *every* pixi command in the repo -- the install task included:
#
#   $ pixi run install_hooks
#   Error: this project requires pixi '>=0.60,<0.77', but you have pixi 0.81.0
#
# The hook exists to catch a lock that a too-new pixi rewrote. The people who need it
# are the people running a too-new pixi, and they were the ones who could not install
# it. Found by running the documented command rather than reading it.


def test_the_hook_installer_does_not_need_pixi():
    """It may *mention* pixi -- the comment explains the catch-22 -- but never call it."""
    installer = os.path.join(REPO_ROOT, "scripts/install_hooks.sh")
    assert os.access(installer, os.X_OK), "scripts/install_hooks.sh is not executable"

    offenders = []
    for number, line in enumerate(open(installer), start=1):
        code = line.split("#", 1)[0]
        if re.search(r"(^|[|&;(`$]|\s)pixi(\s|$)", code):
            offenders.append(f"{number}: {line.rstrip()}")
    assert not offenders, (
        "the installer invokes pixi, which is exactly what made it unreachable for "
        "anyone whose pixi requires-pixi excludes:\n  " + "\n  ".join(offenders)
    )


def test_the_installer_actually_sets_hookspath(tmp_path):
    """Run it against a throwaway repo, never this one."""
    work = tmp_path / "repo"
    (work / ".githooks").mkdir(parents=True)
    hook = work / ".githooks" / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 0\n")
    hook.chmod(0o755)
    subprocess.run(["git", "init", "-q"], cwd=work, check=True)

    result = subprocess.run(
        ["bash", os.path.join(REPO_ROOT, "scripts/install_hooks.sh")],
        cwd=work,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    configured = subprocess.run(
        ["git", "config", "--get", "core.hooksPath"],
        cwd=work,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert configured == ".githooks", f"core.hooksPath is {configured!r}"


def test_the_installer_refuses_where_there_is_no_hook(tmp_path):
    """Better to say so than to point git at an empty directory."""
    work = tmp_path / "repo"
    work.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=work, check=True)

    result = subprocess.run(
        ["bash", os.path.join(REPO_ROOT, "scripts/install_hooks.sh")],
        cwd=work,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "nothing to install" in result.stdout + result.stderr
