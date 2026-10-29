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


def test_apply_copies_backs_up_and_reinstalls(tmp_path):
    """The --apply path, with pixi stubbed out.

    Only the dry run had coverage, because --apply ends in `pixi install --frozen`,
    which would mean a real multi-gigabyte install. A stub on PATH records the call
    instead, so the copy, the backup and the install invocation are all checked without
    touching any environment -- and in particular without applying this to a shared
    cluster PIXI_HOME, which is a decision for whoever owns it.
    """
    target = _fake_pixi_home(tmp_path, differing=True)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "pixi_called.txt"
    stub = bin_dir / "pixi"
    stub.write_text(f'#!/bin/sh\necho "$@" > "{marker}"\n')
    stub.chmod(0o755)

    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
    result = subprocess.run(
        ["bash", SCRIPT, str(target), "--apply"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    repo_lock = open(os.path.join(REPO_ROOT, "pixi.lock")).read()
    assert (
        target / "pixi.lock"
    ).read_text() == repo_lock, "the lock was not copied over"
    backups = list(target.glob("pixi.lock.bak-*"))
    assert backups, "no backup of the replaced lock was kept"
    assert marker.exists(), "pixi install was never invoked"
    assert "--frozen" in marker.read_text(), (
        "pixi install ran without --frozen, so it would re-solve and rewrite the lock "
        "to a format the clusters cannot read"
    )


def test_apply_strips_requires_pixi_from_the_target_manifest(tmp_path):
    """The pixi bound must not travel into a cluster's own PIXI_HOME.

    requires-pixi stops a too-new pixi rewriting the *repo's* lock to a format helios
    cannot read. A cluster's PIXI_HOME is its own project, and carrying the bound there
    would stop that machine's pixi managing its own environment -- entropy runs pixi
    0.77, which the bound excludes, so a verbatim copy would make this script unusable
    on the very cluster that needs it most.
    """
    target = _fake_pixi_home(tmp_path, differing=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "pixi"
    stub.write_text("#!/bin/sh\nexit 0\n")
    stub.chmod(0o755)

    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
    result = subprocess.run(
        ["bash", SCRIPT, str(target), "--apply"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )
    assert result.returncode == 0, result.stderr

    written = (target / "pixi.toml").read_text()
    assert "requires-pixi" not in written, (
        "requires-pixi was copied into the cluster PIXI_HOME, which would stop that "
        "machine's pixi installing the environment"
    )
    # and the repo's own manifest still has it
    assert "requires-pixi" in open(os.path.join(REPO_ROOT, "pixi.toml")).read()


def test_apply_survives_a_filesystem_that_refuses_cp_p(tmp_path):
    """`cp -p` exits 1 on a filesystem that will not preserve ownership.

    The script runs under `set -euo pipefail`, so a failing cp aborts it -- and it
    aborts *between* the two copies, leaving the lock replaced and the manifest not,
    which is the one state the backups exist to avoid.

    This is not hypothetical: entropy's own /tmp refuses it, and this branch's suite
    failed there with "preserving permissions ... Operation not supported" while passing
    on a workstation. NFS with root_squash does the same, and that is what a cluster
    PIXI_HOME often sits on.

    Simulated with a `cp` on PATH that rejects -p the way those filesystems do: it
    copies the content, complains, and exits 1.
    """
    target = _fake_pixi_home(tmp_path, differing=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    (bin_dir / "pixi").write_text("#!/bin/sh\nexit 0\n")
    (bin_dir / "pixi").chmod(0o755)

    # Copies, then fails -- exactly what cp does when only the chown is refused.
    (bin_dir / "cp").write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-p" ]; then\n'
        "  shift\n"
        '  /bin/cp "$@"\n'
        '  echo "cp: preserving permissions for $2: Operation not supported" >&2\n'
        "  exit 1\n"
        "fi\n"
        'exec /bin/cp "$@"\n'
    )
    (bin_dir / "cp").chmod(0o755)

    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
    result = subprocess.run(
        ["bash", SCRIPT, str(target), "--apply"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )
    assert (
        result.returncode == 0
    ), f"the sync aborted on a filesystem that refuses cp -p:\n{result.stderr}"

    # Both halves landed, not just the first.
    assert (target / "pixi.lock").read_text() == open(
        os.path.join(REPO_ROOT, "pixi.lock")
    ).read()
    assert "requires-pixi" not in (target / "pixi.toml").read_text()
    assert list(target.glob("pixi.lock.bak-*")), "no backup of the replaced lock"
    assert list(target.glob("pixi.toml.bak-*")), "no backup of the replaced manifest"
