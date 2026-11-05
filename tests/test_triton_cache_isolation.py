"""Every rank needs its own Triton compile cache on node-local storage.

Triton defaults TRITON_CACHE_DIR to `$HOME/.triton/cache` and publishes entries
with `os.replace(temp, final)` -- atomic on a local filesystem, not on NFS. On
lem (2026-11-04, jobs 6236712 and 6236713) four ranks and two concurrent jobs
shared one NFS cache and tore each other's entries apart during compilation,
before step 1, producing three different errors that each named a file rather
than the cause:

    FileNotFoundError [Errno 2] ... __triton_launcher.so
    OSError [Errno 26] Text file busy
    ImportError ... cannot open shared object file: Permission denied

Both jobs died with zero training steps. This is the regression guard.
"""

import importlib
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in ("TRITON_CACHE_DIR", "SLURM_JOB_ID", "RANK", "TMPDIR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    return tmp_path


def _call():
    main = importlib.import_module("main")
    main.isolate_triton_cache()
    return os.environ.get("TRITON_CACHE_DIR")


def test_each_rank_gets_a_distinct_directory(clean_env, monkeypatch):
    """The race is between ranks, so two ranks must never share a path."""
    monkeypatch.setenv("SLURM_JOB_ID", "6236712")
    seen = set()
    for rank in ("0", "1", "2", "3"):
        monkeypatch.setenv("RANK", rank)
        monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
        seen.add(_call())
    assert len(seen) == 4, f"ranks shared a cache dir: {seen}"


def test_two_concurrent_jobs_do_not_share(clean_env, monkeypatch):
    """lem's two jobs raced each other through $HOME, on separate nodes."""
    monkeypatch.setenv("RANK", "0")
    paths = []
    for job in ("6236712", "6236713"):
        monkeypatch.setenv("SLURM_JOB_ID", job)
        monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
        paths.append(_call())
    assert paths[0] != paths[1], f"two jobs shared a cache dir: {paths}"


def test_it_lands_on_node_local_storage_not_home(clean_env, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("RANK", "0")
    path = _call()
    assert path.startswith(str(clean_env)), path
    assert ".triton" not in path
    assert os.path.isdir(path), "the directory must exist before Triton needs it"


def test_an_explicit_setting_is_honoured(clean_env, monkeypatch):
    """An operator who has pointed this somewhere deliberate keeps it."""
    monkeypatch.setenv("TRITON_CACHE_DIR", "/deliberate/path")
    assert _call() == "/deliberate/path"


def test_an_unwritable_tmpdir_does_not_kill_the_run(clean_env, monkeypatch):
    """A cache directory is never worth failing a run over."""
    monkeypatch.setenv("TMPDIR", "/proc/nonexistent-and-unwritable")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    assert _call() is None  # left unset, run continues


def test_setup_enviroment_calls_it(clean_env, monkeypatch):
    """It must run on the real startup path, after RANK is known."""
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    main = importlib.import_module("main")
    main.setup_enviroment()
    assert os.environ.get(
        "TRITON_CACHE_DIR"
    ), "setup_enviroment did not isolate the Triton cache"
