"""run_exp.py's repo versioning must put the branch back, including when killed.

version_code() commits onto whatever branch you are standing on, creates an experiment
branch and checks it out. It restored in a `finally`, which covers exceptions and normal
exit but *not* a signal: the default SIGTERM handler kills the process outright, so a
cancelled or preempted submission left the repo on the experiment branch with the user's
own branch unrestored. That is a bad failure for a tool people run from their working
checkout, and it is recorded in documents/issues.md.
"""

import signal

import pytest

from run_exp import restoring_repo_state


class FakeRepo:
    """Records the restore instead of touching a real repository."""

    def __init__(self):
        self.restores = 0


@pytest.fixture
def restore_calls(monkeypatch):
    calls = []

    def fake_reset(repo, original_branch, original_commit, versioning_branch):
        calls.append((original_branch, original_commit, versioning_branch))

    monkeypatch.setattr("run_exp.reset_to_original_repo_state", fake_reset)
    return calls


def test_normal_exit_restores_once(restore_calls):
    with restoring_repo_state(FakeRepo(), "main", "abc123", "exp"):
        pass
    assert restore_calls == [("main", "abc123", "exp")]


def test_exception_still_restores(restore_calls):
    with pytest.raises(RuntimeError):
        with restoring_repo_state(FakeRepo(), "main", "abc123", "exp"):
            raise RuntimeError("push failed")
    assert restore_calls == [("main", "abc123", "exp")]


def test_sigterm_restores_before_dying(restore_calls, monkeypatch):
    """The regression this exists for: a signal must not skip the restore."""
    killed = []
    monkeypatch.setattr("run_exp.os.kill", lambda pid, sig: killed.append(sig))

    with restoring_repo_state(FakeRepo(), "main", "abc123", "exp"):
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler), "SIGTERM handler was not installed"
        handler(signal.SIGTERM, None)
        assert restore_calls == [
            ("main", "abc123", "exp")
        ], "the signal handler did not restore the repo"

    assert killed == [signal.SIGTERM], "the signal was swallowed instead of re-raised"
    # The finally must not restore a second time.
    assert len(restore_calls) == 1


def test_handlers_are_put_back_afterwards(restore_calls):
    before = signal.getsignal(signal.SIGTERM)
    with restoring_repo_state(FakeRepo(), "main", "abc123", "exp"):
        pass
    assert signal.getsignal(signal.SIGTERM) is before
