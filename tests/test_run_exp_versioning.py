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


# ---------------------------------------------------------------------------
# A remote command that fails for a known reason should say so.
#
# fabric raises UnexpectedExit carrying the exit status and nothing else, so a cluster
# home over its disk quota surfaces as `exit 128` from git clone with the quota message
# buried in remote stderr. configs/_cluster/lem.yaml carries a comment about exactly
# that, which is a sign it cost somebody an afternoon. It cost one again today.


class _FakeResult:
    def __init__(self, failed=False, exited=0, stdout="", stderr=""):
        self.failed = failed
        self.exited = exited
        self.stdout = stdout
        self.stderr = stderr


class _FakeConnection:
    host = "lem"

    def __init__(self, result):
        self._result = result
        self.commands = []

    def run(self, command, warn=False):
        self.commands.append(command)
        return self._result


def test_run_remote_returns_the_result_when_the_command_works():
    from run_exp import run_remote

    ok = _FakeResult()
    conn = _FakeConnection(ok)
    assert run_remote(conn, "mkdir -p /x", "Creating /x") is ok
    assert conn.commands == ["mkdir -p /x"]


@pytest.mark.parametrize(
    "message",
    [
        "fatal: could not create work tree dir: Disk quota exceeded",
        "cp: writing: No space left on device",
        "mkdir: cannot create directory: Disk quota exceeded",
    ],
)
def test_run_remote_names_a_full_disk(message):
    """The point: 'exit 128' becomes a sentence you can act on."""
    from run_exp import run_remote

    conn = _FakeConnection(_FakeResult(failed=True, exited=128, stderr=message))
    with pytest.raises(RuntimeError) as excinfo:
        run_remote(conn, "git clone ...", "Cloning the experiment")

    text = str(excinfo.value)
    assert "out of space or over its disk quota" in text
    assert "lem" in text, "the error does not say which host"
    assert "cluster_cemetery_dir" in text, "no pointer to the setting that moves it"


def test_run_remote_still_reports_an_unrecognised_failure():
    """Anything else must not be swallowed, and must carry the remote's own words."""
    from run_exp import run_remote

    conn = _FakeConnection(
        _FakeResult(failed=True, exited=1, stderr="fatal: repository not found")
    )
    with pytest.raises(RuntimeError) as excinfo:
        run_remote(conn, "git clone ...", "Cloning the experiment")

    text = str(excinfo.value)
    assert "exit 1" in text
    assert "repository not found" in text
    assert "disk quota" not in text.lower()
