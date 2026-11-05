"""Pytest configuration for the repo.

Two jobs:

1. Put the repo root on `sys.path`. Tests import `src.*`, and `tests/` is not a package,
   so without this only `python -m pytest` works (it injects the cwd) while a bare
   `pytest` fails collection with `ModuleNotFoundError: No module named 'src'`.

2. Register the markers that split the suite by what it needs. The default run is
   everything that needs neither a GPU nor cluster data; CI runs `-m "not gpu and not slow"`.
"""

import os
import socket
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Registers `base_schema` in Hydra's ConfigStore; tests compose runnable configs
# whose defaults lists reference it. Must follow the sys.path setup above.
import src.definitions  # noqa: E402,F401


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a CUDA device")
    config.addinivalue_line("markers", "slow: takes more than a few seconds")
    config.addinivalue_line("markers", "network: needs to reach the HuggingFace hub")


def free_port():
    """A port the OS says is free right now.

    The distributed tests and the subprocess smoke tests each rendezvous on a
    TCP port. A fixed number works on a laptop and collides on a shared cluster
    node -- with other users, and with the previous test's socket in TIME_WAIT.
    Found when 18 tests failed on entropy's h100a and none failed locally.
    """
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def spawn_on_free_port(fn, make_args, nprocs, attempts=4):
    """`mp.spawn` on a free port, retrying when the port is taken anyway.

    `free_port()` asks the OS for a port, closes the socket, and hands the
    number to a child that binds it a moment later. Between those two instants
    anything else on the machine can take it -- and on a shared node, with
    several jobs each spawning several ranks, something does. The failure is
    `DistNetworkError ... EADDRINUSE` out of `init_process_group`, it looks
    exactly like a correctness failure in whichever test drew the short straw,
    and it cost one gate run on 2026-11-05 (job 32836, where
    `test_fusing_the_qk_pair_changes_nothing[4]` "failed" without ever
    running its assertions).

    The race cannot be closed by holding the socket open -- the child has to
    bind the same port -- so the honest fix is to notice and retry with a new
    one. Only EADDRINUSE is retried; every other error propagates on the first
    attempt, because the point is to stop infrastructure noise masquerading as
    a test result, not to paper over flaky code.

    `make_args` is a callable taking the port and returning the args tuple, so
    the port can sit anywhere in the signature.
    """
    import torch.multiprocessing as mp

    for attempt in range(attempts):
        try:
            mp.spawn(fn, args=make_args(free_port()), nprocs=nprocs, join=True)
            return
        except Exception as exc:  # noqa: BLE001 -- re-raised unless EADDRINUSE
            if "EADDRINUSE" not in str(exc) or attempt == attempts - 1:
                raise
