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
