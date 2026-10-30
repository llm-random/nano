#!/usr/bin/env bash
# Point git at .githooks, without needing pixi.
#
# There was a catch-22 here. The only documented way to install the hook was
# `pixi run install_hooks`, but pixi.toml declares requires-pixi to stop a too-new pixi
# rewriting pixi.lock to v7 -- and that bound refuses *every* pixi command in the repo,
# `pixi run install_hooks` among them. So on pixi 0.81:
#
#   $ pixi run install_hooks
#   Error: this project requires pixi '>=0.60,<0.77', but you have pixi 0.81.0
#
# The hook's main job is catching a lock that a too-new pixi rewrote. The people who
# need it are exactly the people running a too-new pixi, and they were the ones who
# could not install it.
#
#   scripts/install_hooks.sh
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

if [[ ! -x .githooks/pre-commit ]]; then
    echo "No executable .githooks/pre-commit here; nothing to install." >&2
    exit 1
fi

git config --local core.hooksPath .githooks
echo "core.hooksPath -> $(git config --get core.hooksPath)"
echo "Installed. The hook checks pixi.lock is v6 and runs black."
