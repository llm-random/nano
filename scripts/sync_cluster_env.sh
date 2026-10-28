#!/usr/bin/env bash
# Sync this repo's pixi manifest and lock into a cluster's PIXI_HOME.
#
# Why this exists: every cluster sets PIXI_HOME=$PROJECT_HOME_PATH/$USER/pixi, a directory
# holding its *own* copy of pixi.toml and pixi.lock. The repo's lock is the source of
# truth, but nothing carries it across, so fixing the lock in a PR leaves every cluster
# environment exactly as it was. That is why `pytest` is declared in pixi.toml yet missing
# from the environments on lem and entropy.
#
# Dry run by default. It prints what differs and changes nothing until you pass --apply.
#
#   scripts/sync_cluster_env.sh /storage_nvme_4/nano/$USER/pixi
#   scripts/sync_cluster_env.sh /storage_nvme_4/nano/$USER/pixi --apply
#
# A cluster PIXI_HOME is often SHARED between people. Re-installing changes the
# environment under anyone else using it -- check before you pass --apply.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-${PIXI_HOME:-}}"
APPLY="${2:-}"

if [[ -z "$TARGET" ]]; then
    echo "usage: $0 <PIXI_HOME> [--apply]" >&2
    exit 2
fi
if [[ ! -d "$TARGET" ]]; then
    echo "error: '$TARGET' is not a directory" >&2
    exit 2
fi

echo "repo:   $REPO_ROOT"
echo "target: $TARGET"
echo

changed=0
for f in pixi.toml pixi.lock; do
    if [[ ! -f "$REPO_ROOT/$f" ]]; then
        echo "error: $REPO_ROOT/$f is missing" >&2
        exit 2
    fi
    if [[ ! -f "$TARGET/$f" ]]; then
        echo "$f: absent in the target, would be created"
        changed=1
    elif cmp -s "$REPO_ROOT/$f" "$TARGET/$f"; then
        echo "$f: identical"
    else
        echo "$f: DIFFERS"
        diff <(head -1 "$TARGET/$f") <(head -1 "$REPO_ROOT/$f") >/dev/null 2>&1 \
            || echo "  first line: target '$(head -1 "$TARGET/$f")' vs repo '$(head -1 "$REPO_ROOT/$f")'"
        changed=1
    fi
done

if [[ "$changed" -eq 0 ]]; then
    echo
    echo "Nothing to do."
    exit 0
fi

if [[ "$APPLY" != "--apply" ]]; then
    echo
    echo "Dry run. Re-run with --apply to copy these over and re-install."
    echo "Check first whether anyone else uses '$TARGET'."
    exit 0
fi

ts="$(date +%Y%m%d-%H%M%S)"
for f in pixi.toml pixi.lock; do
    [[ -f "$TARGET/$f" ]] && cp -p "$TARGET/$f" "$TARGET/$f.bak-$ts"
    cp -p "$REPO_ROOT/$f" "$TARGET/$f"
done
echo "copied (previous files kept as *.bak-$ts)"

cd "$TARGET"
# --frozen so the lock is used as given rather than re-solved, which is the whole point:
# a re-solve here would rewrite the lock to whatever the local pixi version emits, and
# pixi 0.81 writes v7, which helios's 0.67.1 cannot read.
pixi install --frozen
echo "installed"
