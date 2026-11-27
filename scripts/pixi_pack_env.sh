#!/usr/bin/env bash
# Pack a pixi.lock's linux-64 env into one tar on shared storage (documents/notes/lem_packed_pixi_env.md).
#
#   scripts/pixi_pack_env.sh <dir holding pixi.lock> <pack dir> [--force]
#
# Writes <pack dir>/nano-env-<hash>.tar and its unpacker nano-env-<hash>.pixi-unpack,
# <hash> = first 16 hex of sha256(pixi.lock), the same as resolver.pixi_lock_hash().
# Downloads, caches and the test unpack live in a node-local temp dir removed on exit.
set -euo pipefail

PIXI_PACK_VERSION=0.7.12
# GitHub's digest for that release's pixi-unpack-x86_64-unknown-linux-musl
PIXI_UNPACK_SHA256=70bddd77078ac2fe320075f8a03e7731e67e0583f732034ff2d7004b0a0b817c

die() { echo "pixi_pack_env: $*" >&2; exit 1; }

[[ $# -ge 2 ]] || die "usage: $0 <dir holding pixi.lock> <pack dir> [--force]"
MANIFEST_DIR=$(cd "$1" && pwd)
PACK_DIR=$2
FORCE=${3:-}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

PIXI=$(command -v pixi || echo "$HOME/.pixi/bin/pixi")
[[ -x $PIXI ]] || die "pixi not found on PATH or in ~/.pixi/bin"
[[ -f $MANIFEST_DIR/pixi.lock ]] || die "no pixi.lock in $MANIFEST_DIR"

HASH=$(sha256sum "$MANIFEST_DIR/pixi.lock" | cut -c1-16)
OUT=$PACK_DIR/nano-env-$HASH.tar
UNPACKER=$PACK_DIR/nano-env-$HASH.pixi-unpack
echo "pixi.lock hash: $HASH"

if [[ -f $OUT && -x $UNPACKER && $FORCE != --force ]]; then
    echo "already packed: $OUT (pass --force to rebuild)"
    exit 0
fi
mkdir -p "$PACK_DIR"

# TMPDIR_LOCAL is the job's node-local NVMe on lem; nothing in WORK may land on Lustre.
SCRATCH_ROOT=${TMPDIR_LOCAL:-${TMPDIR:-/tmp}}
WORK=$(mktemp -d "$SCRATCH_ROOT/nano-pixi-pack.XXXXXX")
trap 'rm -rf "$WORK" "$OUT.partial" "$UNPACKER.partial"' EXIT
echo "scratch: $WORK"

export PIXI_CACHE_DIR=$WORK/cache/pixi RATTLER_CACHE_DIR=$WORK/cache/rattler UV_CACHE_DIR=$WORK/cache/uv
export PIP_CACHE_DIR=$WORK/cache/pip XDG_CACHE_HOME=$WORK/cache/xdg TMPDIR=$WORK/tmp
mkdir -p "$TMPDIR" "$WORK/wheels"

# 1. pixi-pack only packs wheels: build each sdist in the lock into a wheel, injected in step 2.
"$PIXI" exec --spec python=3.12 --spec pip --spec pyyaml -- python - "$MANIFEST_DIR/pixi.lock" "$WORK" <<'PY'
import hashlib, pathlib, re, subprocess, sys, urllib.request
import yaml

lock_path, work = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
lock = yaml.load(lock_path.read_text(), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
meta = {p["pypi"]: p for p in lock["packages"] if "pypi" in p}
urls = [e["pypi"] for e in lock["environments"]["default"]["packages"]["linux-64"] if "pypi" in e]
sdists = [u for u in urls if not u.endswith(".whl")]
print(f"{len(urls)} pypi packages, {len(sdists)} sdists to build: {[meta[u]['name'] for u in sdists]}")
# What step 4 checks the unpacked env against, so the check follows whatever this branch's lock pins.
(work / "pypi_pins.txt").write_text("".join(f"{meta[u]['name']}=={meta[u]['version']}\n" for u in urls))
(work / "sdists").mkdir()
for url in sdists:
    p = meta[url]
    src = work / "sdists" / url.rsplit("/", 1)[1]
    urllib.request.urlretrieve(url, src)
    if hashlib.sha256(src.read_bytes()).hexdigest() != p["sha256"]:
        sys.exit(f"sha256 mismatch for {url}")
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--quiet", "--no-deps",
                    "--wheel-dir", str(work / "wheels"), str(src)], check=True)
    stem = re.sub(r"[-_.]+", "_", p["name"]).lower() + f"-{p['version']}-"
    built = [w.name for w in (work / "wheels").glob("*.whl") if w.name.lower().startswith(stem)]
    if len(built) != 1:
        sys.exit(f"expected one wheel {stem}*.whl for {url}, found {built}")
    print(f"built {built[0]}")
PY

inject=()
for whl in "$WORK"/wheels/*.whl; do
    [[ -e $whl ]] && inject+=(--inject "$whl")
done

# 2. Pack: downloads go to WORK (pixi-pack's temp dir), the tar is written straight to the pack dir.
(cd "$MANIFEST_DIR" && "$PIXI" exec --spec "pixi-pack==$PIXI_PACK_VERSION" -- pixi-pack \
    --environment default --platform linux-64 --ignore-pypi-non-wheel \
    --output-file "$OUT.partial" "${inject[@]}")

# 3. The matching unpacker next to the tar: the static release binary (conda-forge's links the env's OpenSSL).
curl -sSfL -o "$UNPACKER.partial" \
    "https://github.com/Quantco/pixi-pack/releases/download/v$PIXI_PACK_VERSION/pixi-unpack-x86_64-unknown-linux-musl"
echo "$PIXI_UNPACK_SHA256  $UNPACKER.partial" | sha256sum -c --quiet || die "pixi-unpack checksum mismatch"
chmod +x "$UNPACKER.partial"
"$UNPACKER.partial" --version

mv -f "$OUT.partial" "$OUT"
mv -f "$UNPACKER.partial" "$UNPACKER"
echo "packed: $OUT ($(du -h "$OUT" | cut -f1))"

# 4. Test the published pack exactly as a job would; unpublish it if that fails.
verify() {
    # job shells run without set -eu; set -e is off under `if !` anyway, so each step checks its status
    set +euo pipefail
    export PIXI_PACK_DIR=$PACK_DIR PIXI_LOCK_HASH=$HASH PIXI_ENV_DIR=$WORK/verify
    source "$SCRIPT_DIR/pixi_unpack_env.sh" || return 1
    python - "$WORK/pypi_pins.txt" <<'PY' || return 1
import importlib.metadata, re, sys

norm = lambda name: re.sub(r"[-_.]+", "-", name).lower()
have = {norm(d.metadata["Name"]): d.version for d in importlib.metadata.distributions()}
pins = [line.split("==") for line in open(sys.argv[1]).read().split()]
wrong = [f"{n}=={v} (env has {have.get(norm(n))})" for n, v in pins if have.get(norm(n)) != v]
if wrong:
    sys.exit("not installed at the locked version: " + ", ".join(wrong))
import torch

print(f"all {len(pins)} locked pypi packages installed; torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
PY
    torchrun --help > /dev/null || return 1
    echo "torchrun ok: $(command -v torchrun)"
}
if ! (verify); then
    rm -f "$OUT" "$UNPACKER"
    die "test unpack failed, removed $OUT"
fi
echo "done: nano-env-$HASH"
