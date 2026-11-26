# Sourced by the lem job preamble: unpack this lock's packed pixi env to node-local disk on every node, then activate it.
# See documents/notes/lem_packed_pixi_env.md. Pack with update_pixi.py (scripts/pixi_pack_env.sh).
#
# Needs PIXI_PACK_DIR and PIXI_LOCK_HASH. PIXI_ENV_DIR defaults to $TMPDIR_LOCAL/nano_env.
# Sourced into exp.job: no set -e/-u (they would leak), and failures `exit` the job instead of returning.

_nano_die() { echo "pixi_unpack_env: $*" >&2; exit 1; }

[ -n "${PIXI_PACK_DIR:-}" ] && [ -n "${PIXI_LOCK_HASH:-}" ] || _nano_die "PIXI_PACK_DIR and PIXI_LOCK_HASH must be set"
_nano_tar=$PIXI_PACK_DIR/nano-env-$PIXI_LOCK_HASH.tar
_nano_unpacker=$PIXI_PACK_DIR/nano-env-$PIXI_LOCK_HASH.pixi-unpack
[ -f "$_nano_tar" ] && [ -x "$_nano_unpacker" ] \
    || _nano_die "no packed env for this pixi.lock ($PIXI_LOCK_HASH) in $PIXI_PACK_DIR; pack it with update_pixi.py"
if [ -z "${PIXI_ENV_DIR:-}" ]; then
    # WCSS gives multi-node jobs Lustre TMPDIR unless they ask for local disk.
    [ -n "${TMPDIR_LOCAL:-}" ] \
        || _nano_die "TMPDIR_LOCAL is unset, so the job has no node-local disk; add storage:local:40G to slurm gres"
    PIXI_ENV_DIR=$TMPDIR_LOCAL/nano_env
fi
export PIXI_ENV_DIR

# Multi-node: one unpack per node in a single step; the path is the same on every node.
_nano_on_nodes=
if [ "${SLURM_JOB_NUM_NODES:-1}" -gt 1 ]; then
    _nano_on_nodes="srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1"
    _nano_on_nodes+=" --cpus-per-task=${SLURM_CPUS_ON_NODE:-1} --kill-on-bad-exit=1"
fi

# Unpacking peaks near 3x the tar (extracted tar + env: 11.7G for a 3.9G tar); ask 4x for margin.
_nano_need_gb=$(( $(stat -c %s "$_nano_tar") * 4 / 1024 ** 3 + 1 ))
mkdir -p "$PIXI_ENV_DIR/tmp" || _nano_die "cannot create $PIXI_ENV_DIR"
_nano_free_gb=$(df -P -BG "$PIXI_ENV_DIR" | awk 'NR == 2 {print $4 + 0}')
[ "$_nano_free_gb" -ge "$_nano_need_gb" ] \
    || _nano_die "need ~${_nano_need_gb}G free under $PIXI_ENV_DIR, have ${_nano_free_gb}G; raise gres storage:local:<N>G"

# WCSS moves a finished job's TMPDIR to Lustre unless deleted; lem.yaml also sets --extra=FORCE_RM_TMPDIR.
trap "$_nano_on_nodes rm -rf '$PIXI_ENV_DIR'" EXIT
echo "pixi_unpack_env: unpacking $_nano_tar to $PIXI_ENV_DIR on ${SLURM_JOB_NUM_NODES:-1} node(s) (${_nano_free_gb}G free)"
_nano_t0=$SECONDS
$_nano_on_nodes bash -c 'mkdir -p "$1/tmp" && TMPDIR="$1/tmp" "$2" --output-directory "$1" --env-name env "$3" && rm -rf "$1/tmp"' \
    _ "$PIXI_ENV_DIR" "$_nano_unpacker" "$_nano_tar" \
    || _nano_die "pixi-unpack failed; if the disk filled up, raise gres storage:local:<N>G"
. "$PIXI_ENV_DIR/activate.sh" || _nano_die "sourcing $PIXI_ENV_DIR/activate.sh failed"
[ "$(command -v python)" = "$PIXI_ENV_DIR/env/bin/python" ] \
    || _nano_die "python resolves to $(command -v python), not the unpacked env"
echo "pixi_unpack_env: ready in $((SECONDS - _nano_t0))s, $(du -sh "$PIXI_ENV_DIR/env" | cut -f1) in $PIXI_ENV_DIR/env"

unset _nano_tar _nano_unpacker _nano_on_nodes _nano_need_gb _nano_free_gb _nano_t0
unset -f _nano_die
