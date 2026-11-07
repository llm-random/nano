#!/bin/bash -l
# Keeps a copy of FineWeb on helios $SCRATCH, where opening it takes seconds instead of ~16 min.
#
# Why: datasets.load_from_disk walks every record-batch header before step 1 -- 12,990 files x
# ~100 headers = 1.3M small random reads per open -- and a nano job opens the dataset twice (the
# train and the eval split). Group storage answers a cold read in ~23 ms, so a helios job spent
# ~32 min before its first step. $SCRATCH answers in ~0.35 ms. Both measured 2026-11-07 from a
# GH200 node.
#
# Catch: $SCRATCH deletes files 30 days after they were created (/etc/motd). Each run therefore:
#   1. re-creates every file older than MAX_AGE_DAYS: copy to *.part, then rename over the old
#      file, so a job already reading the old file keeps its open copy;
#   2. verifies names, sizes and a checksum sample against the source;
#   3. re-creates the .complete stamp, dated to the copy's oldest file.
# configs/_cluster/helios.yaml reads the scratch copy only while that stamp is younger than 28
# days, and falls back to group storage otherwise. A failed or late refresh costs startup time,
# never a crash.
#
# Scheduling: helios refuses sbatch from inside a job, and crond does not run on its login nodes.
# So the refreshes are queued ahead of time, from your laptop, in the repo root:
#   ssh helios 'sbatch --qos=now' < scripts/helios_fineweb_scratch.sh  # copy now
#   for d in 14 28 42 56 70 84 98 112; do                              # refresh every 2 weeks
#       ssh helios "sbatch --begin=now+${d}days" < scripts/helios_fineweb_scratch.sh
#   done
# The queued runs use the default QoS. --qos=now allows one submitted job per user, and a job
# parked for weeks would hold that slot.
#
# Status:  ssh helios 'squeue -u $USER -n fineweb_scratch; cat $SCRATCH/datasets/fineweb/train/.complete'
# Logs:    $SCRATCH/fineweb_scratch-<jobid>.out on helios
# Stop:    ssh helios 'scancel -u $USER -n fineweb_scratch'
#
# Space: during a refresh, files that running jobs hold open keep their space until those jobs
# end. That is up to two copies (11.8 TiB) against the 12 TiB $SCRATCH quota.
#
#SBATCH --job-name=fineweb_scratch
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/net/scratch/hscra/plgrid/%u/fineweb_scratch-%j.out

set -uo pipefail
: "${SLURM_JOB_ID:?submit this with sbatch}"

SRC=${FINEWEB_SRC:-/net/storage/pr3/plgrid/plggllmeffi3/datasets/fineweb/train}
DST=${FINEWEB_DST:-$SCRATCH/datasets/fineweb/train}
MAX_AGE_DAYS=${MAX_AGE_DAYS:-2}  # only lets a resumed run skip files a recent run copied
STREAMS=${STREAMS:-16}
NAME=fineweb_scratch

log() { echo "[$(date '+%F %T')] $*"; }

# Two runs at once would delete each other's *.part files below.
others=$(squeue -h -u "$USER" -n "$NAME" -t RUNNING -o %i | grep -vx "$SLURM_JOB_ID")
if [ -n "$others" ]; then
    log "job $others is already running; leaving the copy to it"
    exit 0
fi

log "copy $SRC -> $DST: re-create files older than $MAX_AGE_DAYS days, $STREAMS streams"
mkdir -p "$DST" || exit 1
find "$DST" -maxdepth 1 -name '*.part' -delete  # left by a killed run

copy_one() {  # re-create $DST/$1 unless it already has the source's size and is recent
    local s="$SRC/$1" d="$DST/$1"
    if [ -f "$d" ] && [ "$(stat -c %s "$s")" = "$(stat -c %s "$d")" ] \
        && [ -n "$(find "$d" -mtime "-$MAX_AGE_DAYS")" ]; then
        return 0
    fi
    # Plain cp, never -p or rsync -a: the purge goes by age, so the copy needs fresh timestamps.
    cp "$s" "$d.part" && mv -f "$d.part" "$d"
}
export -f copy_one
export SRC DST MAX_AGE_DAYS

t0=$(date +%s)
find "$SRC" -maxdepth 1 -type f -printf '%f\0' | xargs -0 -P "$STREAMS" -I{} bash -c 'copy_one "$1"' _ {}
copy_rc=$?
log "copy pass: $(($(date +%s) - t0)) s, rc $copy_rc"

data_files() {  # every file of the dataset copy in $1, without the stamp and partial copies
    find "$1" -maxdepth 1 -type f ! -name '.complete*' ! -name '*.part' "${@:2}"
}
fail=0
if ! mismatch=$(diff <(data_files "$SRC" -printf '%f %s\n' | sort) <(data_files "$DST" -printf '%f %s\n' | sort)); then
    log "FAIL: names or sizes differ:"
    echo "$mismatch" | head -n 20
    fail=1
fi
bad=$({ printf '%s\n' state.json dataset_info.json; find "$SRC" -maxdepth 1 -name '*.arrow' -printf '%f\n' | shuf -n 16; } \
    | xargs -P 8 -I{} bash -c '[ "$(md5sum < "$SRC/$1")" = "$(md5sum < "$DST/$1")" ] || echo "$1"' _ {})
if [ -n "$bad" ]; then
    log "FAIL: checksums differ: $bad"
    fail=1
fi
if [ "$copy_rc" -ne 0 ] || [ "$fail" -ne 0 ]; then
    log "FAILED: the stamp is unchanged, so jobs keep the previous verdict"
    exit 1
fi

# The stamp's mtime is the oldest file's: that file is the first the purge takes, so this is
# the age helios.yaml must check. A run that skipped recent files must not make it look younger.
# It is a new file each time, not a touch: if the purge goes by creation time, a touched stamp
# would still be removed 30 days after it was first made.
oldest=$(data_files "$DST" -printf '%T@\n' | sort -n | head -n 1)
oldest=${oldest%.*}
{
    echo "verified $(date -Is) by job $SLURM_JOB_ID"
    echo "source $SRC"
    echo "files $(data_files "$DST" | wc -l), $(du -s --apparent-size --block-size=1G "$DST" | cut -f1) GiB"
    echo "oldest file $(date -d "@$oldest" -Is); helios.yaml trusts this copy for 28 days from then"
} > "$DST/.complete.tmp" \
    && touch -m -d "@$oldest" "$DST/.complete.tmp" \
    && mv -f "$DST/.complete.tmp" "$DST/.complete" \
    || { log "FAIL: could not write the stamp"; exit 1; }
log "verified; stamped $DST/.complete (oldest file $(date -d "@$oldest" -Is))"
