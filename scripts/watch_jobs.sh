#!/usr/bin/env bash
# Emit one line per state change and per terminal state for a set of slurm jobs,
# across more than one cluster, and exit when they are all finished.
#
#   watch_jobs.sh helios:25256461 helios:25256588 lem:6174094
#
# Design notes, each earned the hard way during this audit:
#  * Terminal state is inferred from ABSENCE in squeue, not from sacct. lem's accounting
#    database was unreachable for a stretch, which makes any sacct-based wait hang forever
#    without ever saying why.
#  * A job is not in squeue for the first few seconds after sbatch returns, so "absent"
#    only counts as finished once the job has been seen at least once.
#  * Failure is reported, not just success. A waiter that only matches the happy path is
#    silent through a crash, and silence looks exactly like "still running". The first
#    version of this list omitted pytest's own failure lines and duly reported a job
#    with a FAILED test in it as clean -- widen the alternation rather than narrow it.
#  * Every line is flushed as it is produced; no `head`, which would SIGPIPE the producer.
#  * A job that is never seen in the queue at all -- already finished, or a typo -- is
#    reported and dropped after MISSING_LIMIT polls instead of being waited on forever.
#    An unfindable job is a result too, and hanging on it is the same silent failure as
#    not reporting a crash.
# WATCH_LOG_DIRS: colon-separated directories to look for <name>_<jobid>.out in, so this
# is not tied to one experiment's layout. $USER is expanded on the remote side.
: "${WATCH_LOG_DIRS:=\$HOME:\$SCRATCH}"
set -uo pipefail

declare -A LAST SEEN MISSES
POLL="${POLL:-60}"
MISSING_LIMIT="${MISSING_LIMIT:-3}"

state_of() {  # $1=host $2=jobid -> slurm state, or "" if absent
  timeout 45 ssh -o BatchMode=yes "$1" \
    "squeue -j $2 -h -o %T 2>/dev/null" 2>/dev/null | tr -d '[:space:]'
}

log_errors() {  # $1=host $2=jobid -- surface the first real error line, once
  local LOG_GLOBS d
  LOG_GLOBS=""
  for d in $(echo "$WATCH_LOG_DIRS" | tr ':' ' '); do
    LOG_GLOBS="$LOG_GLOBS $d/*_$2.out"
  done
  timeout 45 ssh -o BatchMode=yes "$1" \
    "grep -ohE 'Traceback|RuntimeError|CUDA out of memory|OOM|srun: error|slurmstepd: error|^FAILED |[0-9]+ failed|Error executing job' \
     \$(ls -t $LOG_GLOBS 2>/dev/null | head -1) \
     2>/dev/null | sort -u | head -3" 2>/dev/null
}

for spec in "$@"; do LAST[$spec]=""; SEEN[$spec]=0; MISSES[$spec]=0; done
remaining=$#

while [ "$remaining" -gt 0 ]; do
  for spec in "$@"; do
    [ "${LAST[$spec]}" = "__done__" ] && continue
    host=${spec%%:*}; job=${spec##*:}
    st=$(state_of "$host" "$job")
    if [ -n "$st" ]; then
      SEEN[$spec]=1
      if [ "$st" != "${LAST[$spec]}" ]; then
        echo "[$(date +%H:%M:%S)] $spec -> $st"
        LAST[$spec]=$st
      fi
    elif [ "${SEEN[$spec]}" = "0" ]; then
      MISSES[$spec]=$(( ${MISSES[$spec]} + 1 ))
      if [ "${MISSES[$spec]}" -ge "$MISSING_LIMIT" ]; then
        echo "[$(date +%H:%M:%S)] $spec NOT IN QUEUE after ${MISSES[$spec]} polls -- already finished, or wrong id"
        LAST[$spec]="__done__"
        remaining=$((remaining-1))
      fi
    else
      errs=$(log_errors "$host" "$job")
      if [ -n "$errs" ]; then
        echo "[$(date +%H:%M:%S)] $spec FINISHED with errors in log:"
        echo "$errs" | sed 's/^/    /'
      else
        echo "[$(date +%H:%M:%S)] $spec FINISHED (no error signatures in log)"
      fi
      LAST[$spec]="__done__"
      remaining=$((remaining-1))
    fi
  done
  [ "$remaining" -gt 0 ] && sleep "$POLL"
done
echo "[$(date +%H:%M:%S)] all jobs finished"
