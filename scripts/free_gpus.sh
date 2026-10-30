#!/usr/bin/env bash
# List free GPUs on the clusters we have access to: entropy, lem and helios.
#
#   scripts/free_gpus.sh                    # all three, per-partition summary
#   scripts/free_gpus.sh lem helios         # only these
#   scripts/free_gpus.sh -n entropy         # per-node detail
#   ENTROPY_HOST=entropy scripts/free_gpus.sh entropy    # override the ssh alias
#
# Free is computed per node as CfgTRES minus AllocTRES, not from sinfo node states: a
# MIXED node has some GPUs free and sinfo will not tell you how many, while a node that
# is IDLE but DRAINing has none you can take. GPUs on nodes that are DOWN/DRAIN/MAINT/
# RESERVED/not-responding are counted under DOWN and never under FREE.
#
# A node in more than one partition is counted in each partition's row (same convention
# as sinfo), so partition rows can sum to more than the cluster's physical GPU count.
# The TOTAL line counts every node once.
#
# FREE is an upper bound on what you can actually get:
#   * entropy's QoS caps GPUs per account and that account is shared with everyone else
#     on it, so a job can pend with QOSMaxGRESPerUser while a node sits half idle;
#   * a job can pend on RAM while GPUs are free (entropy defaults to 90G per GPU).
# The "yours" line reports what you are already holding against that cap. Before
# submitting, probe the real start time with:  sbatch --test-only ... --wrap=true
set -uo pipefail

CLUSTERS_DEFAULT=(entropy lem helios)
TIMEOUT="${TIMEOUT:-60}"
MODE=summary

usage() {
    sed -n '2,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

# Default ssh aliases. entropy defaults to entropy_j because the `entropy` alias
# (user mstefaniak) has been failing publickey; override with ENTROPY_HOST if that
# changes. An unknown cluster name is passed through as an ssh alias as-is.
host_for() {
    case "$1" in
        entropy) echo "${ENTROPY_HOST:-entropy_j}" ;;
        lem)     echo "${LEM_HOST:-lem}" ;;
        helios)  echo "${HELIOS_HOST:-helios}" ;;
        *)       echo "$1" ;;
    esac
}

CLUSTERS=()
while [ $# -gt 0 ]; do
    case "$1" in
        -n|--nodes)   MODE=nodes ;;
        -t|--timeout) shift; TIMEOUT="${1:?-t needs seconds}" ;;
        -h|--help)    usage 0 ;;
        -*)           echo "unknown option: $1" >&2; usage 2 ;;
        *)            CLUSTERS+=("$1") ;;
    esac
    shift
done
[ ${#CLUSTERS[@]} -eq 0 ] && CLUSTERS=("${CLUSTERS_DEFAULT[@]}")

# The remote half. Sent over stdin rather than as an argument: quoting an awk program
# through ssh -> bash -lc mangles it on parentheses and braces.
read -r -d '' REMOTE <<'REMOTE_EOF'
command -v scontrol >/dev/null 2>&1 || { echo "FATAL|no slurm on this host"; exit 3; }

scontrol show node --oneliner 2>/dev/null | awk '
# scontrol --oneliner is Key=Value pairs, but some values contain spaces (OS=, Reason=),
# so pull out only the keys we need by regex instead of splitting the whole line.
function getv(s, key,   v) {
    if (match(s, key "=[^ ]*"))
        return substr(s, RSTART + length(key) + 1, RLENGTH - length(key) - 1)
    return ""
}
function gpus(tres,   n, a, i, k) {          # untyped total: gres/gpu=8
    n = split(tres, a, ",")
    for (i = 1; i <= n; i++)
        if (a[i] ~ /^gres\/gpu=/) { split(a[i], k, "="); return k[2] + 0 }
    return 0
}
function gtype(tres, gres,   n, a, i, k, m, b) {
    n = split(tres, a, ",")                  # typed TRES: gres/gpu:h100=8
    for (i = 1; i <= n; i++)
        if (a[i] ~ /^gres\/gpu:/) { split(a[i], k, "="); return substr(k[1], 10) }
    m = split(gres, b, ",")                  # else the Gres= field: gpu:hopper:4(S:0-1)
    for (i = 1; i <= m; i++)
        if (b[i] ~ /^gpu:/) {
            split(b[i], k, ":")
            if (k[2] !~ /^[0-9]+$/) return k[2]
        }
    return "-"
}
{
    cfg   = getv($0, "CfgTRES")
    total = gpus(cfg)
    if (total == 0) next                     # CPU-only node, not our business
    node  = getv($0, "NodeName")
    state = getv($0, "State")
    alloc = gpus(getv($0, "AllocTRES"))
    type  = gtype(cfg, getv($0, "Gres"))
    # A node in any of these states has nothing to give, whatever the TRES arithmetic
    # says. The trailing * means the node is not responding.
    down = (state ~ /DOWN|DRAIN|FAIL|MAINT|RESERV|INVAL|POWER|UNKNOWN|\*/)
    free = down ? 0 : total - alloc
    if (free < 0) free = 0
    n = split(getv($0, "Partitions"), parts, ",")
    for (i = 1; i <= n; i++)
        printf "NODE|%s|%s|%d|%d|%d|%d|%s|%s\n", parts[i], type, total, alloc, free,
               (down ? total - alloc : 0), state, node
}'

# What you are already holding against the shared cap. %b is TRES-per-node and its
# spelling varies between slurm versions (gres:gpu:2, gres/gpu:h100=2, gpu:2), so take
# the last number in it and multiply by the node count.
squeue -h -u "$USER" -t RUNNING -o '%D|%b' 2>/dev/null | awk -F'|' '
    { g = 0; if (match($2, /[0-9]+$/)) g = substr($2, RSTART) + 0; n += $1 * g }
    END { printf "SELF|%d\n", n + 0 }'
squeue -h -u "$USER" -t PENDING -o '%i' 2>/dev/null | awk 'END { printf "PEND|%d\n", NR + 0 }'
REMOTE_EOF

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Query the clusters in parallel; each is a round trip of several seconds and they do
# not depend on each other.
for c in "${CLUSTERS[@]}"; do
    (
        timeout "$TIMEOUT" ssh -o BatchMode=yes -o ConnectTimeout=10 \
            "$(host_for "$c")" bash -s <<< "$REMOTE" > "$TMP/$c.out" 2> "$TMP/$c.err"
        echo $? > "$TMP/$c.rc"
    ) &
done
wait

render() {   # $1 = cluster name
    local c="$1" host rc
    host=$(host_for "$c")
    rc=$(cat "$TMP/$c.rc" 2>/dev/null || echo 255)

    printf '\n\033[1m%s\033[0m (%s)\n' "$c" "$host"
    if [ "$rc" -ne 0 ] || ! [ -s "$TMP/$c.out" ]; then
        local why
        case "$rc" in
            124) why="timed out after ${TIMEOUT}s" ;;
            255) why="ssh failed: $(head -1 "$TMP/$c.err" 2>/dev/null)" ;;
            3)   why="no slurm on the login node" ;;
            *)   why="exit $rc: $(head -1 "$TMP/$c.err" 2>/dev/null)" ;;
        esac
        printf '  unreachable -- %s\n' "${why:-no output}"
        return
    fi

    if [ "$MODE" = nodes ]; then
        awk -F'|' '
            $1 == "NODE" { printf "  %-22s %-10s %5d %6d %6d  %s\n", $9, $3, $6, $5, $4, $8 }
        ' "$TMP/$c.out" | sort -u -k1,1 | \
            { printf '  %-22s %-10s %5s %6s %6s  %s\n' NODE TYPE FREE ALLOC TOTAL STATE; cat; }
    else
        printf '  %-26s %-10s %5s %6s %6s %5s  %s\n' \
               PARTITION TYPE FREE ALLOC TOTAL DOWN NODES
        awk -F'|' '
            $1 == "NODE" {
                k = $2 "|" $3
                tot[k] += $4; alc[k] += $5; fre[k] += $6; dwn[k] += $7; nod[k]++
                if ($6 > 0) hasfree[k]++
                if (!(seen[$9]++)) { ctot += $4; calc += $5; cfre += $6; cdwn += $7 }
            }
            END {
                for (k in tot) {
                    split(k, p, "|")
                    # free desc, then partition name: the row you want is the top one
                    printf "%09d|  %-26s %-10s %5d %6d %6d %5d  %d/%d free\n",
                           99999999 - fre[k], p[1], p[2], fre[k], alc[k], tot[k], dwn[k],
                           hasfree[k] + 0, nod[k]
                }
                printf "999999999|  %-26s %-10s %5d %6d %6d %5d\n",
                       "TOTAL (nodes counted once)", "", cfre, calc, ctot, cdwn
            }' "$TMP/$c.out" | sort -t'|' -k1,1 -k2,2 | cut -d'|' -f2-
    fi

    local mine pend
    mine=$(awk -F'|' '$1 == "SELF" { print $2 }' "$TMP/$c.out")
    pend=$(awk -F'|' '$1 == "PEND" { print $2 }' "$TMP/$c.out")
    printf '  yours: %s GPUs running, %s jobs pending\n' "${mine:-?}" "${pend:-?}"
}

for c in "${CLUSTERS[@]}"; do render "$c"; done
printf '\nFREE is an upper bound: QoS caps (shared account on entropy) and RAM can still
hold a job in the queue. Probe before submitting: sbatch --test-only ... --wrap=true\n'
