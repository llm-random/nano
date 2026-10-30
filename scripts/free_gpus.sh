#!/usr/bin/env bash
# List free GPUs on the clusters we have access to: entropy, lem and helios.
#
#   scripts/free_gpus.sh                    # all three, per-partition summary
#   scripts/free_gpus.sh lem helios         # only these
#   scripts/free_gpus.sh -1                 # ALL CLUSTERS IN ONE TABLE, free first
#   scripts/free_gpus.sh -1 -r              # ... plus the biggest 1-GPU request that fits
#   scripts/free_gpus.sh -a                 # every partition, no filtering
#   scripts/free_gpus.sh -n entropy         # per-node detail
#   ENTROPY_HOST=entropy scripts/free_gpus.sh entropy    # override the ssh alias
#
# By default only the partitions worth submitting a nano job to are shown:
#   entropy  all but `common` -- Titan V/X and 2080 Ti, where the pixi torch build has no
#            sm_70/sm_52 kernels and /storage_nvme_* is not even mounted
#   lem      plgrid-lem-gpu-h100 only, not the 4-GPU lem-gpu-interactive queue
#   helios   8k3-gpu-gh200 and plgrid-gpu-gh200 only
# The whole-cluster TOTAL row is dropped with them, since it would still be counting the
# hidden partitions. Pass -a/--all for every GPU partition and the totals; override a
# cluster's rule with ENTROPY_FILTER / LEM_FILTER / HELIOS_FILTER ("keep|re" or "drop|re").
#
# -r/--resources adds two columns: for partitions that have a free GPU, the largest CPU
# and RAM a ONE-GPU job can ask for and still FIT. Fitting is not the same as starting:
# on lem every --test-only probe came back at now+46min, even a 1-CPU/1G/1-min job pinned
# to a node with 1 GPU, 40 CPU and 495G free, so the site gates on more than capacity.
# The number is what fits; sbatch --test-only is the only authority on when it runs.
# The figure is the free CPU/RAM of one node -- the one with most free RAM, then most free CPU, so the pair is jointly
# satisfiable, unlike the per-column maxima -- then capped by the site's per-GPU limit:
#
#   lem     16 CPU / 250G per gpu:hopper, enforced at submit: asking for a node's full
#           64 CPU / 1005G is REJECTED outright, not queued (verified with --test-only)
#   helios  72 CPU / 119G, a quarter of a 288-CPU/489500M node per GH200
#   entropy no per-GPU cap, so only node headroom binds
#
# Override with LEM_GPU_CAP / HELIOS_GPU_CAP / ENTROPY_GPU_CAP as "cpu:memG" ("" = none).
# A trailing + on CPU means another node with a free GPU has more CPU but less RAM
# (-n -r lists every node). QoS caps on *concurrent* GPUs are a separate limit again.
#
# Free is computed per node as CfgTRES minus AllocTRES, not from sinfo node states: a
# MIXED node has some GPUs free and sinfo will not tell you how many, while a node that
# is IDLE but DRAINing has none you can take. GPUs on nodes that are DOWN/DRAIN/MAINT/
# not-responding are counted under DOWN and never under FREE.
#
# HELD is the third bucket: nodes that are up and have idle GPUs, but are PLANNED or
# RESERVED -- backfill is holding them for a higher-priority job, so those GPUs are not
# yours either. They were counted as FREE until sbatch --test-only was asked to confirm
# it: helios showed "5 free" while slurm quoted a week's wait, and lem's "free" GPUs were
# quoted 45 minutes out. Only a short job that fits in the gap can backfill into HELD.
#
# A node in more than one partition is counted in each partition's row (same convention
# as sinfo), so partition rows can sum to more than the cluster's physical GPU count.
# The TOTAL line counts every node once. In -1 the double counting is collapsed instead:
# partitions holding exactly the same nodes become one row, named after the plgrid-*
# alias (the one a grant actually submits to) with +N for the other names.
#
# FREE is an upper bound on what you can actually get:
#   * entropy's QoS caps GPUs per account and that account is shared with everyone else
#     on it, so a job can pend with QOSMaxGRESPerUser while a node sits half idle;
#   * a job can pend on RAM while GPUs are free (entropy defaults to 90G per GPU).
# The "<login> holds:" line reports what is already held against that cap -- named by
# remote login, because entropy's j321m is shared and some of it may be other people's.
# Before submitting, probe the real start time with:  sbatch --test-only ... --wrap=true
set -uo pipefail

CLUSTERS_DEFAULT=(entropy lem helios)
TIMEOUT="${TIMEOUT:-60}"
MODE=summary
ALL=0
RES=0

usage() {   # the header comment is the help text, however long it grows
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"
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

# Which partitions are worth showing, per cluster: "keep|<re>" shows only partitions
# matching <re>, "drop|<re>" shows all but those. See the header for the reasoning.
filter_for() {
    case "$1" in
        entropy) echo "${ENTROPY_FILTER:-drop|^common$}" ;;
        lem)     echo "${LEM_FILTER:-drop|^lem-gpu-interactive$}" ;;
        helios)  echo "${HELIOS_FILTER:-keep|^(8k3-gpu-gh200|plgrid-gpu-gh200)$}" ;;
        *)       echo "keep|." ;;
    esac
}

# Per-GPU request caps as "cpu:memG" -- what one GPU entitles you to ask for, which is
# not the same as what is physically free on the node. lem's is enforced by a submit
# plugin (it prints these exact numbers when it rejects you); helios's is the quarter
# node its `now` QoS spells out; entropy has none.
gpu_cap_for() {   # $1 = cluster -> "cpu:memG", empty when uncapped
    case "$1" in
        lem)     echo "${LEM_GPU_CAP-16:250}" ;;
        helios)  echo "${HELIOS_GPU_CAP-72:119}" ;;
        entropy) echo "${ENTROPY_GPU_CAP-}" ;;
        *)       echo "" ;;
    esac
}

cap_cpu() { local c; c=$(gpu_cap_for "$1"); echo "${c%%:*}" ; }
cap_mem() { local c; c=$(gpu_cap_for "$1"); c=${c#*:}; [ -n "$c" ] && echo $(( c * 1024 )) || echo "" ; }

# Writes the node rows to render for one cluster. Filtering here rather than on the
# remote side keeps -a a re-render instead of a second round of ssh.
apply_filter() {   # $1 = cluster
    local rule mode re want
    rule=$(filter_for "$1"); mode=${rule%%|*}; re=${rule#*|}
    if [ "$ALL" = 1 ]; then
        grep '^NODE|' "$TMP/$1.out" > "$TMP/$1.rows" 2>/dev/null
        return
    fi
    # keep-mode keeps the partitions that match, drop-mode keeps the ones that do not
    [ "$mode" = keep ] && want=1 || want=0
    # NODE|partition|type|... -- partition is $2 here; it is $3 only in the one-table
    # pass, where every line carries a cluster prefix.
    awk -F'|' -v want="$want" -v re="$re" \
        '$1 == "NODE" && (($2 ~ re) == want)' "$TMP/$1.out" > "$TMP/$1.rows"
}

# Colour only for a terminal: this gets piped into files and greps often enough that
# bare escape codes in the output would be a nuisance.
if [ -t 1 ]; then BOLD=$'\033[1m'; RESET=$'\033[0m'; else BOLD=""; RESET=""; fi

CLUSTERS=()
while [ $# -gt 0 ]; do
    case "$1" in
        -n|--nodes)   MODE=nodes ;;
        -1|--one-table) MODE=one ;;
        -a|--all)     ALL=1 ;;
        -r|--resources) RES=1 ;;
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
function tres(t, key,   n, a, i, k) {        # plain TRES field: cpu=256, mem=1031600M
    n = split(t, a, ",")
    for (i = 1; i <= n; i++) { split(a[i], k, "="); if (k[1] == key) return k[2] }
    return ""
}
function mb(v,   u) {                        # slurm writes mem with a unit suffix
    if (v == "") return 0
    u = substr(v, length(v), 1)
    if (u == "G") return (v + 0) * 1024
    if (u == "T") return (v + 0) * 1048576
    if (u == "K") return (v + 0) / 1024
    return v + 0                             # M, or already bare megabytes
}
function gtype(tres, gres,   n, a, i, k, m, b) {
    n = split(tres, a, ",")                  # typed TRES: gres/gpu:h100=8
    for (i = 1; i <= n; i++)
        if (a[i] ~ /^gres\/gpu:/) { split(a[i], k, "="); return substr(k[1], 10) }
    m = split(gres, b, ",")                  # else the Gres= field: gpu:hopper:4(S:0-1)
    for (i = 1; i <= m; i++)
        if (b[i] ~ /^gpu:/) {
            split(b[i], k, ":")
            # gpu:4(S:0-3) is untyped with socket info, gpu:a100:8(S:0-1) is typed;
            # drop the "(S:..." tail before deciding which of the two this is.
            sub(/\(.*/, "", k[2])
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
    # Three buckets, not two. A broken node has nothing to give; a PLANNED or RESERVED
    # one is working and idle but backfill is holding it for a higher-priority job, so
    # its GPUs are not yours either -- counting them as free is what made this script
    # claim 5 free GPUs on helios while sbatch --test-only quoted a week of waiting.
    down = (state ~ /DOWN|DRAIN|FAIL|MAINT|INVAL|POWER|UNKNOWN|NOT_RESPONDING|\*/)
    held = (!down && state ~ /PLANNED|RESERV/)
    free = (down || held) ? 0 : total - alloc
    if (free < 0) free = 0
    # What a job landing here could still ask for, which is what actually decides
    # whether a 1-GPU request starts now or pends on RAM.
    at   = getv($0, "AllocTRES")
    fcpu = tres(cfg, "cpu") - tres(at, "cpu")
    fmem = mb(tres(cfg, "mem")) - mb(tres(at, "mem"))
    if (fcpu < 0) fcpu = 0
    if (fmem < 0) fmem = 0
    n = split(getv($0, "Partitions"), parts, ",")
    # NODE|partition|type|total|alloc|free|down|state|node|freecpu|freemem|held
    # New fields go on the end so field numbers in the readers stay put.
    for (i = 1; i <= n; i++)
        printf "NODE|%s|%s|%d|%d|%d|%d|%s|%s|%d|%d|%d\n", parts[i], type, total, alloc,
               free, (down ? total - alloc : 0), state, node, fcpu, fmem,
               (held ? total - alloc : 0)
}'

# What you are already holding against the shared cap. %b is TRES-per-node and its
# spelling varies between slurm versions (gres:gpu:2, gres/gpu:h100=2, gpu:2), so take
# the last number in it and multiply by the node count.
squeue -h -u "$USER" -t RUNNING -o '%D|%b' 2>/dev/null | awk -F'|' '
    { g = 0; if (match($2, /[0-9]+$/)) g = substr($2, RSTART) + 0; n += $1 * g }
    END { printf "SELF|%d\n", n + 0 }'
squeue -h -u "$USER" -t PENDING -o '%i' 2>/dev/null | awk 'END { printf "PEND|%d\n", NR + 0 }'
echo "USER|$USER"
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

ok_cluster() {   # $1 = cluster -- did the query come back with usable output?
    local rc
    rc=$(cat "$TMP/$1.rc" 2>/dev/null || echo 255)
    [ "$rc" -eq 0 ] && [ -s "$TMP/$1.out" ]
}

fail_reason() {   # $1 = cluster
    local rc
    rc=$(cat "$TMP/$1.rc" 2>/dev/null || echo 255)
    case "$rc" in
        0)   echo "no output" ;;
        124) echo "timed out after ${TIMEOUT}s" ;;
        255) echo "ssh failed: $(head -1 "$TMP/$1.err" 2>/dev/null)" ;;
        3)   echo "no slurm on the login node" ;;
        *)   echo "exit $rc: $(head -1 "$TMP/$1.err" 2>/dev/null)" ;;
    esac
}

holds_line() {   # $1 = cluster, $2 = indent
    local mine pend who
    mine=$(awk -F'|' '$1 == "SELF" { print $2 }' "$TMP/$1.out")
    pend=$(awk -F'|' '$1 == "PEND" { print $2 }' "$TMP/$1.out")
    who=$(awk -F'|' '$1 == "USER" { print $2 }' "$TMP/$1.out")
    printf '%s%s holds: %s GPUs running, %s jobs pending\n' \
           "$2" "${who:-login}" "${mine:-?}" "${pend:-?}"
}

render() {   # $1 = cluster name
    local c="$1" host
    host=$(host_for "$c")

    printf '\n%s%s%s (%s)\n' "$BOLD" "$c" "$RESET" "$host"
    if ! ok_cluster "$c"; then
        printf '  unreachable -- %s\n' "$(fail_reason "$c")"
        return
    fi

    if [ "$MODE" = nodes ]; then
        awk -F'|' -v res="$RES" '
            $1 == "NODE" {
                if (res)
                    printf "  %-22s %-10s %5d %6d %6d %6d %8s  %s\n",
                           $9, $3, $6, $5, $4, $10, int($11 / 1024) "G", $8
                else
                    printf "  %-22s %-10s %5d %6d %6d  %s\n", $9, $3, $6, $5, $4, $8
            }' "$TMP/$c.rows" | sort -u -k1,1 | {
            if [ "$RES" = 1 ]; then
                printf '  %-22s %-10s %5s %6s %6s %6s %8s  %s\n' \
                       NODE TYPE FREE ALLOC TOTAL CPU RAM STATE
            else
                printf '  %-22s %-10s %5s %6s %6s  %s\n' NODE TYPE FREE ALLOC TOTAL STATE
            fi
            cat
        }
    else
        if [ "$RES" = 1 ]; then
            printf '  %-26s %-10s %5s %6s %5s %6s %5s  %-10s %9s %8s\n' \
                   PARTITION TYPE FREE HELD ALLOC TOTAL DOWN NODES 1GPU-CPU 1GPU-RAM
        else
            printf '  %-26s %-10s %5s %5s %6s %6s %5s  %s\n' \
                   PARTITION TYPE FREE HELD ALLOC TOTAL DOWN NODES
        fi
        awk -F'|' -v all="$ALL" -v res="$RES" \
            -v capcpu="$(cap_cpu "$c")" -v capmem="$(cap_mem "$c")" '
            $1 == "NODE" {
                k = $2 "|" $3
                tot[k] += $4; alc[k] += $5; fre[k] += $6; dwn[k] += $7; nod[k]++
                hld[k] += $12
                if ($6 > 0) {
                    hasfree[k]++
                    # Best = most RAM, then most CPU, from ONE node, so the pair can
                    # actually be requested together.
                    if ($11 > bmem[k] || ($11 == bmem[k] && $10 > bcpu[k])) {
                        bmem[k] = $11; bcpu[k] = $10
                    }
                    if ($10 > maxcpu[k]) maxcpu[k] = $10
                }
                if (!(seen[$9]++)) {
                    ctot += $4; calc += $5; cfre += $6; cdwn += $7; chld += $12
                }
            }
            END {
                for (k in tot) {
                    split(k, p, "|")
                    # free desc, then partition name: the row you want is the top one
                    if (!res) {
                        printf "%09d|  %-26s %-10s %5d %5d %6d %6d %5d  %d/%d free\n",
                               99999999 - fre[k], p[1], p[2], fre[k], hld[k], alc[k],
                               tot[k], dwn[k], hasfree[k] + 0, nod[k]
                        continue
                    }
                    # What you can ask for = node headroom, capped by site policy.
                    ccpu = bcpu[k]; cmem = bmem[k]
                    if (capcpu > 0 && capcpu < ccpu) ccpu = capcpu
                    if (capmem > 0 && capmem < cmem) cmem = capmem
                    cpu = hasfree[k] ? ccpu ((ccpu == bcpu[k] && maxcpu[k] > bcpu[k]) \
                                             ? "+" : "") : "-"
                    ram = hasfree[k] ? int(cmem / 1024) "G" : "-"
                    printf "%09d|  %-26s %-10s %5d %5d %6d %6d %5d  %-10s %9s %8s\n",
                           99999999 - fre[k], p[1], p[2], fre[k], hld[k], alc[k], tot[k],
                           dwn[k], (hasfree[k] + 0) "/" nod[k], cpu, ram
                }
                # Only under -a: with partitions hidden this would be totalling
                # GPUs that are not on screen.
                if (all)
                    printf "999999999|  %-26s %-10s %5d %5d %6d %6d %5d\n",
                           "TOTAL (nodes counted once)", "", cfre, chld, calc, ctot, cdwn
            }' "$TMP/$c.rows" | sort -t'|' -k1,1 -k2,2 | cut -d'|' -f2-
    fi

    # A filter that silently hides every partition looks identical to a cluster with
    # no GPUs at all; say which it is.
    [ -s "$TMP/$c.rows" ] || printf '  no partition matched %s -- run with -a\n' \
                                    "$(filter_for "$c")"

    # Named by remote login, not called "yours": entropy's j321m is shared, so these
    # jobs may well be someone else's -- but they still consume the same QoS cap.
    holds_line "$c" "  "
}

render_one_table() {
    local c
    : > "$TMP/combined"
    for c in "${CLUSTERS[@]}"; do
        ok_cluster "$c" && awk -v c="$c" '{ print c "|" $0 }' "$TMP/$c.rows" >> "$TMP/combined"
    done

    if [ "$RES" = 1 ]; then
        printf '\n%s%-8s %-26s %-11s %5s %5s %6s %6s %5s  %-9s %9s %8s%s\n' \
               "$BOLD" CLUSTER PARTITION TYPE FREE HELD ALLOC TOTAL DOWN NODES \
               1GPU-CPU 1GPU-RAM "$RESET"
    else
        printf '\n%s%-8s %-26s %-11s %5s %5s %6s %6s %5s  %s%s\n' \
               "$BOLD" CLUSTER PARTITION TYPE FREE HELD ALLOC TOTAL DOWN NODES "$RESET"
    fi
    local caps="" c2
    for c2 in "${CLUSTERS[@]}"; do caps="$caps$c2=$(cap_cpu "$c2"):$(cap_mem "$c2"),"; done
    awk -F'|' -v all="$ALL" -v res="$RES" -v caps="$caps" '
        BEGIN {   # caps is "lem=16:256000,helios=72:121856,..."
            n = split(caps, cc, ",")
            for (i = 1; i <= n; i++) if (cc[i] != "") {
                split(cc[i], kv, "="); split(kv[2], lim, ":")
                CAPC[kv[1]] = lim[1] + 0; CAPM[kv[1]] = lim[2] + 0
            }
        }
        # cluster|NODE|partition|type|total|alloc|free|down|state|node|freecpu|freemem
        $2 == "NODE" {
            k = $1 "|" $3 "|" $4
            TOT[k] += $5; ALC[k] += $6; FRE[k] += $7; DWN[k] += $8; NODES[k]++
            HLD[k] += $13
            if ($7 > 0) {
                WITHFREE[k]++
                # Both figures from one node (most RAM, then most CPU): a pair you can
                # actually request together, unlike the per-column maxima.
                if ($12 > BMEM[k] || ($12 == BMEM[k] && $11 > BCPU[k])) {
                    BMEM[k] = $12; BCPU[k] = $11
                }
                if ($11 > MAXCPU[k]) MAXCPU[k] = $11
            }
            CL[k] = $1; PART[k] = $3; TYPE[k] = $4
            # Node names accumulate in scontrol order, so two partitions over the same
            # nodes build an identical string -- that string is the collapse key.
            SET[k] = SET[k] "," $10
            if (!(($1 "|" $10) in seen)) {
                seen[$1 "|" $10] = 1
                cfre[$1] += $7; calc[$1] += $6; ctot[$1] += $5; cdwn[$1] += $8
                chld[$1] += $13
            }
        }
        END {
            for (k in TOT) {
                sig = CL[k] "|" TYPE[k] "|" SET[k]
                if (sig in keep) { NAMES[sig] = NAMES[sig] "," PART[k]; continue }
                keep[sig] = k; NAMES[sig] = PART[k]
            }
            for (sig in keep) {
                k = keep[sig]
                n = split(NAMES[sig], nm, ",")
                disp = nm[1]
                for (i = 1; i <= n; i++) if (nm[i] ~ /^plgrid/) { disp = nm[i]; break }
                if (n > 1) disp = disp " +" (n - 1)
                if (!res) {
                    printf "%09d|%-8s %-26s %-11s %5d %5d %6d %6d %5d  %d/%d\n",
                           99999999 - FRE[k], CL[k], disp, TYPE[k], FRE[k], HLD[k],
                           ALC[k], TOT[k], DWN[k], WITHFREE[k] + 0, NODES[k]
                    continue
                }
                # node headroom, capped by what one GPU entitles you to ask for
                ccpu = BCPU[k]; cmem = BMEM[k]
                if (CAPC[CL[k]] > 0 && CAPC[CL[k]] < ccpu) ccpu = CAPC[CL[k]]
                if (CAPM[CL[k]] > 0 && CAPM[CL[k]] < cmem) cmem = CAPM[CL[k]]
                cpu = WITHFREE[k] ? ccpu ((ccpu == BCPU[k] && MAXCPU[k] > BCPU[k]) \
                                          ? "+" : "") : "-"
                ram = WITHFREE[k] ? int(cmem / 1024) "G" : "-"
                printf "%09d|%-8s %-26s %-11s %5d %5d %6d %6d %5d  %-9s %9s %8s\n",
                       99999999 - FRE[k], CL[k], disp, TYPE[k], FRE[k], HLD[k], ALC[k],
                       TOT[k], DWN[k], (WITHFREE[k] + 0) "/" NODES[k], cpu, ram
            }
            if (all)
                for (c in ctot)
                    printf "999999999|%-8s %-26s %-11s %5d %5d %6d %6d %5d\n",
                           c, "-- whole cluster --", "", cfre[c], chld[c], calc[c],
                           ctot[c], cdwn[c]
        }' "$TMP/combined" | sort -t'|' -k1,1 -k2,2 | cut -d'|' -f2-

    echo
    for c in "${CLUSTERS[@]}"; do
        if ok_cluster "$c"; then
            printf '  %-8s ' "$c"; holds_line "$c" ""
        else
            printf '  %-8s unreachable -- %s\n' "$c" "$(fail_reason "$c")"
        fi
    done
}

for c in "${CLUSTERS[@]}"; do
    ok_cluster "$c" && apply_filter "$c"
done

if [ "$MODE" = one ]; then
    render_one_table
else
    for c in "${CLUSTERS[@]}"; do render "$c"; done
fi
[ "$ALL" = 1 ] || printf '\nShowing submit-worthy partitions only (-a for all).\n'
[ "$RES" = 1 ] && printf '1GPU-CPU/RAM: largest 1-GPU request that FITS (node headroom, capped per-GPU by
site policy: lem 16c/250G, helios 72c/119G, entropy uncapped). Fitting is not starting:
lem quoted now+46min even for 1 CPU / 1G pinned to a node with 40 CPU and 495G free.\n'
printf '\nFREE is an upper bound: QoS caps (shared account on entropy) and RAM can still
hold a job in the queue. Probe before submitting: sbatch --test-only ... --wrap=true\n'
