"""Mean of each timers_ms section per rank, over a step window.

Multi-GPU only: the spread between ranks is the straggler/imbalance signal.
Usage: python timers_by_rank.py <logfile> [lo hi]
"""

import re
import statistics
import sys

LINE = re.compile(r"timers_ms: (?:rank=(?P<rank>\d+) )?step=(?P<step>\d+) (?P<rest>.*)")
SECTIONS = (
    "data_fetch",
    "data_h2d",
    "forward",
    "backward",
    "clip",
    "optim",
    "log",
    "step_total",
)

log = sys.argv[1]
lo, hi = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (16, 29)

per_rank = {}
for line in open(log, errors="replace"):
    m = LINE.search(line)
    if not m:
        continue
    step = int(m["step"])
    if not (lo <= step <= hi):
        continue
    rank = int(m["rank"] or 0)
    fields = dict(kv.split("=") for kv in m["rest"].split() if "=" in kv)
    bucket = per_rank.setdefault(rank, {s: [] for s in SECTIONS})
    for s in SECTIONS:
        if s in fields:
            bucket[s].append(float(fields[s]))

if not per_rank:
    raise SystemExit(f"no timers_ms lines in steps {lo}-{hi}")

header = "rank  " + "".join(f"{s:>12s}" for s in SECTIONS)
print(header)
for rank in sorted(per_rank):
    row = f"{rank:<6d}"
    for s in SECTIONS:
        vals = per_rank[rank][s]
        row += f"{statistics.mean(vals):12.1f}" if vals else f"{'-':>12s}"
    print(row)

if len(per_rank) > 1:
    totals = {
        r: statistics.mean(v["step_total"])
        for r, v in per_rank.items()
        if v["step_total"]
    }
    spread = max(totals.values()) - min(totals.values())
    print(
        f"\nstep_total spread across ranks: {spread:.1f} ms "
        f"({spread / min(totals.values()):.2%} of the fastest rank)"
    )
