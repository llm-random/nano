"""Split a profiled step into FORWARD and BACKWARD, per kernel category.

`kprof_agg.py` aggregates by kernel name, which cannot be split by pass: a GEMM
or an elementwise kernel serves both, under one name. This reads the chrome
trace instead, where every GPU kernel keeps the `External id` of the CPU
operator that launched it, and walks each kernel back to whichever `section:`
range LoopTimers had open at the time -- forward, backward, clip or optim.

    python scripts/kprof_phase.py kprof_trace_rank0.json
    python scripts/kprof_phase.py ame.json dense.json      # side by side

Requires `trainer.kernel_profile=true` (writes the trace) and
`trainer.profile_timers=true` (opens the `section:` ranges). Times are ms per
step; the profiled window is two steps, hence the /2.
"""

import gzip
import json
import re
import sys
from bisect import bisect_right

STEPS_PROFILED = 2

CATS = [
    ("ATTENTION (flex)", re.compile(r"^triton_tem_")),
    ("ATTENTION (flash)", re.compile(r"pytorch_flash|flash_fwd|flash_bwd|fmha")),
    ("NCCL all-gather", re.compile(r"nccl.*AllGather", re.I)),
    ("NCCL reduce-scatter", re.compile(r"nccl.*ReduceScatter", re.I)),
    ("NCCL other", re.compile(r"nccl", re.I)),
    (
        "bank / embedding",
        re.compile(r"embedding|indexSelect|sum_and_scatter|partial_segment", re.I),
    ),
    ("GEMM", re.compile(r"nvjet|cutlass|_gemm|xmma|cublas", re.I)),
    ("optimizer", re.compile(r"multi_tensor|adam|lerp|foreach", re.I)),
    ("FSDP2 packing", re.compile(r"chunk_cat|split_with_sizes", re.I)),
    ("softmax / CE", re.compile(r"SoftMax|cross_entropy|nll", re.I)),
    ("triton glue", re.compile(r"^triton_")),
    (
        "copy / memcpy",
        re.compile(r"CatArray|Memcpy|Memset|copy_kernel|direct_copy", re.I),
    ),
]
KERNEL_CATS = {"kernel", "gpu_memcpy", "gpu_memset", "Kernel"}


def categorize(name):
    for cat, rx in CATS:
        if rx.search(name):
            return cat
    return "other eager"


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        return json.load(fh)["traceEvents"]


def phase_table(path):
    events = load(path)

    # Attribute by OPERATOR IDENTITY, not by timestamp.
    #
    # Two timestamp schemes were tried and both are wrong, in ways that look
    # like results:
    #   - keyed by thread: autograd runs backward on its own thread, so nothing
    #     matched `section:backward` and 65% landed in "unattributed".
    #   - global time intervals: the autograd thread lags, so its ops'
    #     timestamps fall inside the NEXT micro-step's `section:forward`. That
    #     reported dense flash as 327.6 fwd / 177.6 bwd when the kernel names
    #     say 141.5 / 365.4 -- a confident, inverted answer.
    #
    # What is actually reliable is the operator tree: a kernel is backward iff
    # the CPU op that launched it sits under an autograd node. Ops nest by time
    # within a thread, so a stack scan per thread gives each op its ancestors.
    by_thread = {}
    for e in events:
        if (
            e.get("ph") == "X"
            and e.get("cat") not in KERNEL_CATS
            and e.get("dur") is not None
        ):
            by_thread.setdefault(e.get("tid"), []).append(e)

    BWD = re.compile(r"Backward|autograd::engine|AccumulateGrad|torch::autograd")
    ext_phase = {}
    for tid, evs in by_thread.items():
        evs.sort(key=lambda x: (x["ts"], -(x.get("dur") or 0)))
        stack = []  # (end_ts, is_backward)
        for e in evs:
            ts, dur = e["ts"], (e.get("dur") or 0)
            while stack and stack[-1][0] < ts:
                stack.pop()
            inherited = stack[-1][1] if stack else False
            mine = inherited or bool(BWD.search(e.get("name", "")))
            stack.append((ts + dur, mine))
            ext = (e.get("args") or {}).get("External id")
            if ext is not None:
                # A backward verdict anywhere wins: the same External id can be
                # seen on both the launching op and a parent.
                ext_phase[ext] = ext_phase.get(ext, False) or mine

    table = {}
    unattributed = 0.0
    for e in events:
        if e.get("cat") not in KERNEL_CATS or e.get("ph") != "X":
            continue
        dur = e.get("dur", 0) / 1000.0 / STEPS_PROFILED  # ms/step
        ext = (e.get("args") or {}).get("External id")
        cat = categorize(e.get("name", ""))
        if ext not in ext_phase:
            unattributed += dur
            ph = "unattributed"
        else:
            ph = "backward" if ext_phase[ext] else "forward"
        row = table.setdefault(cat, {})
        row[ph] = row.get(ph, 0.0) + dur
    return table, unattributed


def show(path):
    table, unattributed = phase_table(path)
    phases = ["forward", "backward", "unattributed"]
    present = [p for p in phases if any(p in r for r in table.values())]
    hdr = (
        f"{'category':24s}" + "".join(f"{p:>13s}" for p in present) + f"{'total':>11s}"
    )
    print(f"\n### {path}")
    print(hdr)
    print("-" * len(hdr))
    totals = dict.fromkeys(present, 0.0)
    for cat, row in sorted(table.items(), key=lambda kv: -sum(kv[1].values())):
        line = f"{cat:24s}"
        for p in present:
            v = row.get(p, 0.0)
            totals[p] += v
            line += f"{v:13.1f}"
        line += f"{sum(row.values()):11.1f}"
        print(line)
    print("-" * len(hdr))
    line = f"{'TOTAL':24s}"
    for p in present:
        line += f"{totals[p]:13.1f}"
    line += f"{sum(totals.values()):11.1f}"
    print(line)
    if unattributed:
        print(
            f"\nnote: {unattributed:.1f} ms/step could not be tied to a section "
            f"({100*unattributed/max(sum(totals.values()),1e-9):.1f}%) -- kernels "
            "launched outside any LoopTimers range (e.g. the optimizer's own "
            "collectives, or async work draining after a section closed)."
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    for p in sys.argv[1:]:
        show(p)
