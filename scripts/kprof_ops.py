"""Operator-level view of a kprof dump: where GPU time goes inside each pass.

Reads the `kprof_op:` lines (inclusive GPU time per operator / annotated
range), which exist because the profiler records CPU activity alongside CUDA.
Its companion scripts/kprof_agg.py reads the `kprof_us:` self-time lines and
buckets raw kernels; this one answers "how much GPU time is inside forward vs
backward, and which operators own it".

`section:*` rows come from LoopTimers' GPU sections, so forward and backward
are bounded exactly rather than guessed from kernel names -- which does not
work, since GEMMs and elementwise kernels serve both passes identically.

Inclusive times nest (a parent contains its children), so only compare rows at
the same level: the section rows against each other, and within a pass the
operator rows against each other.

Usage:
    python scripts/kprof_ops.py <logfile> [--top N] [--steps 2]
"""

import argparse
import re

ROW = re.compile(r"kprof_op: cuda_total=(\d+) self=(\d+) count=(\d+) name=(.*)")
# Backward work is reported by the autograd engine under these names.
BACKWARD = re.compile(r"autograd::engine|Backward|backward", re.I)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile")
    parser.add_argument("--top", type=int, default=18)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()

    sections, ops = [], []
    for line in open(args.logfile, errors="replace"):
        m = ROW.search(line)
        if not m:
            continue
        total = int(m[1]) / (args.steps * 1000)
        count = int(m[3]) // args.steps
        name = m[4].strip()
        (sections if name.startswith("section:") else ops).append((total, count, name))
    if not sections and not ops:
        raise SystemExit(
            "no kprof_op: lines -- the run predates operator profiling, "
            "or kernel_profile was off"
        )

    if sections:
        print("GPU time per loop section (inclusive, ms/step)")
        print(f"{'ms/step':>9} {'entries':>8}  section")
        for total, count, name in sorted(sections, reverse=True):
            print(f"{total:9.1f} {count:8d}  {name.replace('section:', '')}")

    def dump(title, subset):
        if not subset:
            return
        print(f"\n{title} (inclusive ms/step, top {args.top})")
        print(f"{'ms/step':>9} {'calls':>8}  operator")
        for total, count, name in sorted(subset, reverse=True)[: args.top]:
            print(f"{total:9.1f} {count:8d}  {name[:70]}")

    dump("BACKWARD operators", [o for o in ops if BACKWARD.search(o[2])])
    dump("FORWARD / other operators", [o for o in ops if not BACKWARD.search(o[2])])


if __name__ == "__main__":
    main()
