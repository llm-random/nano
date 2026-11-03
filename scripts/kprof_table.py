"""Every kernel from a kprof dump, one row each, ms/step -- the level below
scripts/kprof_agg.py's categories.

Kernel names carry the phase: pytorch_flash::flash_fwd / flash_bwd, inductor's
triton_*_fused_* names list the ops they fused, and nvjet GEMM names encode the
tile shape. That is what makes a forward/backward split possible without extra
instrumentation -- see the --phase flag, which groups by a name-based guess and
prints what it could not classify rather than hiding it.

Usage:
    python scripts/kprof_table.py <logfile> [--top N] [--phase] [--min-ms 1.0]
"""

import argparse
import re

ROW = re.compile(r"kprof_us: self_cuda=(\d+) count=(\d+) name=(.*)")

# Name fragments that place a kernel in the backward pass. Inductor appends
# _backward_ / _bwd to the fused names it generates for grad kernels, and the
# flash kernels say so outright.
BACKWARD = re.compile(r"flash_bwd|_backward|_bwd|GradFunctor|grad_weight", re.I)
# Optimizer/communication run outside both passes.
OUTSIDE = re.compile(r"adam|multi_tensor|nccl|onerank|chunk_cat|split_with_sizes", re.I)


def short(name: str, width: int = 64) -> str:
    """Readable kernel identity: drop template args and namespaces."""
    name = re.sub(r"<[^<>]*>", "<>", name)
    name = re.sub(r"\(.*", "", name)
    name = name.replace("void ", "").strip()
    name = re.sub(r"^at::native::(\(anonymous namespace\)::)?", "", name)
    return name if len(name) <= width else name[: width - 1] + "…"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile")
    parser.add_argument("--top", type=int, default=0, help="0 = every kernel")
    parser.add_argument("--min-ms", type=float, default=0.0)
    parser.add_argument("--phase", action="store_true", help="group fwd/bwd/outside")
    parser.add_argument("--steps", type=int, default=2, help="steps the dump covers")
    args = parser.parse_args()

    rows = []
    for line in open(args.logfile, errors="replace"):
        m = ROW.search(line)
        if not m:
            continue
        ms = int(m[1]) / (args.steps * 1000)
        rows.append((ms, int(m[2]) // args.steps, m[3].strip()))
    if not rows:
        raise SystemExit("no kprof_us lines found")

    rows.sort(reverse=True)
    rows = [r for r in rows if r[0] >= args.min_ms]
    if args.top:
        rows = rows[: args.top]

    def dump(title, subset):
        if not subset:
            return
        total = sum(r[0] for r in subset)
        print(f"\n{title}  --  {total:,.1f} ms/step, {len(subset)} kernels")
        print(f"{'ms/step':>9} {'launches':>9}  kernel")
        for ms, count, name in subset:
            print(f"{ms:9.1f} {count:9d}  {short(name)}")

    if not args.phase:
        dump("all kernels", rows)
        print(f"\nTOTAL {sum(r[0] for r in rows):,.1f} ms/step")
        return

    outside = [r for r in rows if OUTSIDE.search(r[2])]
    rest = [r for r in rows if not OUTSIDE.search(r[2])]
    bwd = [r for r in rest if BACKWARD.search(r[2])]
    fwd = [r for r in rest if not BACKWARD.search(r[2])]
    dump("FORWARD (name-based; includes anything not marked backward)", fwd)
    dump("BACKWARD", bwd)
    dump("OUTSIDE both passes (optimizer, collectives, plumbing)", outside)
    print(
        f"\nfwd {sum(r[0] for r in fwd):,.1f} | bwd {sum(r[0] for r in bwd):,.1f} | "
        f"outside {sum(r[0] for r in outside):,.1f} ms/step"
    )


if __name__ == "__main__":
    main()
