"""Aggregate kprof_us kernel dumps (2 steps) into comparable categories, ms/step.

Input is any log holding the `kprof_us: self_cuda=... count=... name=...` lines
that `trainer.kernel_profile=true` prints (steps 12-13), or the same lines from a
patched nanoGPT clone -- the categories are framework-agnostic on purpose, so the
two sides of a comparison are reduced identically. Every kernel table in
documents/notes/training_efficiency.md (Parts 4.2 and 4.4) was produced with it.

    python scripts/kprof_agg.py slurm-32064_0.out        # category totals
    python scripts/kprof_agg.py slurm-32064_0.out -v     # + top kernels per category

Dumps cover two profiled steps, hence the /2 in the per-step reductions.
"""
import re
import sys

CATS = [
    ("attention(flash)", re.compile(r"flash_fwd|flash_bwd|fmha|attention", re.I)),
    ("optimizer", re.compile(r"adam|multi_tensor|lerp|foreach", re.I)),
    ("gemm", re.compile(r"nvjet|cutlass|gemm|xmma|cublas", re.I)),
    ("compiled(triton)", re.compile(r"^triton_")),
    ("softmax/CE", re.compile(r"SoftMax|cross_entropy|nll", re.I)),
    ("norm-ish eager (pow/reduce)", re.compile(r"pow_tensor|reduce_kernel|rsqrt|norm", re.I)),
    ("eager elementwise", re.compile(r"elementwise_kernel|CUDAFunctor|BinaryFunctor|Functor", re.I)),
    ("cat/copy/memcpy", re.compile(r"CatArray|Memcpy|copy_kernel|direct_copy", re.I)),
]

def categorize(name):
    for cat, rx in CATS:
        if rx.search(name):
            return cat
    return "other"

totals, counts, per_kernel = {}, {}, {}
for line in open(sys.argv[1]):
    m = re.search(r"kprof_us: self_cuda=(\d+) count=(\d+) name=(.*)", line)
    if not m:
        continue
    us, cnt, name = int(m.group(1)), int(m.group(2)), m.group(3).strip()
    cat = categorize(name)
    totals[cat] = totals.get(cat, 0) + us
    counts[cat] = counts.get(cat, 0) + cnt
    per_kernel.setdefault(cat, []).append((us, cnt, name[:90]))

grand = sum(totals.values())
print(f"{'category':32s} {'ms/step':>9s} {'launches/step':>14s}")
for cat in sorted(totals, key=totals.get, reverse=True):
    print(f"{cat:32s} {totals[cat]/2000:9.1f} {counts[cat]//2:14d}")
print(f"{'TOTAL busy':32s} {grand/2000:9.1f} {sum(counts.values())//2:14d}")

if len(sys.argv) > 2 and sys.argv[2] == "-v":
    for cat in sorted(totals, key=totals.get, reverse=True):
        print(f"\n== {cat}")
        for us, cnt, name in sorted(per_kernel[cat], reverse=True)[:6]:
            print(f"  {us/2000:8.1f} ms/step  x{cnt//2:6d}  {name}")
