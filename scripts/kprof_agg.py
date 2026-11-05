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
    # Two categories added 2026-11-04 for the AME optimization work (O31). Both
    # are ADDITIVE: every kernel they claim previously fell through to "other",
    # so no existing category's number moves and the tables already published in
    # training_efficiency.md stay reproducible. tests/test_kprof_agg.py asserts
    # that property rather than leaving it as a claim.
    #
    # POSITION MATTERS and is why these sit here rather than at the top. First
    # match wins, and inductor names its fused kernels after the ops it fused --
    # `triton_poi_fused_embedding_dense_backward_3` is real. Ahead of
    # `compiled(triton)` the embedding pattern would steal those and silently
    # move an existing category's number; behind it, eager ATen kernels still
    # land here because they do not start with `triton_`.
    #
    # comm(nccl) is what makes gradient communication visible. Inductor fuses
    # pointwise work into opaque triton_poi_fused_* kernels but never touches
    # NCCL, so reduce-scatter and all-reduce stay legible under compile -- which
    # is what lets O9 (reduce once per step, not once per micro-step) and Tier 3
    # (shard the banks) be measured at all.
    (
        "comm(nccl)",
        re.compile(
            r"nccl|reduce_?scatter|all_?gather|all_?reduce"
            r"|all_?to_?all|alltoall|send_?recv",
            re.I,
        ),
    ),
    # embedding/bank is AME's bank gradient path. The banks are nn.Embedding, so
    # their backward is embedding_dense_backward -- which on CUDA decomposes into
    # a segment-offset/partial-sum/scatter trio, none of which looks like
    # "embedding" by name. Catch the whole family or undercount the thing Tier 2
    # exists to remove.
    (
        "embedding/bank",
        re.compile(
            r"embedding|indexSelect|index_select|compute_grad_weight"
            r"|sum_and_scatter|partial_segment|index_add|indexAdd"
            # index_add_ on CUDA dispatches to indexFunc*Index, which contains
            # neither "index_add" nor "embedding". Without this the Tier 2
            # scatter -- 93.8 ms/step at h20 -- lands in "other" and the bank
            # path reads as cheaper than it is.
            r"|indexFunc",
            re.I,
        ),
    ),
    ("softmax/CE", re.compile(r"SoftMax|cross_entropy|nll", re.I)),
    (
        "norm-ish eager (pow/reduce)",
        re.compile(r"pow_tensor|reduce_kernel|rsqrt|norm", re.I),
    ),
    (
        "eager elementwise",
        re.compile(r"elementwise_kernel|CUDAFunctor|BinaryFunctor|Functor", re.I),
    ),
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
