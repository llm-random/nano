"""The two categories added to scripts/kprof_agg.py must be purely additive.

`kprof_agg.py` reduces a kernel dump into categories, and every kernel table in
documents/notes/training_efficiency.md was produced with it. Adding a category
is only safe if it claims kernels that previously fell through to "other" --
otherwise an existing category's number moves and old tables stop being
reproducible by the tool that made them.

That is easy to get wrong in one specific way: `CATS` is first-match-wins, and
inductor names fused kernels after the ops it fused, so a real kernel is called
`triton_poi_fused_embedding_dense_backward_3`. An `embedding` pattern placed
ahead of `compiled(triton)` silently steals it. This file is the check.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "kprof_agg.py"

# The category list exactly as it stood before the O31 additions. Hardcoded on
# purpose: the point is to compare against history, so this must not be derived
# from the file under test.
CATS_BEFORE = [
    ("attention(flash)", re.compile(r"flash_fwd|flash_bwd|fmha|attention", re.I)),
    ("optimizer", re.compile(r"adam|multi_tensor|lerp|foreach", re.I)),
    ("gemm", re.compile(r"nvjet|cutlass|gemm|xmma|cublas", re.I)),
    ("compiled(triton)", re.compile(r"^triton_")),
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

# Representative CUDA kernel names, spelled the way the profiler emits them.
KERNELS = [
    "ncclDevKernel_ReduceScatter_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>)",
    "ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)",
    "void at::native::(anonymous namespace)::embedding_dense_backward_kernel<float>(...)",
    "void at::native::(anonymous namespace)::compute_grad_weight<float, float>(...)",
    "void at::native::(anonymous namespace)::krn_partial_segment_offset(...)",
    "void at::native::(anonymous namespace)::sum_and_scatter<float, float>(...)",
    "void at::native::indexSelectLargeIndex<c10::BFloat16, long, 2, 2, -2, true>(...)",
    # the trap: inductor fused an embedding op into a triton kernel
    "triton_poi_fused_embedding_dense_backward_3",
    "triton_poi_fused_add_mul_7",
    "void at::native::multi_tensor_apply_kernel<...fused_adamw...>(...)",
    "void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_f32>(...)",
    "void at::native::(anonymous namespace)::cunn_SoftMaxForward<float>(...)",
    "void at::native::elementwise_kernel<128, 2, ...>(...)",
]


def _categorize(name, cats):
    for cat, rx in cats:
        if rx.search(name):
            return cat
    return "other"


def _cats_now():
    """Load CATS out of the live script without executing its top-level work
    (it reads sys.argv[1] at import)."""
    source = SCRIPT.read_text()
    start = source.index("CATS = [")
    end = source.index("\n]\n", start) + 3
    namespace = {"re": re}
    exec(compile(source[start:end], str(SCRIPT), "exec"), namespace)  # noqa: S102
    return namespace["CATS"]


def test_the_new_categories_only_take_from_other():
    """Nothing may move between two pre-existing categories, and nothing may
    move out of a pre-existing category into a new one."""
    cats_now = _cats_now()
    new_names = {name for name, _ in cats_now} - {name for name, _ in CATS_BEFORE}
    assert new_names == {"comm(nccl)", "embedding/bank"}, new_names

    moved = {}
    for kernel in KERNELS:
        before = _categorize(kernel, CATS_BEFORE)
        after = _categorize(kernel, cats_now)
        if before != after:
            moved[kernel] = (before, after)
            assert before == "other", (
                f"{kernel!r} moved out of an existing category "
                f"{before!r} into {after!r} -- the addition is not additive, and "
                "tables in training_efficiency.md would no longer reproduce"
            )
    assert moved, "no kernel was reclassified; the new categories do nothing"


def test_the_inductor_fused_embedding_kernel_stays_with_triton():
    """The specific ordering trap: first match wins, and `embedding` would
    otherwise claim inductor's fused kernel."""
    assert (
        _categorize("triton_poi_fused_embedding_dense_backward_3", _cats_now())
        == "compiled(triton)"
    )


@pytest.mark.parametrize(
    "kernel,expected",
    [
        ("ncclDevKernel_ReduceScatter_Sum_f32_RING_LL(...)", "comm(nccl)"),
        ("ncclDevKernel_AllGather_RING_LL(...)", "comm(nccl)"),
        (
            "void at::native::(anonymous namespace)::embedding_dense_backward_kernel<float>",
            "embedding/bank",
        ),
        (
            "void at::native::(anonymous namespace)::sum_and_scatter<float, float>",
            "embedding/bank",
        ),
        (
            "void at::native::(anonymous namespace)::krn_partial_segment_offset()",
            "embedding/bank",
        ),
        (
            "void at::native::indexSelectLargeIndex<c10::BFloat16, long>",
            "embedding/bank",
        ),
    ],
)
def test_the_paths_tier_2_and_3_target_are_visible(kernel, expected):
    """These are the kernels the bank gradient path and the gradient
    communication actually show up as. If any of them lands in "other", the
    O31 attribution table cannot answer the question it exists for."""
    assert _categorize(kernel, _cats_now()) == expected


def test_the_script_still_runs_end_to_end(tmp_path):
    log = tmp_path / "kprof.out"
    log.write_text(
        "".join(
            f"kprof_us: self_cuda={1000 * (i + 1)} count=2 name={k}\n"
            for i, k in enumerate(KERNELS)
        )
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(log)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "comm(nccl)" in result.stdout
    assert "embedding/bank" in result.stdout
    assert "TOTAL busy" in result.stdout
