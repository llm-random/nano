"""Forward and backward cost of AME's attention kernel, measured separately.

The aggregated kernel dump cannot answer this. AME's `flex_attention` is
compiled into the whole-block graph, so no operator boundary survives for the
profiler to attribute against -- unlike the dense model, whose flash path leaves
`aten::_scaled_dot_product_flash_attention{,_backward}` rows and splits cleanly
(182.1 ms forward / 365.4 ms backward per step, job 32601).

So measure it directly, on the real layer, at the real shape.

    python scripts/bench_attention.py                 # h20 shape, one GPU

Reports milliseconds per CALL and, scaled by layers x micro-steps, the
contribution to one optimizer step of the h20 4-GPU recipe -- so the numbers
line up with the kernel tables in documents/notes/archive/ame_optimization_plan.md.

The SDPA row is the floor, not a drop-in replacement: `is_causal=True` with no
additive bias is what the dense twin runs. AME needs an ALiBi distance bias and
a learned sink, neither of which PyTorch 2.7's flash backend accepts. The gap
between the two rows is what a bias-free kernel would be worth, which is the
thing worth knowing before anyone engineers one.
"""

import argparse
import sys

import torch

import os

# Repo root from this file, so the script runs from a cemetery clone too.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def timed(fn, *, warmup=5, iters=20):
    """ms per call, CUDA-event timed, warmed and synchronised."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16, help="micro-batch per GPU")
    ap.add_argument("--heads", type=int, default=20)
    ap.add_argument("--dhead", type=int, default=64)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--micro-steps", type=int, default=8, help="ga")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("needs a GPU")
        return 1
    dev = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(dev)}  torch {torch.__version__}")
    print(f"shape: B={args.batch} H={args.heads} n={args.seq} D={args.dhead} bf16")
    per_step = args.layers * args.micro_steps
    print(
        f"per-step scale: {args.layers} layers x {args.micro_steps} micro-steps "
        f"= {per_step} calls\n"
    )

    from src.core.ame import AMEAttention

    layer = (
        AMEAttention(
            args.heads * args.dhead,
            args.heads,
            args.heads,
            vocab_size=50304,
            seq_len=args.seq,
            dhead=args.dhead,
        )
        .to(dev)
        .to(torch.bfloat16)
    )

    shape = (args.batch, args.heads, args.seq, args.dhead)

    def fresh(requires_grad=True):
        return [
            torch.randn(
                shape, device=dev, dtype=torch.bfloat16, requires_grad=requires_grad
            )
            for _ in range(3)
        ]

    rows = []

    # ---- AME's actual kernel: compiled flex_attention + score_mod + block mask
    q, k, v = fresh()
    fwd_ms = timed(lambda: layer._attend_flex(q, k, v))
    out = layer._attend_flex(q, k, v)
    grad = torch.randn_like(out)

    def bwd():
        for t in (q, k, v):
            t.grad = None
        out.backward(grad, retain_graph=True)

    bwd_ms = timed(bwd)
    rows.append(("AME flex_attention (ALiBi + sink)", fwd_ms, bwd_ms))

    # ---- the floor: SDPA flash, causal, no bias
    q2, k2, v2 = fresh()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def sdpa_fwd():
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return torch.nn.functional.scaled_dot_product_attention(
                q2, k2, v2, is_causal=True
            )

    try:
        f2 = timed(sdpa_fwd)
        o2 = sdpa_fwd()
        g2 = torch.randn_like(o2)

        def sdpa_bwd():
            for t in (q2, k2, v2):
                t.grad = None
            o2.backward(g2, retain_graph=True)

        b2 = timed(sdpa_bwd)
        rows.append(("SDPA flash, causal, NO bias (floor)", f2, b2))
    except Exception as exc:  # noqa: BLE001
        print(f"SDPA flash unavailable: {type(exc).__name__}: {exc}")

    # ---- decomposition: what inside flex costs what -----------------------
    #
    # Three variants of the same kernel, differing only in how much work
    # score_mod does per element. flex evaluates score_mod for EVERY score, so
    # an fp32 round-trip or a torch.where there is paid n^2 times per head.
    from torch.nn.attention.flex_attention import create_block_mask

    def causal_mask(b, h, q_idx, kv_idx):
        return kv_idx <= q_idx

    plain_mask = create_block_mask(
        causal_mask, B=None, H=None, Q_LEN=args.seq, KV_LEN=args.seq, device=dev
    )
    slopes = layer.slopes.to(device=dev, dtype=torch.float32)

    def alibi_only(score, b, h, q_idx, kv_idx):
        # AME's bias without the sink's torch.where
        distance = (q_idx - kv_idx).to(torch.float32)
        return (score.to(torch.float32) - slopes[h] * distance).to(score.dtype)

    def alibi_bf16(score, b, h, q_idx, kv_idx):
        # the same bias with no fp32 round-trip
        return score - slopes[h].to(score.dtype) * (q_idx - kv_idx).to(score.dtype)

    from src.core.ame import _flex_attention

    for label, mod in (
        ("flex: ALiBi fp32, NO sink", alibi_only),
        ("flex: ALiBi bf16, NO sink", alibi_bf16),
        ("flex: no score_mod, causal only", None),
    ):
        qa, ka, va = fresh()
        try:
            call = lambda: _flex_attention(
                qa, ka, va, score_mod=mod, block_mask=plain_mask, scale=1.0
            )
            f3 = timed(call)
            o3 = call()
            g3 = torch.randn_like(o3)

            def bwd3():
                for t in (qa, ka, va):
                    t.grad = None
                o3.backward(g3, retain_graph=True)

            rows.append((label, f3, timed(bwd3)))
        except Exception as exc:  # noqa: BLE001
            print(f"{label}: {type(exc).__name__}: {str(exc)[:90]}")

    # ---- diagnostics for the sink's cost ----------------------------------
    # (a) does asking flex for the LSE make its BACKWARD expensive?
    qd, kd, vd = fresh()
    try:
        call = lambda: _flex_attention(
            qd,
            kd,
            vd,
            score_mod=alibi_only,
            block_mask=plain_mask,
            scale=1.0,
            return_lse=True,
        )
        f4 = timed(lambda: call()[0])
        o4, l4 = call()
        g4, gl4 = torch.randn_like(o4), torch.randn_like(l4)

        def bwd4():
            for t in (qd, kd, vd):
                t.grad = None
            torch.autograd.backward([o4, l4], [g4, gl4], retain_graph=True)

        rows.append(("flex: ALiBi, no sink, return_lse=True", f4, timed(bwd4)))
    except Exception as exc:  # noqa: BLE001
        print(f"return_lse row: {type(exc).__name__}: {str(exc)[:90]}")

    # (b) sink prepended (KV_LEN = n+1, off the 128 block grid) but score_mod
    #     with NO branch -- separates the ragged-block cost from the per-element
    #     torch.where.
    qs, ks, vs = fresh()
    ks1 = (
        torch.cat([torch.randn_like(ks[:, :, :1]), ks], dim=2).detach().requires_grad_()
    )
    vs1 = (
        torch.cat([torch.randn_like(vs[:, :, :1]), vs], dim=2).detach().requires_grad_()
    )
    sink_mask = create_block_mask(
        lambda b, h, q, kv: (kv < 1) | (kv - 1 <= q),
        B=None,
        H=None,
        Q_LEN=args.seq,
        KV_LEN=args.seq + 1,
        device=dev,
    )

    def no_branch(score, b, h, q_idx, kv_idx):
        distance = (q_idx - (kv_idx - 1)).to(torch.float32)
        return (score.to(torch.float32) - slopes[h] * distance).to(score.dtype)

    def with_branch(score, b, h, q_idx, kv_idx):
        distance = (q_idx - (kv_idx - 1)).to(torch.float32)
        biased = (score.to(torch.float32) - slopes[h] * distance).to(score.dtype)
        return torch.where(kv_idx < 1, score, biased)

    for label, mod in (
        ("flex: KV=n+1, sink, NO branch", no_branch),
        ("flex: KV=n+1, sink, with branch", with_branch),
    ):
        try:
            c = lambda: _flex_attention(
                qs, ks1, vs1, score_mod=mod, block_mask=sink_mask, scale=1.0
            )
            f5 = timed(c)
            o5 = c()
            g5 = torch.randn_like(o5)

            def bwd5():
                for t in (qs, ks1, vs1):
                    t.grad = None
                o5.backward(g5, retain_graph=True)

            rows.append((label, f5, timed(bwd5)))
        except Exception as exc:  # noqa: BLE001
            print(f"{label}: {type(exc).__name__}: {str(exc)[:90]}")

    # ---- S3b: the sink merge's own arithmetic ----------------------------
    #
    # The decomposition puts 241.2 ms per step between "flex, ALiBi, no sink"
    # and the full AME kernel, which is the log-sum-exp merge. That is ~6.7x
    # the bandwidth floor for the traffic it moves, so it is a shape or fusion
    # problem rather than work that has to be paid.
    #
    # These rows isolate it. out_seq and lse_seq are LEAF tensors here, so flex
    # is out of the picture entirely and what is measured is the merge and
    # nothing else -- including, in the backward, the production of dq, dk_sink,
    # dv_sink and dlse, but NOT flex's cost of consuming dlse. The `detached
    # lse` row at the end sizes that separately.
    #
    # Both eager and compiled, because the real model runs this inside a
    # compiled block while every row above is eager -- which is the most likely
    # reason the sink merge transferred only 52% of its benchmarked gain into
    # training while the alignment fix transferred 100%.
    if layer.use_sink:
        H, D = args.heads, args.dhead

        def merge_einsum(q_, out_seq, lse_seq):
            k_sink = layer.k_sink.to(q_.dtype)
            s = torch.einsum("bhnd,hd->bhn", q_, k_sink)
            a = torch.sigmoid(s.to(torch.float32) - lse_seq.to(torch.float32)).to(
                out_seq.dtype
            )
            v_sink = layer.v_sink.to(out_seq.dtype).view(1, H, 1, D)
            return out_seq + a.unsqueeze(-1) * (v_sink - out_seq)

        def merge_mulsum(q_, out_seq, lse_seq):
            # Same value, but a reduction over D instead of a matmul whose
            # output is one scalar per (b, h, n) -- the degenerate shape a
            # tensor core is worst at.
            k_sink = layer.k_sink.to(q_.dtype).view(1, H, 1, D)
            s = (q_.to(torch.float32) * k_sink.to(torch.float32)).sum(-1)
            a = torch.sigmoid(s - lse_seq.to(torch.float32)).to(out_seq.dtype)
            v_sink = layer.v_sink.to(out_seq.dtype).view(1, H, 1, D)
            return out_seq + a.unsqueeze(-1) * (v_sink - out_seq)

        variants = [
            ("merge only: einsum (current)", merge_einsum, False),
            ("merge only: mul+sum over D", merge_mulsum, False),
            ("merge only: einsum, compiled", merge_einsum, True),
            ("merge only: mul+sum, compiled", merge_mulsum, True),
        ]
        for label, fn, do_compile in variants:
            try:
                f = torch.compile(fn) if do_compile else fn
                qm = torch.randn(
                    shape, device=dev, dtype=torch.bfloat16, requires_grad=True
                )
                om = torch.randn(
                    shape, device=dev, dtype=torch.bfloat16, requires_grad=True
                )
                lm = torch.randn(
                    (args.batch, H, args.seq),
                    device=dev,
                    dtype=torch.float32,
                    requires_grad=True,
                )
                f(qm, om, lm)  # warm the compile before timing
                f6 = timed(lambda: f(qm, om, lm))
                o6 = f(qm, om, lm)
                g6 = torch.randn_like(o6)

                def bwd6():
                    for t in (qm, om, lm):
                        t.grad = None
                    o6.backward(g6, retain_graph=True)

                rows.append((label, f6, timed(bwd6)))
            except Exception as exc:  # noqa: BLE001
                print(f"{label}: {type(exc).__name__}: {str(exc)[:90]}")

        # Diagnostic ONLY, and the number it produces is not a valid model:
        # detaching lse drops a real term of the true gradient. It is here to
        # separate "the merge's own arithmetic" from "what flex pays to
        # differentiate its LSE output", which no other row distinguishes.
        try:
            qd, kd, vd = fresh()

            def flex_detached_lse():
                out_seq, lse_seq = layer._attend_flex_sequence(
                    qd, kd, vd, return_lse=True
                )
                return merge_einsum(qd, out_seq, lse_seq.detach())

            f7 = timed(flex_detached_lse)
            o7 = flex_detached_lse()
            g7 = torch.randn_like(o7)

            def bwd7():
                for t in (qd, kd, vd):
                    t.grad = None
                o7.backward(g7, retain_graph=True)

            rows.append(("flex+merge, lse DETACHED (wrong math)", f7, timed(bwd7)))
        except Exception as exc:  # noqa: BLE001
            print(f"flex+merge detached-lse: {type(exc).__name__}: {str(exc)[:90]}")

    hdr = f"{'kernel':38s} {'fwd ms':>8} {'bwd ms':>8} {'bwd/fwd':>8} {'total':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name, f, b in rows:
        print(f"{name:38s} {f:8.3f} {b:8.3f} {b/f:8.2f} {f+b:8.3f}")
    print()
    print(
        f"{'scaled to one optimizer step (ms)':38s} {'fwd':>8} {'bwd':>8} {'':>8} {'total':>8}"
    )
    print("-" * len(hdr))
    for name, f, b in rows:
        print(
            f"{name:38s} {f*per_step:8.1f} {b*per_step:8.1f} {'':>8} "
            f"{(f+b)*per_step:8.1f}"
        )
    if len(rows) == 2:
        (_, f1, b1), (_, f2_, b2_) = rows
        print()
        print(
            f"headroom if the bias were free: fwd {(f1-f2_)*per_step:+.1f} ms, "
            f"bwd {(b1-b2_)*per_step:+.1f} ms, "
            f"total {((f1+b1)-(f2_+b2_))*per_step:+.1f} ms per step"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
