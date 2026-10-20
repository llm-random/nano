"""Paper Fig. 2: analytic PC overhead (as originally plotted) + measured wall-clock overhead.

Gold: the original analytic curve from nano_local/nano/mastest.ipynb — fixed fusion cost
6rd(d+r) over the dynamic pass cost 6br^2 (d=2048, r=960), same styling.
Blue: measured (t_PC - t_HP) / t_HP from benchmarks/bench_block.py — one Llama-1B block at
90% compression on 1xH100, mean of the last 50 of 100 steps (results/*_100step*.jsonl).

Usage (conda env nanollmr, from repo root):
  python benchmarks/plot_overhead_fig2.py --out pc_paper/<id>/Figures/projection_compression_overhead_measured.png
"""

import argparse
import glob
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def analytic_overhead_pct(b, d=2048, r=960):
    return 6 * r * d * (d + r) / (6 * b * r**2) * 100


def measured_overhead(pattern):
    """Overhead per batch size. With repeated seeds, each PC run is paired with the HP run of the
    same seed; we return the mean over seeds and its std (overhead_std, used as error bars)."""
    rows = [json.loads(line) for f in sorted(glob.glob(pattern)) for line in open(f)]
    df = pd.DataFrame(rows)
    df = df[df.status == "ok"]
    if "seed" in df.columns and df.seed.nunique() > 1:
        w = df.pivot_table(index=["tokens", "seed"], columns="mode", values="s_per_step").dropna().reset_index()
        w["ovh"] = 100 * (w.pc - w.hp) / w.hp
        w = w.groupby("tokens").agg(hp=("hp", "mean"), pc=("pc", "mean"), overhead_pct=("ovh", "mean"),
                                    overhead_std=("ovh", "std"), n_seeds=("ovh", "size")).reset_index()
    else:
        w = df.pivot_table(index="tokens", columns="mode", values="s_per_step").dropna().reset_index()
        w["overhead_pct"] = 100 * (w.pc - w.hp) / w.hp
    return w.sort_values("tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="benchmarks/results/bench_block_90p_100step*.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ymax", type=float, default=10.0)
    ap.add_argument("--xmax", type=float, default=2_000_000)
    ap.add_argument("--no-errorbars", action="store_true", help="plot seed means without std error bars")
    ap.add_argument("--inset", type=float, nargs=2, metavar=("XMIN", "XMAX"), default=None,
                    help="add a zoomed inset over this batch range so small error bars are visible")
    ap.add_argument("--r", type=int, default=960, help="compressed width for the analytic curve")
    ap.add_argument("--measured-label", default="Measured wall-clock (1 block, 90%, 1×H100)")
    ap.add_argument("--series", action="append", default=[], metavar="LABEL=GLOB",
                    help="measured series to plot (repeatable); overrides --results/--measured-label")
    args = ap.parse_args()

    b = np.linspace(1, args.xmax, 1000)
    series = [tuple(x.rsplit("=", 1)) for x in args.series] or [(args.measured_label, args.results)]
    colors = ["royalblue", "crimson", "seagreen", "darkorange"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=b, y=analytic_overhead_pct(b, r=args.r), name=f"Analytic FLOP estimate (d=2048, r={args.r})",
        line=dict(color="gold", width=3.5),
    ))
    for (label, pattern), color in zip(series, colors):
        m = measured_overhead(pattern)
        err = (dict(type="data", array=m.overhead_std, visible=True, thickness=2.5, width=7, color="black")
               if "overhead_std" in m and not args.no_errorbars else None)
        fig.add_trace(go.Scatter(
            x=m.tokens, y=m.overhead_pct, mode="lines+markers", name=label, error_y=err,
            line=dict(color=color, width=3.5), marker=dict(size=6),
        ))
        if args.inset and "overhead_std" in m:
            z = m[(m.tokens >= args.inset[0]) & (m.tokens <= args.inset[1])]
            fig.add_trace(go.Scatter(
                x=z.tokens, y=z.overhead_pct, mode="lines+markers", showlegend=False, xaxis="x2", yaxis="y2",
                line=dict(color=color, width=2.5), marker=dict(size=6),
                error_y=dict(type="data", array=z.overhead_std, visible=True, thickness=2.5, width=6, color="black"),
            ))
            lo = float((z.overhead_pct - z.overhead_std).min()); hi = float((z.overhead_pct + z.overhead_std).max())
            pad = 0.08 * (hi - lo)
            inset_axes = dict(
                xaxis2=dict(domain=[0.48, 0.98], anchor="y2", range=list(args.inset), showgrid=True,
                            gridcolor="lightgray", linecolor="black", mirror=True, tickfont=dict(size=14)),
                yaxis2=dict(domain=[0.45, 0.97], anchor="x2", range=[lo - pad, hi + pad], showgrid=True,
                            gridcolor="lightgray", linecolor="black", mirror=True, tickfont=dict(size=14),
                            title=dict(text="zoom (±1 std, 5 seeds)", font=dict(size=14))),
            )
        print(f"--- {label} ({pattern})")
        cols = [c for c in ["tokens", "hp", "pc", "overhead_pct", "overhead_std", "n_seeds"] if c in m]
        print(m[cols].round(4).to_string(index=False))
    fig.update_layout(
        title=None,
        margin=dict(l=80, r=30, t=20, b=70),
        font=dict(size=20),
        hovermode="x unified",
        legend=dict(yanchor="top", y=-0.2, font=dict(size=18), xanchor="center", x=0.5, orientation="h",
                    bgcolor="rgba(255, 255, 255, 0.8)"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(title="Batch Size (Tokens, b)", range=[0, args.xmax], showgrid=True,
                   gridcolor="lightgray", linecolor="black"),
        yaxis=dict(title=dict(text="Relative Computational Overhead (%)", font=dict(size=19)), range=[0, args.ymax],
                   showgrid=True, gridcolor="lightgray", linecolor="black"),
    )
    if args.inset and "inset_axes" in locals():
        fig.update_layout(**inset_axes)
    fig.write_image(args.out, format="png", width=700, height=600, scale=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
