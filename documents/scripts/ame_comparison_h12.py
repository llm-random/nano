"""AME against dense at h12: the training logs of the two best h12 runs, side by side.

The numbers and figures in documents/notes/ame_comparison_h12.md. Read-only on W&B.

    python documents/scripts/ame_comparison_h12.py tables
    python documents/scripts/ame_comparison_h12.py figures OUT_DIR

Run it with the project env (`./.pixi/envs/default/bin/python`); WANDB_API_KEY must
be set. Both runs are finished, so their histories are cached as parquet under
$AME_COMPARISON_CACHE (default ~/.cache/nano/ame_comparison_h12) and fetched once.

The runs, in ideas_cv/nano-ame:
- AME:   S2-PS (ALiBi + KV shift), n = 8, `doi36p8k`, job 6256577;
- dense: S1-D (GQA + RoPE),        n = 8, `x0xlqroj`, job 6252398_0.
Both are h12, 3,815 steps of 1,048,576 tokens, with data seed 123: the same batches
in the same order. So their per-step train losses are paired, and their difference
is free of batch-to-batch noise.

What the logs hold (src/core/trainer.py, Residual in src/core/model.py,
AMEAttention.instrumentation in src/core/ame.py):
- every step: train/loss; train/grad_norm, the total norm before clipping at 1.0,
  which for AME includes the bank gradients (src/core/memory_parallel.py, O4);
  train/lr; train/step_time.
- every 100 steps, accumulated over all 16 micro-batches of that step:
  block[i]/residual_{attention,feedforward}/{residual_norms, update_norms,
  update_to_residual_ratio}/{mean, std}. These are per-token L2 norms of the stream
  entering the sublayer and of the sublayer's output, and their per-token ratio.
- every 100 steps: eval/loss (10 batches, bf16-quantized: steps of 1/64 near 3)
  and AME's per-layer readouts (tau, alpha_2, beta_2, horizon, entropy, top-8 mass).
- once: final_ce, on 8,192 x 2,048 held-out tokens.
"""

import os
import sys

import numpy as np
import pandas as pd

PROJECT = "ideas_cv/nano-ame"
RUNS = {"AME": "doi36p8k", "dense": "x0xlqroj"}
N_BLOCKS = 12
CLIP = 1.0
SMOOTH = 25  # steps, for the per-step series in the figures
CACHE = os.environ.get(
    "AME_COMPARISON_CACHE", os.path.expanduser("~/.cache/nano/ame_comparison_h12")
)

# Light chart surface and ink. Series colours: categorical slots 1 and 2, checked
# with the dataviz palette validator (CVD dE 24.7, normal-vision dE 33.6, both >= 3:1).
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
COLOUR = {"dense": "#2a78d6", "AME": "#eb6834"}
# Diverging: blue where dense is larger, red where AME is, a gray midpoint.
DIVERGING = ["#2a78d6", "#f0efec", "#e34948"]


# --------------------------------------------------------------------------- data


def load(label):
    """The run's full history, steps as the index, the `steps/` prefix dropped."""
    rid = RUNS[label]
    path = os.path.join(CACHE, f"{rid}.parquet")
    if not os.path.exists(path):
        import wandb

        run = wandb.Api(timeout=180).run(f"{PROJECT}/{rid}")
        df = pd.DataFrame(list(run.scan_history(page_size=2000)))
        df = df[[c for c in df.columns if not c.startswith("tokens/")]]
        os.makedirs(CACHE, exist_ok=True)
        df.to_parquet(path)
    df = pd.read_parquet(path).set_index("step").sort_index()
    df.columns = [c.removeprefix("steps/") for c in df.columns]
    return df


def series(df, key):
    return df[key].dropna()


def per_block(df, sub, metric):
    """[logged step x block] for one Residual metric; sub is attention or feedforward."""
    cols = {b: df[f"block[{b}]/residual_{sub}/{metric}/mean"] for b in range(N_BLOCKS)}
    return pd.DataFrame(cols).dropna(how="all")


def ame_readout(df, key):
    cols = {
        b: df[f"encoder.blocks.{b}.attention_layer.layer/{key}"] for b in range(N_BLOCKS)
    }
    return pd.DataFrame(cols).dropna(how="all")


def paired_gap(a, d):
    """AME minus dense train loss, step by step, on the same batches."""
    return (series(a, "train/loss") - series(d, "train/loss")).dropna()


def crossover(gap):
    """The first step after which the 50-step mean gap stays above zero."""
    smooth = gap.rolling(50, center=True, min_periods=1).mean()
    below = smooth[smooth <= 0]
    return int(below.index.max()) + 1 if len(below) else int(gap.index.min())


def spikes(df, window=50, threshold=0.5):
    """Steps whose train loss exceeds the median of the previous `window` steps by more
    than `threshold`: the rule of stage_1_report.md, here on the full history."""
    loss = series(df, "train/loss")
    excess = loss - loss.rolling(window).median().shift(1)
    return excess[excess > threshold]


# ------------------------------------------------------------------------- tables


def _md(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


PHASES = [
    (0, 38), (38, 100), (100, 300), (300, 600), (600, 1000), (1000, 1500),
    (1500, 2000), (2000, 2500), (2500, 3000), (3000, 3500), (3500, 3815),
]


def cmd_tables():
    a, d = load("AME"), load("dense")

    print("## Headline\n")
    rows = []
    for label, df in (("AME", a), ("dense", d)):
        st = series(df, "train/step_time").loc[100:]
        rows.append([
            label, RUNS[label], f"{series(df, 'final_ce').iloc[-1]:.4f}",
            f"{series(df, 'eval/loss').iloc[-1]:.4f}", f"{st.median():.3f}",
            f"{series(df, 'train/max_memory_gib').iloc[-1]:.1f}",
            f"{df['_runtime'].max() / 3600:.2f}",
        ])
    print(_md(rows, ["run", "W&B", "final_ce", "last eval", "s/step", "peak GiB", "runtime h"]))

    gap = paired_gap(a, d)
    print(f"\ncrossover (50-step mean gap stays > 0 from): step {crossover(gap)}")
    late = gap.loc[1500:]
    slope = np.polyfit(late.index.values, late.values, 1)[0] * 1000
    print(f"gap slope over steps 1500-3814: {slope:+.4f} nats per 1,000 steps")
    ea, ed = series(a, "eval/loss"), series(d, "eval/loss")
    target = ea.iloc[-1]
    hit = ed[ed <= target]
    if len(hit):
        s1 = hit.index[0]
        s0 = ed.index[ed.index.get_loc(s1) - 1]
        frac = (ed.loc[s0] - target) / (ed.loc[s0] - ed.loc[s1])
        step = s0 + frac * (s1 - s0)
        print(f"dense eval first reaches AME's last eval {target:.4f} at step ~{step:.0f} "
              f"({step / 3815:.0%} of the run, {step * 1048576 / 1e9:.2f}B tokens)")

    print("\n## Paired train-loss gap and grad norm by phase\n")
    ga, gd = series(a, "train/grad_norm"), series(d, "train/grad_norm")
    rows = []
    for lo, hi in PHASES:
        g = gap.loc[lo:hi - 1]
        rows.append([
            f"{lo}-{hi - 1}", f"{g.mean():+.4f}", f"{(g < 0).mean():.0%}",
            f"{ga.loc[lo:hi - 1].median():.3f}", f"{gd.loc[lo:hi - 1].median():.3f}",
            f"{ga.loc[lo:hi - 1].median() / gd.loc[lo:hi - 1].median():.2f}",
            f"{ga.loc[lo:hi - 1].max():.2f}", f"{gd.loc[lo:hi - 1].max():.2f}",
            f"{int((ga.loc[lo:hi - 1] > CLIP).sum())} / {int((gd.loc[lo:hi - 1] > CLIP).sum())}",
        ])
    print(_md(rows, ["steps", "gap AME-dense", "AME lower", "gnorm median AME",
                     "dense", "ratio", "max AME", "max dense", "clipped AME / dense"]))

    print("\n## Loss spikes (stage_1_report.md rule, full history)\n")
    for label, df in (("AME", a), ("dense", d)):
        s = spikes(df)
        loss = series(df, "train/loss")
        print(f"{label}: " + (", ".join(f"step {k} loss {loss.loc[k]:.2f} (+{v:.2f})"
                                       for k, v in s.items()) or "none"))

    end = 3800
    print(f"\n## Per block at step {end}\n")
    rows = []
    blocks = {}
    for label, df in (("AME", a), ("dense", d)):
        blocks[label] = {
            "stream": per_block(df, "attention", "residual_norms").loc[end],
            "attn": per_block(df, "attention", "update_to_residual_ratio").loc[end],
            "ff": per_block(df, "feedforward", "update_to_residual_ratio").loc[end],
            "attn_u": per_block(df, "attention", "update_norms").loc[end],
            "ff_u": per_block(df, "feedforward", "update_norms").loc[end],
        }
    for b in range(N_BLOCKS):
        A, D = (blocks[k] for k in ("AME", "dense"))
        rows.append([b] + [
            f"{A[m][b]:.{p}f} / {D[m][b]:.{p}f} ({A[m][b] / D[m][b]:.2f})"
            for m, p in (("stream", 0), ("attn_u", 1), ("attn", 3), ("ff_u", 1), ("ff", 3))
        ])
    print(_md(rows, ["block", "stream norm into attention", "attention update norm",
                     "attention update/stream", "FF update norm", "FF update/stream"]))

    print(f"\n## AME readouts per layer, step 0 -> step {end}\n")
    keys = ["tau/mean", "alpha_2/mean", "alpha_2/absmean", "beta_2/mean",
            "attention_entropy/normalized", "top8_mass/mean"]
    tabs = {k: ame_readout(a, k) for k in keys}
    rows = [[b] + [f"{tabs[k].loc[0][b]:.2f} -> {tabs[k].loc[end][b]:.2f}" for k in keys]
            for b in range(N_BLOCKS)]
    print(_md(rows, ["layer", "tau", "alpha_2 mean", "alpha_2 |mean|", "beta_2 mean",
                     "entropy (normalized)", "top-8 mass"]))


# ------------------------------------------------------------------------ figures


def _style(ax, xlabel=None, ylabel=None, title=None):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left")


def _end_labels(ax, ends, min_px=12):
    """Direct labels at line ends, in ink, pushed apart so they never collide.
    `ends` is [(x, y, text)] in data coordinates."""
    fig = ax.figure
    fig.canvas.draw()
    placed = []
    for x, y, text in sorted(ends, key=lambda e: ax.transData.transform((e[0], e[1]))[1]):
        px = ax.transData.transform((x, y))[1]
        target = px if not placed else max(px, placed[-1] + min_px)
        placed.append(target)
        ax.annotate(text, (x, y), xytext=(6, (target - px) * 72 / fig.dpi),
                    textcoords="offset points", va="center", fontsize=8, color=INK2)


def _hline(ax, y, text):
    ax.axhline(y, color=MUTED, linewidth=0.8, zorder=1)
    ax.annotate(text, (1.0, y), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                textcoords="offset points", ha="right", fontsize=7.5, color=MUTED)


def _save(fig, out_dir, name, title):
    fig.suptitle(title, fontsize=11, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print(f"wrote {path}")


def fig_loss(a, d, out_dir, plt):
    fig, (top, bot) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [1.15, 1]})
    fig.patch.set_facecolor(SURFACE)
    ends = []
    for label, df in (("dense", d), ("AME", a)):
        loss = series(df, "train/loss").rolling(SMOOTH, center=True, min_periods=1).mean()
        loss = loss.loc[30:]
        top.plot(loss.index, loss.values, color=COLOUR[label], linewidth=1.8, label=label)
        ends.append((loss.index[-1], loss.iloc[-1],
                     f"{label}  final_ce {series(df, 'final_ce').iloc[-1]:.4f}"))
    top.set_ylim(2.9, 7.0)
    _style(top, ylabel=f"train loss ({SMOOTH}-step mean)",
           title="Train loss: the same batches, in the same order")
    top.legend(fontsize=8, frameon=False, loc="upper right")
    top.margins(x=0.02)
    top.set_xlim(0, 4500)
    _end_labels(top, ends)

    gap = paired_gap(a, d)
    smooth = gap.rolling(SMOOTH, center=True, min_periods=1).mean()
    bot.plot(smooth.index, smooth.values, color=INK, linewidth=1.6,
             label=f"train loss, {SMOOTH}-step mean of the per-step gap")
    eg = (series(a, "eval/loss") - series(d, "eval/loss")).dropna()
    bot.scatter(eg.index, eg.values, s=14, color=MUTED, zorder=3,
                label="eval loss (10 batches, bf16 steps of 1/64)")
    _hline(bot, 0, "")
    x0 = crossover(gap)
    bot.axvline(x0, color=AXIS, linewidth=0.8)
    bot.annotate(f"AME behind from step {x0}", (x0, -0.1), xytext=(6, 0),
                 textcoords="offset points", fontsize=8, color=INK2)
    floor = -0.35
    low = smooth.loc[180:230].min()
    bot.annotate(f"dense loss spike, steps 194-212\n(the gap reaches {low:.2f}, off scale)",
                 (198, floor), xytext=(150, 22), textcoords="offset points", fontsize=8,
                 color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7))
    bot.set_ylim(floor, 0.2)
    _style(bot, xlabel="step  (1 step = 1,048,576 tokens)",
           ylabel="AME minus dense (nats)",
           title="The gap: below zero, AME is ahead")
    bot.legend(fontsize=8, frameon=False, loc="lower right")
    _save(fig, out_dir, "h12_loss.png",
          "AME (S2-PS) against dense (S1-D), h12, n = 8: loss")
    plt.close(fig)


def fig_grad_norm(a, d, out_dir, plt):
    fig, (top, bot) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [1.4, 1]})
    fig.patch.set_facecolor(SURFACE)
    meds = {}
    ends = []
    for label, df in (("dense", d), ("AME", a)):
        g = series(df, "train/grad_norm")
        top.plot(g.index, g.values, color=COLOUR[label], linewidth=0.5, alpha=0.3)
        med = g.rolling(SMOOTH, center=True, min_periods=1).median()
        meds[label] = med
        top.plot(med.index, med.values, color=COLOUR[label], linewidth=1.8, label=label)
        ends.append((med.index[-1], med.iloc[-1], f"{label}  {med.iloc[-1]:.3f}"))
    top.set_yscale("log")
    _hline(top, CLIP, "clipping threshold 1.0")
    top.annotate("step 10: one batch, both spike\n(AME 46.5, dense 15.9)", (10, 40),
                 xytext=(40, -4), textcoords="offset points", fontsize=8, color=INK2,
                 arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7))
    top.annotate("dense, steps 194-200\n(peak 2.58 at step 198)", (198, 2.58),
                 xytext=(40, 6), textcoords="offset points", fontsize=8, color=INK2,
                 arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7))
    top.set_xlim(0, 4500)
    _style(top, ylabel="grad norm, before clipping",
           title=f"Total gradient norm: every step (faint) and its {SMOOTH}-step median")
    top.legend(fontsize=8, frameon=False, loc="upper right")
    _end_labels(top, ends)

    ratio = (meds["AME"] / meds["dense"]).loc[38:]
    bot.plot(ratio.index, ratio.values, color=INK, linewidth=1.6)
    _hline(bot, 1.0, "equal")
    bot.set_ylim(0.4, 1.6)
    _style(bot, xlabel="step", ylabel="AME / dense",
           title=f"Ratio of the {SMOOTH}-step medians, after warmup")
    _save(fig, out_dir, "h12_grad_norm.png",
          "AME (S2-PS) against dense (S1-D), h12, n = 8: gradient norm")
    plt.close(fig)


def fig_depth(a, d, out_dir, plt, step=3800):
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter

    ticks = {"attention": [0.08, 0.1, 0.15, 0.2, 0.3, 0.4],
             "feedforward": [0.25, 0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, 3]}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    fig.patch.set_facecolor(SURFACE)
    panels = [
        ("attention", "residual_norms", "stream norm entering the block", False),
        ("attention", "update_to_residual_ratio", "attention update / stream", True),
        ("feedforward", "update_to_residual_ratio", "feed-forward update / stream", True),
    ]
    blocks = np.arange(N_BLOCKS)
    for ax, (sub, metric, title, log) in zip(axes, panels):
        ends = []
        for label, df in (("dense", d), ("AME", a)):
            v = per_block(df, sub, metric).loc[step].values
            ax.plot(blocks, v, color=COLOUR[label], linewidth=1.8, marker="o",
                    markersize=4.5, label=label)
            ends.append((blocks[-1], v[-1], label))
        if log:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(FixedLocator(ticks[sub]))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(blocks)
        ax.set_xlim(-0.5, N_BLOCKS + 0.8)
        _style(ax, xlabel="block", title=title)
        _end_labels(ax, ends)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    _save(fig, out_dir, "h12_depth_profile.png",
          f"Per block, step {step} (end of training), mean over the step's 16 micro-batches")
    plt.close(fig)


def fig_ratio_heatmap(a, d, out_dir, plt):
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("ame_dense", DIVERGING)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    fig.patch.set_facecolor(SURFACE)
    for ax, sub, title in ((axes[0], "attention", "attention update / stream"),
                           (axes[1], "feedforward", "feed-forward update / stream")):
        ra = per_block(a, sub, "update_to_residual_ratio")
        rd = per_block(d, sub, "update_to_residual_ratio")
        z = np.log2((ra / rd).T.values)  # [block x step]
        steps = ra.index.values
        im = ax.imshow(z, aspect="auto", cmap=cmap, vmin=-2, vmax=2, interpolation="nearest",
                       extent=(steps[0] - 50, steps[-1] + 50, N_BLOCKS - 0.5, -0.5))
        ax.set_yticks(range(N_BLOCKS))
        _style(ax, xlabel="step", ylabel="block", title=title)
        ax.grid(False)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, ticks=[-2, -1, 0, 1, 2])
    cbar.ax.set_yticklabels(["1/4x", "1/2x", "1x", "2x", "4x"], fontsize=8, color=INK2)
    cbar.set_label("AME / dense   (red: AME writes more; blue: dense does)", fontsize=8,
                   color=INK2)
    cbar.outline.set_visible(False)
    fig.suptitle("How much each sublayer writes into the residual stream, AME relative to "
                 "dense", fontsize=11, x=0.01, ha="left", color=INK)
    path = os.path.join(out_dir, "h12_ratio_heatmap.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {path}")
    plt.close(fig)


def fig_stream(a, d, out_dir, plt, blocks=(2, 6, 11)):
    fig, axes = plt.subplots(1, len(blocks), figsize=(13, 4.0), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, b in zip(axes, blocks):
        ends = []
        for label, df in (("dense", d), ("AME", a)):
            v = per_block(df, "attention", "residual_norms")[b]
            ax.plot(v.index, v.values, color=COLOUR[label], linewidth=1.8, label=label)
            ends.append((v.index[-1], v.iloc[-1], f"{label}  {v.iloc[-1]:.0f}"))
        ax.set_yscale("log")
        ax.set_xlim(0, 4700)
        _style(ax, xlabel="step", title=f"entering block {b}")
        _end_labels(ax, ends)
    axes[0].set_ylabel("residual stream norm, per token", fontsize=9, color=INK2)
    axes[0].legend(fontsize=8, frameon=False, loc="upper right")
    _save(fig, out_dir, "h12_stream_norm.png",
          "Residual stream norm over training (logged every 100 steps)")
    plt.close(fig)


def fig_ame_readouts(a, out_dir, plt, step=3800):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8))
    fig.patch.set_facecolor(SURFACE)
    layers = np.arange(N_BLOCKS)
    panels = [
        ("alpha_2", "alpha_2: weight on the previous token's key"),
        ("beta_2", "beta_2: weight on the previous token's value"),
        ("tau", "tau: score scale (init 8 = sqrt(d_head))"),
        ("attention_entropy", "attention entropy, normalized (1 = uniform)"),
    ]
    for ax, (key, title) in zip(axes.flat, panels):
        if key == "attention_entropy":
            mean, lo, hi = ame_readout(a, "attention_entropy/normalized"), None, None
        elif key == "tau":
            mean, lo, hi = ame_readout(a, "tau/mean"), ame_readout(a, "tau/min"), None
        else:
            mean = ame_readout(a, f"{key}/mean")
            lo, hi = ame_readout(a, f"{key}/min"), ame_readout(a, f"{key}/max")
        ax.plot(layers, mean.loc[0].values, color=MUTED, linewidth=1.4, marker="o",
                markersize=4, label="step 0 (init), mean over heads")
        ax.plot(layers, mean.loc[step].values, color=COLOUR["AME"], linewidth=1.8,
                marker="o", markersize=4.5, label=f"step {step}, mean over heads")
        if lo is not None and hi is not None:
            ax.fill_between(layers, lo.loc[step].values, hi.loc[step].values,
                            color=COLOUR["AME"], alpha=0.15, linewidth=0,
                            label=f"step {step}, min to max over heads")
        elif lo is not None:
            ax.plot(layers, lo.loc[step].values, color=COLOUR["AME"], linewidth=1.0,
                    alpha=0.6, label=f"step {step}, min over heads")
            ax.annotate("min over heads", (layers[-1], lo.loc[step].values[-1]),
                        xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=8, color=INK2)
            ax.set_xlim(-0.5, N_BLOCKS + 1.2)
        ax.set_xticks(layers)
        _style(ax, xlabel="layer", title=title)
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="upper left")
    _save(fig, out_dir, "h12_ame_readouts.png",
          "AME (S2-PS) only: its per-layer readouts, start and end of training")
    plt.close(fig)


def cmd_figures(out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    a, d = load("AME"), load("dense")
    fig_loss(a, d, out_dir, plt)
    fig_grad_norm(a, d, out_dir, plt)
    fig_depth(a, d, out_dir, plt)
    fig_ratio_heatmap(a, d, out_dir, plt)
    fig_stream(a, d, out_dir, plt)
    fig_ame_readouts(a, out_dir, plt)


def main(argv):
    if len(argv) >= 2 and argv[1] == "tables":
        cmd_tables()
    elif len(argv) == 3 and argv[1] == "figures":
        cmd_figures(argv[2])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
