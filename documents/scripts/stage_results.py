"""Stage 1-2 results from W&B: the numbers and figures in stage_1_report.md and
stage_2_report.md (documents/notes/).

Read-only. Runs are named by their W&B id in ideas_cv/nano-ame. The logged config
is the truth (ame_experiments.md, "Reporting"), so every row is built from what
the run logged, never from the config file's name.

    python documents/scripts/stage_results.py table <id> [<id> ...]
    python documents/scripts/stage_results.py evals <id> [<id> ...]
    python documents/scripts/stage_results.py readouts <id> [<id> ...]
    python documents/scripts/stage_results.py sweep OUT.png TITLE CHOSEN_N \
        "<arm label>=<n>:<id>,<n>:<id>,..." ["<arm label>=..." ...]
    python documents/scripts/stage_results.py curves OUT.png TITLE \
        "<panel title>=<id>:<label>,<id>:<label>,..." ["<panel title>=..." ...]

Run it with the project env (`./.pixi/envs/default/bin/python`); WANDB_API_KEY
must be set.

Conventions:
- s/step is the median of `train/step_time` over steps >= 100: the median drops
  the steps that carry an eval, and >= 100 drops compile and warmup.
- MFU is computed here, not read from W&B: LEM's GPUs report as "NVIDIA H100",
  which src/core/mfu.py's peak table does not know, so W&B has no MFU for LEM
  runs. The peak is the H100 SXM dense BF16 figure, 989 TFLOP/s.
- FLOPs per token are the trainer's own (src/core/mfu.py), passed in per arm.
"""

import os
import re
import sys

PROJECT = "ideas_cv/nano-ame"
TOKENS_PER_STEP = 512 * 2048
H100_SXM_PEAK = 989e12
# Training FLOPs per token at seq 2048, as the trainer logged them on LEM, keyed by
# (attention type, h): stage_1_report.md (h12) and stage_1b_report.md (h16) smoke runs.
# Every AME arm of one size has the same; NoPE dense has the same as RoPE dense.
GFLOPS_PER_TOKEN = {("gqa", 12): 1.095, ("ame", 12): 1.053, ("gqa", 16): 2.222, ("ame", 16): 2.121}

# First step plotted by `curves`: the first evals are far above the rest and would
# flatten every curve. STAGE_CURVES_FROM overrides it (e.g. 0 for a smoke run).
FROM_STEP = int(os.environ.get("STAGE_CURVES_FROM", "300"))

LAYER_KEY = re.compile(r"^steps/encoder\.blocks\.(\d+)\.attention_layer\.layer/(.+)$")


def _api():
    import wandb

    return wandb.Api(timeout=120)


def _attention(cfg):
    att = ((cfg.get("common") or {}).get("attention") or {})
    ame = att.get("ame") or {}
    return att.get("type"), ame


def describe(run):
    """The run's identity, from its logged config."""
    cfg = run.config or {}
    trainer = cfg.get("trainer") or {}
    dataset = ((trainer.get("train_dataloader") or {}).get("dataset") or {})
    att_type, ame = _attention(cfg)
    slopes = ame.get("slopes")
    if att_type == "ame":
        pe = "none" if slopes is not None and all(s == 0 for s in slopes) else (
            "ALiBi" if slopes is None else f"slopes {slopes}")
        shift = "shift" if ame.get("shift") else "no shift"
        arm = f"AME, {pe}, {shift}"
    else:
        # gqa takes an options block; `rope: false` is NoPE (src/core/attention.py).
        gqa = ((cfg.get("common") or {}).get("attention") or {}).get("gqa") or {}
        arm = f"{att_type} ({'RoPE' if gqa.get('rope', True) else 'NoPE'})"
    return {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "arm": arm,
        "att": att_type,
        "n": trainer.get("learning_rate"),
        "seed": dataset.get("seed"),
        "h": (cfg.get("common") or {}).get("h"),
        "n_steps": trainer.get("n_steps"),
        "ga": trainer.get("gradient_accumulation_steps"),
        "distributed": trainer.get("distributed"),
    }


def summarize(run):
    s = run.summary
    d = describe(run)
    d["final_ce"] = s.get("steps/final_ce")
    d["last_eval"] = s.get("steps/eval/loss")
    d["last_step"] = s.get("step")
    d["peak_gib"] = s.get("steps/train/max_memory_gib")
    times = [
        row["steps/train/step_time"]
        for row in run.scan_history(keys=["step", "steps/train/step_time"])
        if row.get("step", 0) >= 100 and row.get("steps/train/step_time") is not None
    ]
    times.sort()
    d["s_per_step"] = times[len(times) // 2] if times else None
    key = (d["att"], d["h"])
    if d["s_per_step"] and key in GFLOPS_PER_TOKEN:
        tps = TOKENS_PER_STEP / d["s_per_step"]
        d["tokens_per_s"] = tps
        d["mfu"] = tps * GFLOPS_PER_TOKEN[key] * 1e9 / H100_SXM_PEAK
    else:
        d["tokens_per_s"] = d["mfu"] = None
    return d


def evals(run):
    """[(step, eval loss)], every eval the run logged."""
    return [
        (int(row["step"]), float(row["steps/eval/loss"]))
        for row in run.scan_history(keys=["step", "steps/eval/loss"])
        if row.get("steps/eval/loss") is not None
    ]


def layer_readouts(run):
    """{key: [value per layer]} at the end of the run, from the summary."""
    out = {}
    for key, value in run.summary.items():
        m = LAYER_KEY.match(key)
        if m and isinstance(value, (int, float)):
            out.setdefault(m.group(2), {})[int(m.group(1))] = float(value)
    return {k: [v[i] for i in sorted(v)] for k, v in out.items()}


def readout_history(run, key):
    """{step: [value per layer]} for one readout key, e.g. 'tau/mean'."""
    layers = sorted({
        int(m.group(1)) for k in run.summary.keys()
        if (m := LAYER_KEY.match(k)) and m.group(2) == key
    })
    keys = [f"steps/encoder.blocks.{i}.attention_layer.layer/{key}" for i in layers]
    hist = {}
    for row in run.scan_history(keys=["step"] + keys):
        vals = [row.get(k) for k in keys]
        if all(v is not None for v in vals):
            hist[int(row["step"])] = [float(v) for v in vals]
    return hist


def _fmt(v, spec):
    return "—" if v is None else format(v, spec)


def cmd_table(ids):
    api = _api()
    print("| W&B run | name | arm | n | seed | steps | final_ce | last eval | s/step | MFU | peak GiB | state |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for rid in ids:
        d = summarize(api.run(f"{PROJECT}/{rid}"))
        print(f"| `{d['id']}` | {d['name']} | {d['arm']} | {d['n']} | {d['seed']} | "
              f"{d['last_step']} | {_fmt(d['final_ce'], '.4f')} | {_fmt(d['last_eval'], '.4f')} | "
              f"{_fmt(d['s_per_step'], '.3f')} | {_fmt(d['mfu'] and d['mfu'] * 100, '.1f')}% | "
              f"{_fmt(d['peak_gib'], '.2f')} | {d['state']} |")


def cmd_evals(ids):
    api = _api()
    for rid in ids:
        run = api.run(f"{PROJECT}/{rid}")
        ev = evals(run)
        print(f"{rid} {run.name}: " + ", ".join(f"{s}:{v:.4f}" for s, v in ev))


def cmd_readouts(ids):
    api = _api()
    for rid in ids:
        run = api.run(f"{PROJECT}/{rid}")
        r = layer_readouts(run)
        print(f"== {rid} {run.name}")
        for key in ("tau/mean", "tau/min", "attention_entropy/normalized",
                    "attention_entropy/nats", "top8_mass/mean", "alpha_2/absmean",
                    "beta_2/absmean", "alpha_2/min", "beta_2/min", "horizon/min", "horizon/max"):
            if key in r:
                vals = r[key]
                print(f"   {key:30s} mean {sum(vals) / len(vals):9.4f}  per layer "
                      + " ".join(f"{v:.3g}" for v in vals))


# Categorical slots 1-4 of the reference palette (dataviz skill, light mode), in
# fixed order: the same n always gets the same colour, in every panel.
_SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def cmd_curves(out, title, panels):
    """One panel per arm; eval loss against step, from FROM_STEP on."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    api = _api()
    parsed = []
    labels_in_order = []
    for spec in panels:
        ptitle, runs = spec.split("=", 1)
        entries = []
        for item in runs.split(","):
            rid, label = item.split(":", 1)
            entries.append((rid, label, evals(api.run(f"{PROJECT}/{rid}"))))
            if label not in labels_in_order:
                labels_in_order.append(label)
        parsed.append((ptitle, entries))
    colour = {lab: _SLOT[i % len(_SLOT)] for i, lab in enumerate(labels_in_order)}

    fig, axes = plt.subplots(1, len(parsed), figsize=(5.2 * len(parsed), 4.2), sharey=True,
                             squeeze=False)
    lo, hi = float("inf"), float("-inf")
    ends = []  # per panel: [(x, y, text)], placed once the shared y-range is known
    for ax, (ptitle, entries) in zip(axes[0], parsed):
        panel_ends = []
        for rid, label, ev in entries:
            pts = [(s, v) for s, v in ev if s >= FROM_STEP]
            if not pts:
                continue
            xs, ys = zip(*pts)
            lo, hi = min(lo, min(ys)), max(hi, max(ys))
            ax.plot(xs, ys, color=colour[label], linewidth=2, label=label)
            panel_ends.append((xs[-1], ys[-1], f"{label}  {ys[-1]:.3f}"))
        ends.append(panel_ends)
        ax.set_title(ptitle, fontsize=10, color="#0b0b0b", loc="left")
        ax.set_xlabel("step", fontsize=9, color="#52514e")
        ax.grid(True, color="#e6e5e0", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#b9b8b2")
        ax.tick_params(colors="#52514e", labelsize=8)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        ax.margins(x=0.18)
    axes[0][0].set_ylabel("eval loss (10 batches)", fontsize=9, color="#52514e")
    if lo < hi:
        pad = (hi - lo) * 0.05
        top = hi + pad
        # A loss spike can stretch the axis and flatten every curve; STAGE_CURVES_YMAX
        # cuts the axis there (say so in the caption).
        if os.environ.get("STAGE_CURVES_YMAX"):
            top = min(top, float(os.environ["STAGE_CURVES_YMAX"]))
        axes[0][0].set_ylim(lo - pad, top)
    # End labels, pushed apart so close final losses stay readable: sorted by
    # value, each at least `gap` above the one below it. Only on panels of at most
    # four series; past that they collide, and the legend carries identity alone.
    y0, y1 = axes[0][0].get_ylim()
    gap = (y1 - y0) * 0.045
    for ax, panel_ends in zip(axes[0], ends):
        if len(panel_ends) > 4:
            continue
        x0, x1 = ax.get_xlim()
        dx = (x1 - x0) * 0.015
        placed = None
        for x, y, text in sorted(panel_ends, key=lambda e: e[1]):
            ty = y if placed is None else max(y, placed + gap)
            placed = ty
            ax.annotate(text, (x, y), xytext=(x + dx, ty), textcoords="data", va="center",
                        fontsize=8, color="#52514e")
    fig.suptitle(title, fontsize=11, x=0.01, ha="left", color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="#fcfcfb")
    print(f"wrote {out}")


def cmd_sweep(out, title, chosen_n, series_specs):
    """final_ce against the learning-rate exponent n, one line per arm.

    Each spec is "<label>=<n>:<id>,<n>:<id>,...". The lowest point of each arm is
    ringed, and `chosen_n` (the n* actually used downstream) is a dashed line.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    api = _api()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, spec in enumerate(series_specs):
        label, runs = spec.split("=", 1)
        pts = []
        for item in runs.split(","):
            n, rid = item.split(":", 1)
            pts.append((int(n), float(api.run(f"{PROJECT}/{rid}").summary.get("steps/final_ce"))))
        pts.sort()
        xs, ys = zip(*pts)
        colour = _SLOT[i % len(_SLOT)]
        ax.plot(xs, ys, color=colour, linewidth=2, marker="o", markersize=5, label=label)
        best_n, best = min(pts, key=lambda p: p[1])
        ax.plot([best_n], [best], marker="o", markersize=13, markerfacecolor="none",
                markeredgecolor=colour, markeredgewidth=1.5)
        ax.annotate(f"best n = {best_n}: {best:.3f}", (best_n, best), xytext=(10, 8),
                    textcoords="offset points", ha="left", fontsize=8, color="#52514e")
    ax.axvline(float(chosen_n), color="#8a8984", linewidth=1, linestyle="--")
    ax.annotate(f"n* = {chosen_n} (used)", (float(chosen_n), ax.get_ylim()[1]), xytext=(4, -12),
                textcoords="offset points", fontsize=8, color="#52514e")
    ax.set_xlabel("n, with learning rate 2^-n (higher LR to the left)", fontsize=9, color="#52514e")
    ax.set_ylabel("final_ce (8192 x 2048 held-out tokens)", fontsize=9, color="#52514e")
    ax.set_title(title, fontsize=10, color="#0b0b0b", loc="left")
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b9b8b2")
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="#fcfcfb")
    print(f"wrote {out}")


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd, rest = argv[1], argv[2:]
    if cmd == "table":
        cmd_table(rest)
    elif cmd == "evals":
        cmd_evals(rest)
    elif cmd == "readouts":
        cmd_readouts(rest)
    elif cmd == "curves":
        cmd_curves(rest[0], rest[1], rest[2:])
    elif cmd == "sweep":
        cmd_sweep(rest[0], rest[1], rest[2], rest[3:])
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
