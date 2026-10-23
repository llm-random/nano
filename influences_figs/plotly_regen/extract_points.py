"""Recover (channel, value) data points from the matplotlib influence-chart PDFs.

Each point is a marker XObject placed with `1 0 -0 1 X Y cm /M0 Do` (X, Y = marker centre in
PDF points). Axis mapping: tick marks are short line segments on the axes in the content stream,
tick labels come from `pdftotext -bbox`; a linear fit tick-position -> label value per axis
converts marker centres to data coordinates.
"""

import json
import re
import subprocess
import sys
import zlib
from html import unescape

import numpy as np

NUM = r"-?\d+(?:\.\d+)?"


def content_stream(path):
    raw = open(path, "rb").read()
    best = b""
    for m in re.finditer(rb"stream\r?\n", raw):
        end = raw.find(b"endstream", m.end())
        try:
            d = zlib.decompress(raw[m.end() : end])
        except Exception:
            continue
        if b" Do" in d and len(d) > len(best):
            best = d
    return re.sub(r"\s+", " ", best.decode("latin-1")), raw


def page_height(path):
    out = subprocess.run(["pdfinfo", path], capture_output=True, text=True).stdout
    return float(re.search(r"Page size:\s+[\d.]+ x ([\d.]+)", out).group(1))


def words(path, H):
    html = subprocess.run(
        ["pdftotext", "-bbox", path, "-"], capture_output=True, text=True
    ).stdout
    res = []
    for m in re.finditer(
        r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)</word>',
        html,
    ):
        x0, y0, x1, y1 = map(float, m.groups()[:4])
        # convert to PDF coords (origin bottom-left)
        res.append(dict(t=unescape(m.group(5)), cx=(x0 + x1) / 2, cy=H - (y0 + y1) / 2))
    return res


def to_float(t):
    t = t.replace("−", "-").replace("−", "-")
    try:
        return float(t)
    except ValueError:
        return None


def extract(path):
    s, _ = content_stream(path)
    H = page_height(path)

    # axes frame: the patch "x0 y0 m x1 y0 l x1 y1 l x0 y1 l h" drawn as background of the axes
    frame = re.search(
        rf"({NUM}) ({NUM}) m ({NUM}) \2 l \3 ({NUM}) l \1 \4 l h f",
        s.split("/A1 gs", 1)[1],
    )
    ax_x0, ax_y0, ax_x1, ax_y1 = (float(frame.group(i)) for i in (1, 2, 3, 4))

    # marker centres
    # markers are drawn inside q ... Q blocks with CUMULATIVE translations:
    #   q 1 0 0 1 x0 y0 cm /M0 Do 1 0 0 1 dx1 dy1 cm /M0 Do ... Q   -> positions = running sums
    pts = []
    for block in re.findall(rf"q ((?:1 0 -?0 1 {NUM} {NUM} cm /M\d+ Do ?)+)Q", s):
        x = y = 0.0
        for a, b in re.findall(rf"1 0 -?0 1 ({NUM}) ({NUM}) cm /M\d+ Do", block):
            x += float(a)
            y += float(b)
            pts.append((x, y))
    pts = np.array(pts)

    # tick marks: matplotlib draws them as markers too (/P0, /P1 ...) or as tiny segments.
    # Collect x-tick positions (segments starting on the bottom spine going down) and y-tick (left spine).
    xt = [
        float(a)
        for a, b, c, d in re.findall(rf"({NUM}) ({NUM}) m ({NUM}) ({NUM}) l", s)
        if abs(float(b) - ax_y0) < 1e-3
        and float(d) < float(b)
        and abs(float(a) - float(c)) < 1e-6
    ]
    yt = [
        float(b)
        for a, b, c, d in re.findall(rf"({NUM}) ({NUM}) m ({NUM}) ({NUM}) l", s)
        if abs(float(a) - ax_x0) < 1e-3
        and float(c) < float(a)
        and abs(float(b) - float(d)) < 1e-6
    ]

    ws = words(path, H)
    # x tick labels: numeric words below the axes, centred under a tick
    xlab = [
        (w["cx"], to_float(w["t"]))
        for w in ws
        if w["cy"] < ax_y0 and to_float(w["t"]) is not None
    ]
    ylab = [
        (w["cy"], to_float(w["t"]))
        for w in ws
        if w["cx"] < ax_x0 and to_float(w["t"]) is not None
    ]

    def fit(ticks, labels, name):
        pairs = []
        for t in ticks:
            if not labels:
                break
            pos, val = min(labels, key=lambda l: abs(l[0] - t))
            if abs(pos - t) < 6:  # label centred on its tick (pt)
                pairs.append((t, val))
        pairs = sorted(set(pairs))
        if len(pairs) < 2:
            raise RuntimeError(
                f"{path}: could not match {name} ticks to labels: ticks={ticks} labels={labels}"
            )
        P = np.array(pairs)
        # tick labels may be ROUNDED (e.g. 1.25e-3 shown as "1.3e-03"): iteratively drop the worst
        # label while the fit is not exact; ticks are evenly spaced, so >=2 exact labels suffice.
        keep = np.ones(len(P), bool)
        while True:
            a, b = np.polyfit(P[keep, 0], P[keep, 1], 1)
            r = np.abs(a * P[:, 0] + b - P[:, 1])
            span = np.ptp(P[:, 1])
            if r[keep].max() <= 1e-9 * span or keep.sum() <= 2:
                break
            keep[np.argmax(np.where(keep, r, -1))] = False
        return a, b, [tuple(x) for x in P[keep]], float(r[keep].max())

    ax_, bx_, xpairs, xres = fit(xt, xlab, "x")
    ay_, by_, ypairs, yres = fit(yt, ylab, "y")
    xs = ax_ * pts[:, 0] + bx_
    ys = ay_ * pts[:, 1] + by_
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    xi = np.rint(xs)
    return dict(
        file=path,
        n=len(pts),
        title=" ".join(w["t"] for w in ws if w["cy"] > ax_y1),
        axes_frame=[ax_x0, ax_y0, ax_x1, ax_y1],
        x_ticks=xpairs,
        y_ticks=ypairs,
        x_fit_resid=float(xres),
        y_fit_resid=float(yres),
        x_int_max_err=float(np.abs(xs - xi).max()),
        x_unique=int(len(np.unique(xi))),
        x_min=int(xi.min()),
        x_max=int(xi.max()),
        y_min=float(ys.min()),
        y_max=float(ys.max()),
        channel=xi.astype(int).tolist(),
        value=ys.tolist(),
    )


if __name__ == "__main__":
    out = {}
    for p in sys.argv[1:]:
        d = extract(p)
        key = p.split("/")[-1].replace(".pdf", "")
        out[key] = d
        print(
            f"{key:28s} n={d['n']:5d} channels {d['x_min']}..{d['x_max']} unique={d['x_unique']} "
            f"max|x-round(x)|={d['x_int_max_err']:.4f}  y=[{d['y_min']:.3e},{d['y_max']:.3e}]  "
            f"tick-fit resid x={d['x_fit_resid']:.2e} y={d['y_fit_resid']:.2e}  title='{d['title']}'"
        )
    json.dump(out, open("points.json", "w"))
