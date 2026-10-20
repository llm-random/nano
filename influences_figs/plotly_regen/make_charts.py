"""Regenerate the channel-influence charts (paper Figure 6) in the original Plotly style.

Style recovered from the existing paper PDFs (Chromium/Skia m88 = kaleido 0.2.x export):
  - go.Figure default 'plotly' template, 1000x400 px canvas (750x300 pt), default margins
  - font DejaVu Sans; originals: ticks 14 / axis titles 16 (FONT_SIZE/TITLE_SIZE, now 28/32 = 2x);
    DejaVu is what the original export fell back to
  - square markers size 3, colour #636efa; WebGL (Scattergl) when > 1000 points, like px 'auto'
  - x title "Channel index", y title "<Scattering|Gathering> influence" (title font 16), y tickformat ".1e"
  - page cropped to [4 26 694 226] pt at original sizes, else tight to content (ghostscript bbox)
Data: points.json from extract_points.py (exact values recovered from the matplotlib PDFs).
"""
import json
import re
import subprocess
import sys

import plotly.graph_objects as go
import plotly.io as pio
from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject

CROP = [4, 26, 694, 226]   # original paper crop (fits FONT_SIZE=14 / TITLE_SIZE=16)

# text sizes: paper originals used 14 (ticks) / 16 (axis titles); 2026-10-07 enlarged for readability: 18/20 = largest size where the one-line y title fits the plot height (21/24 and 28/32 needed 2 lines)
FONT_SIZE = 18
TITLE_SIZE = 20
PAD_PT = 2  # padding around the content bbox when auto-cropping
# canvas for enlarged text: trim the empty 100px top margin and lower the height so the cropped
# chart keeps the paper's original aspect (~690x200 pt = 3.45:1); y title on two lines to fit.
BIG_HEIGHT = 297
BIG_MARGIN_T = 10
YTITLE_TWO_LINES = False  # 1.5x text needed 2 lines; smaller text fits one line (preferred)

# which matplotlib source feeds which paper chart
CHARTS = {
    "attn_scatering_influence": ("attn_llama_weight_channels", "Scattering influence"),
    "attn_gathering_influence": ("attn_residual_channels", "Gathering influence"),
    "ff_scatering_influence": ("ff_llama_weight_channels", "Scattering influence"),
    "ff_gathering_influence": ("ff_residual_channels", "Gathering influence"),
}


def content_bbox(pdf, dpi=144):
    """Bbox (pt, PDF coords) of non-white content on page 1, measured on a render.
    (ghostscript's bbox device can't be used: Scattergl raster layers span the whole canvas.)"""
    import numpy as np
    from PIL import Image
    png = pdf + ".bbox"
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile", pdf, png], check=True)
    im = np.asarray(Image.open(png + ".png").convert("L"))
    ys, xs = np.nonzero(im < 250)
    H = im.shape[0]
    k = 72.0 / dpi
    return [xs.min() * k - PAD_PT, (H - ys.max() - 1) * k - PAD_PT, (xs.max() + 1) * k + PAD_PT, (H - ys.min()) * k + PAD_PT]


def apply_crop(raw, out_pdf, crop):
    r = PdfReader(raw)
    w = PdfWriter()
    for p in r.pages:
        p.mediabox = RectangleObject(crop)
        p.cropbox = RectangleObject(crop)
        w.add_page(p)
    w.add_metadata(r.metadata or {})
    with open(out_pdf, "wb") as fh:
        w.write(fh)


def make(points, src, ytitle, out_pdf):
    d = points[src]
    trace_cls = go.Scattergl if len(d["channel"]) > 1000 else go.Scatter
    fig = go.Figure(trace_cls(x=d["channel"], y=d["value"], mode="markers",
                              marker=dict(symbol="square", size=3)))
    # original export fell back to DejaVu Sans (Open Sans absent); pin it so margins/labels match
    original = (FONT_SIZE, TITLE_SIZE) == (14, 16)
    height = 400 if original else BIG_HEIGHT
    fig.update_layout(width=1000, height=height, font_family="DejaVu Sans", font_size=FONT_SIZE)
    if not original:
        fig.update_layout(margin_t=BIG_MARGIN_T)
        if YTITLE_TWO_LINES:
            ytitle = ytitle.replace(" ", "<br>")
    fig.update_xaxes(title_text="Channel index", title_font_size=TITLE_SIZE)
    fig.update_yaxes(title_text=ytitle, tickformat=".1e", title_font_size=TITLE_SIZE)
    raw = out_pdf + ".uncropped.pdf"
    fig.write_image(raw, format="pdf", width=1000, height=height, scale=1, engine="kaleido")
    return raw


if __name__ == "__main__":
    # kaleido 0.2.1 bug: first export can bake in a "Loading [MathJax]..." box; we use no LaTeX, so disable it
    pio.kaleido.scope.mathjax = None
    outdir = sys.argv[1]
    which = sys.argv[2:] or list(CHARTS)
    pts = json.load(open("points.json"))
    raws = {}
    for name in which:
        src, ytitle = CHARTS[name]
        raws[name] = make(pts, src, ytitle, f"{outdir}/{name}.pdf")
    if (FONT_SIZE, TITLE_SIZE) == (14, 16):
        crop = CROP
    else:
        # ONE shared crop (union of content boxes) so all panels scale identically in LaTeX
        boxes = [content_bbox(r) for r in raws.values()]
        crop = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
    for name, raw in raws.items():
        apply_crop(raw, f"{outdir}/{name}.pdf", crop)
        print("wrote", f"{outdir}/{name}.pdf", "<-", CHARTS[name][0], "crop", [round(float(c), 2) for c in crop])
