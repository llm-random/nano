"""Compare regenerated charts with reference PDFs: pixel diff at 150 dpi + exact vector marker positions."""

import re, subprocess, sys, zlib
import numpy as np
from PIL import Image


def render(pdf, out):
    subprocess.run(
        ["pdftoppm", "-png", "-r", "150", "-singlefile", pdf, out], check=True
    )
    return np.asarray(Image.open(out + ".png").convert("RGB")).astype(int)


def markers(pdf):
    raw = open(pdf, "rb").read()
    best = ""
    for m in re.finditer(rb"stream\r?\n", raw):
        e = raw.find(b"endstream", m.end())
        try:
            s = zlib.decompress(raw[m.end() : e]).decode("latin-1")
        except Exception:
            continue
        if len(s) > len(best):
            best = s
    s = re.sub(r"\s+", " ", best)
    return sorted(
        (float(tx), float(ty))
        for tx, ty in re.findall(
            r"q 3\.125 0 0 3\.125 (-?[\d.]+) (-?[\d.]+) cm [^Q]*?-?[\d.]+ -?[\d.]+ 3 3 re f",
            s,
        )
    )


def compare(old_pdf, new_pdf, tag):
    a, b = render(old_pdf, f"cmp/old_{tag}"), render(new_pdf, f"cmp/new_{tag}")
    if a.shape != b.shape:
        print(f"{tag}: SIZE MISMATCH {a.shape} vs {b.shape}")
        return
    d = np.abs(a - b).max(axis=2)
    Image.fromarray((255 - np.clip(d * 3, 0, 255)).astype(np.uint8)).save(
        f"cmp/diff_{tag}.png"
    )
    print(
        f"{tag}: {a.shape[1]}x{a.shape[0]}px  identical {np.mean(d == 0)*100:.3f}%  "
        f"differing(>8/255) {np.mean(d > 8)*100:.4f}%  max diff {d.max()}"
    )
    mo, mn = markers(old_pdf), markers(new_pdf)
    if mo or mn:
        if len(mo) == len(mn):
            dd = np.abs(np.array(mo) - np.array(mn)).max()
            print(
                f"   vector markers: {len(mo)} vs {len(mn)}, max position diff {dd:.4f} (1 unit = 0.24 pt)"
            )
        else:
            print(f"   vector markers COUNT MISMATCH {len(mo)} vs {len(mn)}")


if __name__ == "__main__":
    for old, new, tag in zip(sys.argv[1::3], sys.argv[2::3], sys.argv[3::3]):
        compare(old, new, tag)
