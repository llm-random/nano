"""Check every number the prose of h_balanced_flops.ipynb quotes against the notebook's own functions.

The notebook's markdown quotes its defaults - GFLOPs at 1M tokens, ratios, crossovers - and nothing
ties those numbers to the code. This script runs the notebook's code cells (figures are built, not
shown), recomputes each quoted number, formats it the way the prose does, and checks that the text is
in its cell. Run it after changing a knob in the config cell or the cost model: every MISMATCH line
names the cell and the text the current model gives.

    conda activate nanollmr          # plotly and pandas, like the notebook itself
    python documents/scripts/h_balanced_flops_quotes.py [path/to/h_balanced_flops.ipynb]

The expected strings mirror the prose, so an edit that rewords a quoted sentence needs its check
here updated in the same commit. Exit status 1 on any mismatch.
"""

import contextlib
import io
import json
import re
import sys
from pathlib import Path

import plotly.graph_objects as go

NOTEBOOK = Path(__file__).with_name("h_balanced_flops.ipynb")
M = 1 << 20  # the "1M" the prose quotes


def run(path):
    """The notebook's namespace after running every code cell, with figures not rendered."""
    go.Figure.show = lambda self, *a, **k: None
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for c in json.load(open(path))["cells"]:
            if c["cell_type"] == "code":
                exec("".join(c["source"]), ns)
    return ns


def prose(path):
    """Each markdown cell's text by cell id, whitespace collapsed so line breaks do not matter."""
    return {
        c["id"]: re.sub(r"\s+", " ", "".join(c["source"]))
        for c in json.load(open(path))["cells"]
        if c["cell_type"] == "markdown"
    }


@contextlib.contextmanager
def knobs(g, **override):
    """Run with AME knobs other than the defaults: the notebook's helpers read the global KW."""
    saved = g["KW"]
    g["KW"] = saved | override
    try:
        yield
    finally:
        g["KW"] = saved


def k(x):
    return f"{round(x / 1000)}k"


def mega(x):
    return f"{x / 1e6:.2f}M"


def checks(g):
    """(cell id, the text that cell should contain at the current defaults)."""
    inf, V, KW, HV = g["inference_flops"], g["VOCAB"], g["KW"], g["H_VALUES"]
    cheaper_past, sparse_pass, ame_pass = (
        g["cheaper_past"],
        g["sparse_pass"],
        g["ame_pass"],
    )
    W, HR, HA, HM = g["WEIGHTS"], g["H_REF"], g["H_AME"], g["H_MID"]
    methods = {m[0]: m for m in g["ATTENTION_METHODS"]}
    out = []

    def need(cid, text):
        out.append((cid, text))

    def dec(H, L, variant="ame", **over):
        return inf(V, H, L, mode="decode", variant=variant, **(g["KW"] | over))

    def gf(f):
        return f.total / 1e9

    def wins(name, h=None, variant="ame", mode="decode"):
        return g["wins_past"](
            methods[name][4 if mode == "decode" else 3], mode, h, variant
        )

    def num(s):
        return int(s.replace(",", ""))

    table_gb = V * V * 2 / 1e9  # one head-layer's V x V table, 2 bytes an entry

    # cell 0: the table
    need(
        "cell-00",
        f"{table_gb:.2f} GB in bf16 and {table_gb * 100**2 / 1000:.1f} TB at `H = 100`",
    )

    # cell 12: one generated token at H = 16
    a, v, a1 = dec(16, M), dec(16, M, "vanilla"), dec(16, M, row_miss=1)
    need(
        "cell-12",
        f"AME step at {gf(a):.2f} GFLOPs at `H = 16`, against the vanilla model's "
        f"{gf(v):.1f}, at 1M tokens",
    )
    need("cell-12", f"raise the step to {gf(a1):.2f}")

    # cell 15: the ratio
    def ratios(**over):
        return [dec(H, M, "vanilla").total / dec(H, M, **over).total for H in HV]

    rt, rc = ratios(), ratios(row_miss=1)
    rtn, rcn = ratios(shift=False), ratios(row_miss=1, shift=False)
    need(
        "cell-15",
        f"**{min(rt):.1f}x** for the largest default model to **{max(rt):.1f}x** for "
        f"the smallest",
    )
    need("cell-15", f"the same spread would be {min(rc):.1f}x to {max(rc):.1f}x")
    need(
        "cell-15", f"{min(rtn):.1f}-{max(rtn):.1f}x and {min(rcn):.1f}-{max(rcn):.1f}x"
    )
    edge = [dec(H, 256, "vanilla").total / dec(H, 256).total for H in HV]
    need("cell-15", f"**{min(edge):.2f}-{max(edge):.2f}**")
    need("cell-15", f"about {k((2048 + V) / 2)} key-units")

    # cell 17: where a decode step is half attention
    ct = [g["crossover_exact"](H, variant="ame", mode="decode") for H in HV]
    with knobs(g, row_miss=1):
        cc = [g["crossover_exact"](H, variant="ame", mode="decode") for H in HV]
    need("cell-17", f"lands at {k(min(ct))}-{mega(max(ct))} tokens")
    need("cell-17", f"would land at {k(min(cc))}-{k(max(cc))} with every row computed")

    # cell 20: the decode breakdown
    need(
        "cell-20",
        f"costs {gf(a):.2f} GFLOPs against {gf(v):.1f}, of which "
        f"{a.selection / a.total:.0%} is the bigram selection and "
        f"{a.attn_scores / a.total:.0%} the top-k value sum",
    )
    need(
        "cell-20",
        f"make it {gf(a1):.2f} GFLOPs, {a1.score_rows / a1.total:.0%} of it the row",
    )

    # cells 22-28: the row sweeps
    need(
        "cell-22",
        f"{table_gb:.2f} GB in bf16 per head-layer and "
        f"{table_gb * 100**2 / 1000:.1f} TB over the 10,000 of `H = 100`",
    )
    s32 = [gf(dec(50, 32768, row_miss=m)) for m in g["SWEEP_ROW_MISS"]]
    need(
        "cell-22",
        f"{min(s32):.1f} to {max(s32):.1f} GFLOPs per token at 32k, against the vanilla "
        f"model's {gf(dec(50, 32768, 'vanilla')):.1f}",
    )
    t50, c50, v50 = (
        gf(dec(50, M)),
        gf(dec(50, M, row_miss=1)),
        gf(dec(50, M, "vanilla")),
    )
    need(
        "cell-22",
        f"{t50:.1f} GFLOPs with the table and {c50:.1f} with every row computed, against "
        f"{v50:.0f} - {v50 / t50:.0f}x and {v50 / c50:.0f}x cheaper",
    )
    for cid, ha, hv in [
        ("cell-24", 50, 40),
        ("cell-26", 100, 80),
        ("cell-28", 252, 200),
    ]:
        pt = cheaper_past(lambda L: dec(ha, L).total < dec(hv, L, "vanilla").total)
        pc = cheaper_past(
            lambda L: dec(ha, L, row_miss=1).total < dec(hv, L, "vanilla").total
        )
        need(cid, f"**{k(pt)} tokens** with the table and {k(pc)}")
        at, ac, av = (
            gf(dec(ha, M)),
            gf(dec(ha, M, row_miss=1)),
            gf(dec(hv, M, "vanilla")),
        )
        fmt = (lambda x: f"{x:.1f}") if ha == 50 else (lambda x: f"{x:,.0f}")
        need(cid, f"{fmt(at)}-{fmt(ac)}")
        need(cid, f"{av:,.0f}")
        lo, hi = av / ac, av / at

        def short(x):
            return f"{x:.0f}" if x >= 10 else f"{x:.1f}"

        need(cid, f"{short(lo)}-{short(hi)}")
    need(
        "cell-24",
        f"{gf(dec(50, 4096)):.1f} against {gf(dec(40, 4096, 'vanilla')):.1f} GFLOPs per "
        f"token at a 4k cache",
    )
    need(
        "cell-28",
        f"{gf(dec(252, 4096)):,.0f}-{gf(dec(252, 4096, row_miss=1)):,.0f} against "
        f"{gf(dec(200, 4096, 'vanilla')):,.0f}",
    )

    # cell 32: AME and AMEm as a share of the vanilla step
    def pct(variant, L):
        return [dec(H, L, variant).total / dec(H, L, "vanilla").total * 100 for H in HV]

    p256, pM, a256 = pct("amem", 256), pct("amem", M), pct("ame", 256)
    half = [
        cheaper_past(
            lambda L, H=H: dec(H, L, "amem").total < 0.5 * dec(H, L, "vanilla").total
        )
        for H in HV
    ]
    need("cell-32", f"{min(p256):.1f}-{max(p256):.1f}% of the vanilla step")
    need(
        "cell-32",
        f"below half past {k(min(half))}-{k(max(half))} tokens, to "
        f"{min(pM):.1f}-{max(pM):.1f}% at 1M",
    )
    need(
        "cell-32", f"where AME is {min(a256):.1f}-{max(a256):.1f}% of the vanilla step"
    )

    def gap(H):
        return (dec(H, 4096, "amem").total - dec(H, 4096).total) / 1e9

    need("cell-32", f"{gap(16):.3f} per token at `H = 16`, {gap(50):.2f} at `H = 50`")

    # cell 34: the three references, and AME against an SSM
    for H, variant in [(HA, "ame"), (HM, "ame"), (HR, "amem")]:
        need("cell-34", g["size_label"](H, variant))
    need("cell-34", f"{table_gb * HR**2 / 1000:.1f} TB at `H = {HR}` in bf16")
    need(
        "cell-34", f"{3 * HR * HR * 1000 / 1e9:.3f} GFLOPs per 1k tokens at `H = {HR}`"
    )
    need(
        "cell-34",
        f"{2 * 64 * HR * 2048 * HR / 1e9:.2f} GFLOPs per token at `H = {HR}` against "
        f"Mamba-2's {10 * 64 * HR * 128 * HR / 1e9:.2f}",
    )

    # cell 36b: the prefill figure
    dense_prefill = sparse_pass(HR, M, 1.0, "prefill")
    need(
        "cell-36b",
        f"({ame_pass(HA, M, 'prefill') / dense_prefill:.2f}x at 1M), at `H = {HM}` too "
        f"({ame_pass(HM, M, 'prefill') / dense_prefill:.2f}x at 1M)",
    )

    # cell 37: where AME still wins, per generated token
    def weight_work(H):
        return (
            inf(V, H, 1, mode="decode", variant="ame", top_k=0, shift=False).total / W
        )

    A, AM, AMID = (
        ame_pass(HA, M, "decode"),
        ame_pass(HR, M, "decode", "amem"),
        ame_pass(HM, M, "decode"),
    )
    b_star = 480 * HR + V / (2 * HR)
    need("cell-37", f"AME at `H = {HA}` does {weight_work(HA):.2f}x the weight work")
    need(
        "cell-37",
        f"**{(A - W) / (256 * HR**2):,.0f} keys**, against `B* = 480 H + V/(2H) = {b_star:,.0f}`",
    )
    assert wins("MInference") == wins("dense") == wins("MoBA")
    need("cell-37", f"dense attention past {k(num(wins('dense')))} tokens")
    need(
        "cell-37",
        f"Gemma 2's 1:1 schedule past {k(num(wins('Gemma 2 1:1')))}, the 3:1 linear "
        f"hybrids past {k(num(wins('linear 3:1')))}, Gemma 3's 5:1 past "
        f"{k(num(wins('Gemma 3 5:1')))} and the 7:1 hybrids past {k(num(wins('linear 7:1')))}",
    )
    need("cell-37", f"{M / 16 / 1000:.1f}k and {0.059 * M / 1000:.1f}k at 1M")
    nsa, dsa = (sparse_pass(HR, M, methods[n][4](M), "decode") for n in ("NSA", "DSA"))
    need(
        "cell-37",
        f"above AME's {(A - W) / (256 * HR**2) / 1000:.1f}k at 1M, where they cost "
        f"{nsa / A:.2f}x and {dsa / A:.2f}x as much as AME, but AME overtakes them only past "
        f"{k(num(wins('NSA')))} and {k(num(wins('DSA')))}",
    )
    assert wins("local only").startswith(">")
    need(
        "cell-37",
        f"{sparse_pass(HR, M, methods['local only'][4](M), 'decode') / 1e9:.0f} GFLOPs per "
        f"token against AME's {A / 1e9:.0f} at 1M",
    )
    assert all(wins(n, mode="prefill").startswith(">") for n in methods)
    with knobs(g, row_miss=1):
        A1 = ame_pass(HA, M, "decode")
        need(
            "cell-37",
            f"AME costs {A1 / 1e9:.0f} GFLOPs per token at 1M, `B` rises to "
            f"{(A1 - W) / (256 * HR**2):,.0f} keys, and NSA and DSA stay cheaper until "
            f"{mega(num(wins('NSA')))} and {mega(num(wins('DSA')))} (at 1M AME costs "
            f"{A1 / nsa:.2f}x and {A1 / dsa:.2f}x as much)",
        )
        mid_rows = [k(num(wins(n, HM))) for n in ("NSA", "DSA")]
    with knobs(g, row_miss=0.25):
        need(
            "cell-37",
            f"crossovers are {k(num(wins('NSA')))} and {mega(num(wins('DSA')))}",
        )
    # ... the AME between the two sizes
    b_mid = (AMID - W) / (256 * HR**2)
    assert b_mid < b_star
    assert wins("MInference", HM) == wins("dense", HM) == wins("MoBA", HM)
    assert wins("local only", HM).startswith(">")
    assert all(wins(n, HM, mode="prefill").startswith(">") for n in methods)
    need(
        "cell-37",
        f"AME at `H = {HM}`. It does {weight_work(HM):.2f}x the reference's weight work "
        f"rather than {weight_work(HA):.2f}x",
    )
    need("cell-37", f"**{b_mid:,.0f} keys** per query, fewer than `B*`")
    need(
        "cell-37",
        f"It beats dense attention, MInference and MoBA past {k(num(wins('dense', HM)))} "
        f"tokens, Gemma 2's 1:1 past {k(num(wins('Gemma 2 1:1', HM)))}, the 3:1 hybrids past "
        f"{k(num(wins('linear 3:1', HM)))}, Gemma 3's 5:1 past {k(num(wins('Gemma 3 5:1', HM)))} "
        f"and the 7:1 hybrids past {k(num(wins('linear 7:1', HM)))}, and NSA and DSA past "
        f"{k(num(wins('NSA', HM)))} and {k(num(wins('DSA', HM)))} ({mid_rows[0]} and "
        f"{mid_rows[1]} with every row computed)",
    )
    need(
        "cell-37",
        f"at 1M it is {nsa / AMID:.2f}x and {dsa / AMID:.2f}x cheaper per token than NSA and DSA",
    )
    # ... AMEm, at the reference's weights
    need("cell-37", f"{W / 1e9:.1f} GFLOPs per token")
    dense = sparse_pass(HR, M, 1.0, "decode")
    need(
        "cell-37",
        f"costs {(AM - W) / 1e9:.1f} GFLOPs on top of those weights ({AM / W:.2f}x in all) "
        f"and dense attention's {(dense - W) / 1e9:,.0f} ({dense / W:.1f}x)",
    )
    every = [n for n in methods if wins(n, HR, "amem") == "1"]
    assert sorted(every) == sorted(
        set(methods) - {"linear 3:1", "linear 7:1", "local only"}
    ), every
    assert wins("local only", HR, "amem").startswith(">")
    need(
        "cell-37",
        f"the 3:1 hybrids past {k(num(wins('linear 3:1', HR, 'amem')))}, the 7:1 hybrids "
        f"past {k(num(wins('linear 7:1', HR, 'amem')))}",
    )
    need(
        "cell-37",
        f"{nsa / AM:.2f}x and {dsa / AM:.2f}x cheaper per token than NSA and DSA.",
    )
    f = inf(V, HA, M, mode="decode", variant="ame", **KW)
    assert f.score_rows == 0  # the prose assumes the table
    weights = f.total - f.attn_scores - f.score_rows - f.selection
    need(
        "cell-37",
        f"Of AME's own {A / W:.2f}x over those weights at `H = {HA}`, {weights / W:.2f}x is "
        f"its bigger weights, {f.selection / W:.2f}x the bigram selection and "
        f"{f.attn_scores / W:.2f}x the value sum",
    )

    # cell 39: the caveat
    banks_gb = g["weight_params"](V, 100, variant="ame")["qk_banks"] * 2 / 1e9
    need(
        "cell-39",
        f"{table_gb * 100**2 / 1000:.1f} TB in bf16 at `H = 100`, against {banks_gb:.0f} GB "
        f"for the banks",
    )
    return out


def main(path=NOTEBOOK):
    g = run(path)
    text = prose(path)
    expected = checks(g)
    bad = [(cid, s) for cid, s in expected if s not in text[cid]]
    print(f"{len(expected) - len(bad)}/{len(expected)} quoted values match")
    for cid, s in bad:
        print(f"MISMATCH {cid}: expected {s!r}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
