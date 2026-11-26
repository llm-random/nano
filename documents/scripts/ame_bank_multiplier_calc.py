"""Would Gemma's output multiplier help AME's memory banks? Numbers, not argument.

Gemma 3n reads its per-layer embedding table through a constant, y = 16 * W[t] with
W ~ N(0, 0.01^2). The question (2026-11-10): does that help the three things we want
from AME's banks?

  T1  a rarely updated row has little influence (hence a small init);
  T2  a sparsely updated row does not deteriorate between its updates (hence no
      weight decay);
  T3  rows take small steps, so they stay near zero as long as possible.

Everything is AME as Stage 2 trained it (configs/bank_gates/base.yaml and the llama
trainer defaults): h12 (d_head 64, 12 heads, 12 layers), 1,048,576 tokens per step,
3,815 steps, LR 2^-8 with 38 warmup steps and cosine to 10%, and the banks' own AdamW
at betas (0.9, 0.999), eps 1e-8, no weight decay. Bank variants:

  hard    F.normalize on each head's row, as src/core/ame.py does today;
  soft    Plan A's soft norm, x / sqrt(||x||^2 + 1)
          (documents/notes/pkmf_init_comparison.md, "Plan A");
  xM      Gemma's multiplier on the gathered row, before the norm.

Sections:
  1  the real AMEAttention: what a multiplier changes in the output, the gradient
     and twenty AdamW steps;
  2  T1: how strongly a never-updated row acts;
  3  the gradient a row receives per unit gradient on its normalized vector;
  4  T3: how far AdamW moves an element for one occurrence of its token;
  5  where AME's real per-occurrence gradients sit against Adam's eps, calibrated
     from Stage 2's logged grad_norm/banks;
  6  T1 + T3 over a whole run: rows from each frequency bucket, 3,815 AdamW steps;
  7  T2: weight decay.

Run from the repo root (CPU, about 4 minutes on 8 threads):
    pixi run python documents/scripts/ame_bank_multiplier_calc.py
Options: --mult (default 16, Gemma's), --rho (consistent share of each occurrence's
gradient, default 0), --counts (a token_counts.pt to rebuild the buckets from).
"""

import argparse
import contextlib
import math
import os
import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import src.core.ame as ame  # noqa: E402
from src.core.schedulers import get_cosine_scheduler_with_warmup  # noqa: E402

# ---- Stage 2's h12 recipe ----------------------------------------------------------
D_HEAD, HEADS, LAYERS = 64, 12, 12
DMODEL = D_HEAD * HEADS  # 768
TOKENS_PER_STEP = 512 * 2048  # batch 512 x (2049 - 1) predicted positions
STEPS, WARMUP, PEAK_LR, FINAL_LR_FRACTION = 3815, 38, 2.0**-8, 0.1
# torch.optim.AdamW's defaults: neither main.py nor memory_parallel.py sets them.
BETAS, ADAM_EPS = (0.9, 0.999), 1e-8
TAU = math.sqrt(D_HEAD)  # tau_init = 8; none_std001 ends at 8.04 (layer mean)
SOFT_EPS = 1.0  # Plan A

# Frequency buckets from the h12 runs' token_counts.pt (sha256 ab1df510..., the same
# in all seven Stage 1-2 cells): u = steps a token occurred in, c = its occurrences
# per such step, for the bucket's median token. --counts rebuilds them.
BUCKETS = [  # (label, u, c)
    ("u=1", 1, 1.0),
    ("u=3", 3, 1.1),  # median of u = 1-10
    ("u=48", 48, 1.43),  # 11-100
    ("u=475", 475, 1.54),  # 101-1000
    ("u=3296", 3296, 3.28),  # 1001-3814
    ("u=3815", 3815, 27.1),  # every step
]

# grad_norm/banks, the L2 norm over all 24 bank gradients (12 layers x Q and K),
# median per window (ame_bank_gates_stage2_report.md, "b1_std1's banks trained after
# a slow start"). b3_std1 is the sigma = 1 run used for calibration: its rows stay
# about 8 long per head, so the norm divides their gradient by a known length.
WINDOWS = [(0, 99), (100, 499), (500, 999), (1000, 1999), (2000, 3433), (3434, 3814)]
GRAD_NORM_SIGMA1 = [3.4e-4, 1.3e-3, 1.4e-3, 1.3e-3, 1.1e-3, 9.8e-4]  # b3_std1
GRAD_NORM_SIGMA001 = [0.042, 0.037, 0.018, 0.012, 0.0083, 0.0077]  # none_std001
# ||E + base|| per head in b3_std1 (B3 normalizes (E + base)/sqrt 2, which divides
# E's gradient by ||E + base||): the row is 8 long, the base grows 0 -> 4.8 (q),
# 0 -> 2.2 (k), so 8.0-9.3 over the run. One value; +-8%.
SIGMA1_ROW_LENGTH = 8.5


def bank_norm(x, kind):
    if kind == "hard":
        return F.normalize(x, dim=-1)
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + SOFT_EPS**2)


def lr_schedule():
    """The LR of every step, from the trainer's own scheduler."""
    # SequentialLR passes an epoch to its children internally; torch warns about it.
    warnings.filterwarnings(
        "ignore", message="The epoch parameter", category=UserWarning
    )
    dummy = nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([dummy], lr=PEAK_LR)
    sched = get_cosine_scheduler_with_warmup(opt, WARMUP, STEPS, FINAL_LR_FRACTION)
    lrs = []
    for _ in range(STEPS):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return lrs


def header(text):
    print(f"\n{'=' * 100}\n{text}\n{'=' * 100}")


def table(rows, columns, cells, row_width=30, cell_width=11):
    print(f"{'':<{row_width}}" + "".join(f"{c:>{cell_width}}" for c in columns))
    for name, values in zip(rows, cells):
        print(f"{name:<{row_width}}" + "".join(f"{v:>{cell_width}}" for v in values))


def fmt(x, digits=2):
    if x == 0:
        return "0"
    if abs(x) >= 1e4 or abs(x) < 1e-2:
        return f"{x:.{digits - 1}e}"
    return f"{x:.{digits}f}" if abs(x) < 10 else f"{x:.1f}"


# ---- 1. The real module ------------------------------------------------------------


class Scaled(nn.Module):
    """Gemma's multiplier on a lookup: returns m * W[t]; W stays the parameter."""

    def __init__(self, embedding, m):
        super().__init__()
        self.embedding, self.m = embedding, m

    @property
    def weight(self):
        return self.embedding.weight

    def forward(self, ids):
        return self.embedding(ids) * self.m


class _Functional:
    """Stands in for torch.nn.functional inside src/core/ame.py: `normalize` is
    swapped, everything else (pad, silu, ...) passes through."""

    def __init__(self, normalize):
        self._normalize = normalize

    def __getattr__(self, name):
        return getattr(F, name)

    def normalize(self, x, dim=-1, **_):
        return self._normalize(x, dim)


@contextlib.contextmanager
def norm_in_module(kind, post=1.0):
    """Run src/core/ame.py with the soft norm, or a constant after the hard norm."""
    if kind == "hard" and post == 1.0:
        yield
        return

    def normalize(x, dim):
        if kind == "hard":
            return post * F.normalize(x, dim=dim)
        return x / torch.sqrt((x * x).sum(dim, keepdim=True) + SOFT_EPS**2)

    saved = ame.F
    ame.F = _Functional(normalize)
    try:
        yield
    finally:
        ame.F = saved


def run_layer(sigma, pre=1.0, kind="hard", post=1.0, tau_mult=1.0, bank_lr_mult=1.0):
    """One AMEAttention at h12 widths (vocab cut to 1,024 for speed), trained for
    20 AdamW steps at the peak LR on random inputs and targets. Returns the first
    output, the first q-bank gradient, and the q-bank rows after the steps."""
    torch.manual_seed(0)
    vocab, seq, batch = 1024, 64, 2
    layer = ame.AMEAttention(
        DMODEL,
        HEADS,
        HEADS,
        2048,
        vocab_size=vocab,
        bank_init_std=sigma,
        backend="reference",
    )
    with torch.no_grad():
        layer.tau.mul_(tau_mult)
    if pre != 1.0:
        layer.q_bank = Scaled(layer.q_bank, pre)
        layer.k_bank = Scaled(layer.k_bank, pre)
    banks = [layer.q_bank.weight, layer.k_bank.weight]
    rest = [p for p in layer.parameters() if all(p is not b for b in banks)]
    opt = torch.optim.AdamW(
        [
            {"params": banks, "lr": PEAK_LR * bank_lr_mult},
            {"params": rest, "lr": PEAK_LR},
        ],
        betas=BETAS,
        eps=ADAM_EPS,
        weight_decay=0.0,
    )
    gen = torch.Generator().manual_seed(1)
    with norm_in_module(kind, post):
        for step in range(20):
            ids = torch.randint(0, vocab, (batch, seq), generator=gen)
            x = torch.randn(batch, seq, DMODEL, generator=gen)
            target = torch.randn(batch, seq, DMODEL, generator=gen)
            out = layer(x, ids)
            opt.zero_grad()
            (out * target).sum().backward()
            if step == 0:
                first_out = out.detach().clone()
                first_grad = layer.q_bank.weight.grad.detach().clone()
            opt.step()
    return first_out, first_grad, layer.q_bank.weight.detach().clone()


def section_module(mult):
    header(
        "1. The real AMEAttention (src/core/ame.py), h12 widths: what a multiplier changes\n"
        "   ||a - b|| / ||b|| for the layer output at step 0, the q-bank gradient at step 0,\n"
        "   and the q-bank rows after 20 AdamW steps (peak LR, betas (0.9, 0.999), eps 1e-8, wd 0)"
    )
    base = run_layer(0.01)
    odd = math.sqrt(DMODEL)

    def rel(a, b):
        return ((a - b).norm() / b.norm()).item()

    rows, cells = [], []

    def add(name, a, b, scale_g=1.0, scale_w=1.0):
        rows.append(name)
        cells.append(
            [
                fmt(rel(a[0], b[0])),
                fmt(rel(a[1] * scale_g, b[1])),
                fmt(rel(a[2] * scale_w, b[2])),
            ]
        )

    add(f"hard, x{mult:g} before the norm", run_layer(0.01, pre=mult), base)
    add(f"hard, x{odd:.1f} before the norm", run_layer(0.01, pre=odd), base)
    # A constant after the norm scales every score by post^2: the same forward as tau * post^2.
    post = math.sqrt(mult)
    add(
        f"hard, x{post:g} after it, vs tau x{mult:g}",
        run_layer(0.01, post=post),
        run_layer(0.01, tau_mult=mult),
    )
    # Soft norm: x m on a 0.01 table is a 0.01 m table trained at m x the bank LR
    # (W' = m W; the gradient on W is m times the gradient on W').
    add(
        f"soft, x{mult:g}, vs s={0.01 * mult:g} at {mult:g}x bank LR",
        run_layer(0.01, pre=mult, kind="soft"),
        run_layer(0.01 * mult, kind="soft", bank_lr_mult=mult),
        scale_g=1.0 / mult,
        scale_w=mult,
    )
    add(
        f"soft, x{mult:g}, vs soft without it",
        run_layer(0.01, pre=mult, kind="soft"),
        run_layer(0.01, kind="soft"),
    )
    table(
        rows, ["output", "bank grad", "rows, 20 st"], cells, row_width=40, cell_width=13
    )
    print(
        "   0 = bit-identical. Rounding-level differences (row 2) grow over the 20 steps: AdamW turns any\n"
        "   near-zero gradient element into a full +-LR step whose sign the rounding decides.\n"
        "   Row 3: the same forward and bank gradient; the runs part later because tau's own gradient differs.\n"
        f"   Row 4 compares W' with {mult:g} W, and its gradient with the x{mult:g} run's divided by {mult:g}."
    )


# ---- 2-3. Influence of an idle row, and the gradient a row receives ---------------


def section_idle_and_gradient(variants):
    header(
        "2. T1: how strongly a never-updated row acts (per head; tau = 8 over 64 dims)\n"
        "   strength = ||normalized row||, 1 = a fully trained row. logit std = std of tau * <a, b>,\n"
        "   the score noise the row injects; for keys multiply by |alpha_1|.\n"
        "3. The gradient it receives: ||dL/dW|| per head for a unit gradient on its normalized row"
    )
    gen = torch.Generator().manual_seed(2)
    n = 200_000
    # One set of draws for every variant, so variants that are the same print the same.
    trained = F.normalize(torch.randn(n, D_HEAD, generator=gen), dim=-1)
    z1 = torch.randn(n, D_HEAD, generator=gen)
    z2 = torch.randn(n, D_HEAD, generator=gen)
    g = F.normalize(torch.randn(20000, D_HEAD, generator=gen), dim=-1)
    rows, cells = [], []
    for name, kind, sigma, mult, _ in variants:
        a = bank_norm(mult * sigma * z1, kind)
        b = bank_norm(mult * sigma * z2, kind)
        w = (sigma * z1[:20000]).clone().requires_grad_()
        (grad,) = torch.autograd.grad((g * bank_norm(mult * w, kind)).sum(), w)
        rows.append(name)
        cells.append(
            [
                fmt((mult * sigma * z1).norm(dim=-1).mean().item()),
                fmt(a.norm(dim=-1).mean().item()),
                fmt((TAU * (a * trained).sum(-1)).std().item()),
                fmt((TAU * (a * b).sum(-1)).std().item()),
                fmt(grad.norm(dim=-1).mean().item()),
            ]
        )
    table(
        rows,
        ["row length", "strength", "vs trained", "idle x idle", "grad on W"],
        cells,
        row_width=30,
        cell_width=12,
    )
    print(
        "   row length = ||m * W|| per head at init; logit std columns are against a trained row / another idle row."
    )


# ---- 4-5. AdamW's step for one occurrence, and where real gradients sit ------------


def adam_travel(grads, events, lrs, eps=ADAM_EPS):
    """Element i gets one gradient, grads[i] at step events[i], and zero on every
    other step -- a bank row whose token occurs once -- and is stepped every step,
    as the real bank is. Returns its total travel and the part on the event step."""
    grads = torch.as_tensor(grads, dtype=torch.float32)
    events = torch.as_tensor(events)
    p = torch.zeros_like(grads, requires_grad=True)
    opt = torch.optim.AdamW([p], lr=lrs[0], betas=BETAS, eps=eps, weight_decay=0.0)
    on_event = torch.zeros_like(grads)
    zero = torch.zeros_like(grads)
    for t in range(STEPS):
        hit = events == t
        p.grad = torch.where(hit, grads, zero)
        opt.param_groups[0]["lr"] = lrs[t]
        before = p.detach().clone()
        opt.step()
        on_event += torch.where(hit, (p.detach() - before).abs(), zero)
    # Momentum keeps the sign of -g, so |p| is the whole path.
    return p.detach().abs(), on_event


def section_one_occurrence(lrs):
    header(
        "4. T3: how far AdamW moves one element for ONE occurrence of its token\n"
        "   total travel over the rest of the run, in units of the LR at the occurrence step;\n"
        "   g = that element's gradient. Without eps the answer would not depend on g at all."
    )
    gs = [1e-5, 1e-6, 3.16e-7, 1e-7, 3.16e-8, 1e-8, 3.16e-9]
    t0s = [50, 500, 2000, 3500]
    grads = [g for g in gs for _ in t0s]
    events = [t for _ in gs for t in t0s]
    total, on_event = adam_travel(grads, events, lrs)
    ref = adam_travel([1e-5], [2000], lrs, eps=1e-30)[0]
    cells = []
    for i, g in enumerate(gs):
        row = []
        for j, t in enumerate(t0s):
            k = i * len(t0s) + j
            row.append(fmt(total[k].item() / lrs[t]))
        k = i * len(t0s) + 2
        row.append(fmt(1 - on_event[k].item() / total[k].item()))
        cells.append(row)
    table(
        [f"g = {g:.2g}" for g in gs],
        [f"t={t}" for t in t0s] + ["after"],
        cells,
        row_width=14,
        cell_width=10,
    )
    print(
        f"   eps -> 0 at t=2000: {ref[0].item() / lrs[2000]:.1f} x LR, whatever g is."
        f"   Half that at g = eps / sqrt(1 - beta2) = {ADAM_EPS / math.sqrt(1 - BETAS[1]):.2g}.\n"
        "   'after' (t=2000) = share of the travel taken on later steps, where the token is absent (momentum).\n"
        "   t=50 is low because Adam's bias correction still inflates v early on."
    )


def gamma_n(grad_norm_banks):
    """Per-occurrence gradient on ONE head's normalized row, from a sigma = 1 run's
    grad_norm/banks, assuming a step's occurrences add incoherently: then
    grad_norm^2 = 24 banks x 12 heads x TOKENS x (gamma / row length)^2."""
    per_head_on_w = grad_norm_banks / math.sqrt(2 * LAYERS * HEADS * TOKENS_PER_STEP)
    return per_head_on_w * SIGMA1_ROW_LENGTH


def section_calibration(lrs, variants, counts_summary):
    header(
        "5. Where AME's real per-occurrence gradients sit against Adam's eps\n"
        "   gamma = gradient on one head's normalized row per occurrence, from b3_std1's grad_norm/banks.\n"
        "   Per variant: per-element gradient on W for one occurrence at the INIT row length, and the\n"
        "   travel AdamW gives it (x LR at the window's middle step). Undamped travel ~ 32 x LR late."
    )
    gen = torch.Generator().manual_seed(3)
    factors = []  # ||dL/dW|| per head per unit gradient on n, at init
    for _, kind, sigma, mult, _ in variants:
        w = (sigma * torch.randn(20000, D_HEAD, generator=gen)).requires_grad_()
        g = F.normalize(torch.randn(20000, D_HEAD, generator=gen), dim=-1)
        (grad,) = torch.autograd.grad((g * bank_norm(mult * w, kind)).sum(), w)
        factors.append(grad.norm(dim=-1).mean().item())
    mids = [(a + b) // 2 for a, b in WINDOWS]
    gammas = [gamma_n(G) for G in GRAD_NORM_SIGMA1]
    grads, events = [], []
    for gam, t in zip(gammas, mids):
        for f in factors:
            grads.append(gam * f / math.sqrt(D_HEAD))
            events.append(t)
    travel, _ = adam_travel(grads, events, lrs)
    pick = [i for i, v in enumerate(variants) if v[4] == 1.0]
    columns = [f"{a}-{b}" for a, b in WINDOWS]
    nw, nv = len(WINDOWS), len(variants)
    print("   steps:")
    table(
        ["grad_norm/banks, s=1 run", "gamma per occurrence"],
        columns,
        [[fmt(G) for G in GRAD_NORM_SIGMA1], [fmt(g) for g in gammas]],
        row_width=30,
        cell_width=11,
    )
    print("\n   per-element gradient on W, one occurrence:")
    table(
        [variants[i][0] for i in pick],
        columns,
        [[fmt(grads[w * nv + i]) for w in range(nw)] for i in pick],
        row_width=30,
        cell_width=11,
    )
    print(
        "\n   AdamW travel for it, x LR (eps -> 0: ~29 from step 500 on; half of that at 3.2e-7):"
    )
    table(
        [variants[i][0] for i in pick],
        columns,
        [
            [fmt(travel[w * nv + i].item() / lrs[mids[w]]) for w in range(nw)]
            for i in pick
        ],
        row_width=30,
        cell_width=11,
    )
    # Cross-check: the same gamma predicts none_std001's bank norm if its frequent rows
    # (88% of occurrences) are L long per head; L = gamma * sqrt(24 * 12 * TOKENS) / G.
    implied = [
        g * math.sqrt(2 * LAYERS * HEADS * TOKENS_PER_STEP) / G
        for g, G in zip(gammas, GRAD_NORM_SIGMA001)
    ]
    print(
        "   Cross-check: none_std001's logged grad_norm/banks implies its frequent rows are "
        + ", ".join(fmt(x) for x in implied)
        + " long per head\n   over the six windows (init 0.08). Stage 1 measured ~1.0 at u = 3815 at the end"
        " (ame_bank_gates_report.md, B3 at sigma 0.01)."
    )
    if counts_summary:
        print(
            f"   Coherent bound: if occurrences added in phase, gamma would be {counts_summary:.0f}x smaller."
        )


# ---- 6. A whole run, per frequency bucket ------------------------------------------


def section_simulation(lrs, variants, buckets, rho, rows_per_bucket):
    header(
        "6. T1 + T3 over a whole run: one head's row, 3,815 AdamW steps at the real schedule\n"
        f"   {rows_per_bucket} rows per bucket; each occurrence brings gamma (section 5) on the normalized row,\n"
        f"   random in direction (rho = {rho:g} consistent). The same draws for every variant."
    )
    gen = torch.Generator().manual_seed(4)
    n_b, r = len(buckets), rows_per_bucket
    n = n_b * r
    counts = torch.zeros(n, STEPS)
    for b, (_, u, c) in enumerate(buckets):
        for i in range(r):
            steps = torch.randperm(STEPS, generator=gen)[:u]
            extra = torch.poisson(torch.full((u,), c - 1.0), generator=gen)
            counts[b * r + i, steps] = 1.0 + extra
    z0 = torch.randn(n, D_HEAD, generator=gen)
    signal = F.normalize(torch.randn(n, D_HEAD, generator=gen), dim=-1)
    gam = torch.zeros(STEPS)
    for (a, b_), G in zip(WINDOWS, GRAD_NORM_SIGMA1):
        gam[a : b_ + 1] = gamma_n(G)

    runs = []
    for name, kind, sigma, mult, lr_mult in variants:
        for eps in (ADAM_EPS, 1e-30):
            w = (sigma * z0).clone().requires_grad_()
            opt = torch.optim.AdamW(
                [w], lr=lrs[0], betas=BETAS, eps=eps, weight_decay=0.0
            )
            runs.append(
                dict(
                    name=name,
                    kind=kind,
                    mult=mult,
                    lr_mult=lr_mult,
                    eps=eps,
                    w=w,
                    w0=w.detach().clone(),
                    opt=opt,
                    seen_path=torch.zeros(n),
                    idle_path=torch.zeros(n),
                )
            )
    for t in range(STEPS):
        xi = torch.randn(n, D_HEAD, generator=gen) / math.sqrt(D_HEAD)
        c_t = counts[:, t : t + 1]
        g_n = gam[t] * (math.sqrt(1 - rho**2) * c_t.sqrt() * xi + rho * c_t * signal)
        seen = c_t[:, 0] > 0
        for run in runs:
            w = run["w"]
            (grad,) = torch.autograd.grad(
                (g_n * bank_norm(run["mult"] * w, run["kind"])).sum(), w
            )
            w.grad = grad
            run["opt"].param_groups[0]["lr"] = lrs[t] * run["lr_mult"]
            before = w.detach().clone()
            run["opt"].step()
            step = (w.detach() - before).norm(dim=-1)
            run["seen_path"] += torch.where(seen, step, torch.zeros_like(step))
            run["idle_path"] += torch.where(seen, torch.zeros_like(step), step)

    def per_bucket(values):
        return [values[b * r : (b + 1) * r].mean().item() for b in range(n_b)]

    stats = {}
    for i in range(0, len(runs), 2):
        run, twin = runs[i], runs[i + 1]
        m = run["mult"]
        w, w0 = run["w"].detach(), run["w0"]
        moved = (m * (w - w0)).norm(dim=-1)
        moved_twin = (m * (twin["w"].detach() - w0)).norm(dim=-1)
        cos = F.cosine_similarity(w, w0, dim=-1).clamp(-1, 1)
        stats[run["name"]] = dict(
            length=per_bucket((m * w).norm(dim=-1)),
            strength=per_bucket(bank_norm(m * w, run["kind"]).norm(dim=-1)),
            turned=per_bucket(torch.rad2deg(torch.arccos(cos))),
            moved=per_bucket(moved),
            idle_share=per_bucket(
                run["idle_path"]
                / (run["idle_path"] + run["seen_path"]).clamp_min(1e-30)
            ),
            eps_damping=per_bucket(moved / moved_twin.clamp_min(1e-30)),
            toward_signal=per_bucket(F.cosine_similarity(w - w0, signal, dim=-1)),
        )
    labels = [b[0] for b in buckets]
    names = [v[0] for v in variants]
    blocks = [
        (
            "strength",
            "T1: strength at the end, ||normalized row|| (hard = 1 always; 1 = a fully trained row)",
        ),
        ("length", "Row length at the end, ||m * W|| per head (init: sigma * m * 8)"),
        ("moved", "T3: distance moved from init, ||m * (W_end - W_init)|| per head"),
        (
            "turned",
            "Angle between the row and its init draw, degrees (under the hard norm the direction is all that acts)",
        ),
        (
            "idle_share",
            "Share of the row's path taken on steps WITHOUT its token (AdamW momentum)",
        ),
        (
            "eps_damping",
            "Adam's eps: distance moved with eps = 1e-8, as a fraction of the distance with eps -> 0",
        ),
    ]
    if rho > 0:
        blocks.append(
            ("toward_signal", "cos(W_end - W_init, the consistent direction)")
        )
    for key, title in blocks:
        print(f"\n   {title}")
        table(
            names,
            labels,
            [[fmt(x) for x in stats[nm][key]] for nm in names],
            row_width=30,
            cell_width=10,
        )
    return stats


# ---- 7. Weight decay ----------------------------------------------------------------


def section_weight_decay(lrs, mult):
    header(
        "7. T2: weight decay (the banks have none today: ame.py no_weight_decay_parameters)"
    )
    wd = 0.1
    keep = math.prod(1 - lr * wd for lr in lrs)
    keep_equiv = math.prod(1 - mult * lr * wd for lr in lrs)
    print(
        f"   An idle row under AdamW at wd = {wd} (the backbone's) keeps {keep:.3f} of its length over the run;\n"
        f"   per 1,000 steps at the peak LR it keeps {(1 - PEAK_LR * wd) ** 1000:.3f}.\n"
        f"   With x{mult:g} on the lookup: decoupled decay multiplies W by (1 - lr * wd) whatever m is, so the output\n"
        f"   m * W keeps the same {keep:.3f}. A plain table at {mult:g}x the bank LR, which the multiplier otherwise\n"
        f"   equals, would keep {keep_equiv:.2e}: next to it, the multiplier weakens decay {mult:g}x.\n"
        f"   Under the hard norm decay never reaches a score, but a row at {keep:.3f} of its length turns\n"
        f"   {1 / keep:.1f}x faster per update by the end of the run."
    )


# ---- main ---------------------------------------------------------------------------


def buckets_from_counts(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    occ, u = data["occurrences"].double(), data["steps_seen"]
    out = []
    for label, lo, hi in [
        ("u=1", 1, 1),
        ("1-10", 1, 10),
        ("11-100", 11, 100),
        ("101-1000", 101, 1000),
        ("1001-3814", 1001, 3814),
        ("3815", 3815, 3815),
    ]:
        sel = (u >= lo) & (u <= hi)
        uu = int(u[sel].double().median().item())
        cc = (occ[sel] / u[sel].double()).median().item()
        out.append((f"u={uu}", uu, max(1.0, cc)))
    per_step = occ / data["steps"]
    ratio = math.sqrt((per_step**2).sum().item() / per_step.sum().item())
    return out, ratio


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mult", type=float, default=16.0)
    parser.add_argument("--rho", type=float, default=0.0)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--counts", default=None)
    parser.add_argument("--skip-module", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    m = args.mult
    small = 0.01 / m
    variants = [  # (name, norm, sigma, multiplier, bank LR factor)
        ("hard s=1 (production)", "hard", 1.0, 1.0, 1.0),
        ("hard s=0.01 (stage 2)", "hard", 0.01, 1.0, 1.0),
        (f"hard s=0.01 x{m:g}", "hard", 0.01, m, 1.0),
        ("soft s=0.01 (Plan A)", "soft", 0.01, 1.0, 1.0),
        (f"soft s=0.01 x{m:g}", "soft", 0.01, m, 1.0),
        (f"soft s={0.01 * m:g}", "soft", 0.01 * m, 1.0, 1.0),
        (f"soft s={small:.2g} x{m:g}", "soft", small, m, 1.0),
        (f"soft s=0.01, bank LR x{m:g}", "soft", 0.01, 1.0, m),
    ]
    buckets, coherent = BUCKETS, None
    if args.counts:
        buckets, coherent = buckets_from_counts(args.counts)
        print("buckets from", args.counts, ":", buckets)

    lrs = lr_schedule()
    print(
        f"Recipe: h12, {STEPS} steps, LR peak {PEAK_LR:.5f} (2^-8), warmup {WARMUP}, cosine to {FINAL_LR_FRACTION:g};"
        f" LR at steps 50/500/2000/3500 = "
        + "/".join(f"{lrs[t]:.5f}" for t in (50, 500, 2000, 3500))
    )
    if not args.skip_module:
        section_module(m)
    section_idle_and_gradient([v for v in variants if v[4] == 1.0])
    section_one_occurrence(lrs)
    section_calibration(lrs, variants, coherent)
    section_simulation(lrs, variants, buckets, args.rho, args.rows)
    section_weight_decay(lrs, m)


if __name__ == "__main__":
    main()
