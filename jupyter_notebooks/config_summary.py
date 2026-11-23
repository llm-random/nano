"""Internals of config_summary.ipynb: compose a run config as run_exp.py does and summarise it."""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for path in (REPO, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import resolver  # noqa: E402,F401  (registers the eval resolver)
import src.core.model  # noqa: E402,F401  (imported inside torch.device("meta") it fails)
import src.definitions  # noqa: E402,F401  (registers /base_schema)
from grid_generator.generate_configs import create_grid_config  # noqa: E402
from mfu_calc import PEAK_BF16  # noqa: E402
from src.core.mfu import model_flops_per_token  # noqa: E402
from src.definitions import check_point, validate  # noqa: E402


def load(config):
    """Grid points of configs/<config>.yaml as (overrides, checked cfg), validated as main.py does."""
    directory, name = os.path.split(config)
    config_dir = os.path.join(REPO, "configs", directory)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=name)
    points = []
    for point, overrides in create_grid_config(cfg):
        checked = check_point(OmegaConf.create(point))
        validate(checked)
        points.append((overrides, checked))
    return points


def _model_stats(cfg, tokens_per_row):
    with torch.device("meta"):
        model = instantiate(cfg.model, _convert_="all")
    total = sum(p.numel() for p in model.parameters())
    # MFU's P: every 2-D weight but the embedding lookup
    active = sum(
        p.numel()
        for n, p in model.named_parameters()
        if p.dim() == 2 and "embedding" not in n
    )
    return total, active, model_flops_per_token(model, tokens_per_row)


def _stages(trainer, batch_size):
    """(batch, first step, end step) per batch stage, the final batch last."""
    stages, step = [], 0
    warmup = trainer.batch_size_warmup
    if warmup is not None:
        for size, steps in zip(warmup.batch_sizes, warmup.steps):
            stages.append((size, step, step + steps))
            step += steps
    stages.append((batch_size, step, trainer.n_steps))
    return stages


def _phases(trainer):
    """(LR phase, first step, end step)."""
    n, scheduler = trainer.n_steps, trainer.scheduler
    warmup = scheduler.warmup_steps
    if "WSD" in scheduler._target_:
        decay = scheduler.decay_steps
        return [
            ("warmup", 0, warmup),
            ("stable", warmup, n - decay),
            ("decay", n - decay, n),
        ]
    return [("warmup", 0, warmup), ("cosine", warmup, n)]


def _rows(stages, start, end):
    """Sequences trained in steps start..end-1."""
    return sum(b * max(0, min(end, e) - max(start, s)) for b, s, e in stages)


def _slurm_hours(text):
    """SLURM time "[D-]HH:MM:SS" in hours; None without a limit."""
    if text is None:
        return None
    days, _, clock = str(text).rpartition("-")
    h, m, s = (int(x) for x in clock.split(":"))
    return int(days or 0) * 24 + h + m / 60 + s / 3600


def _n_gpus(cfg):
    gres = OmegaConf.select(cfg, "infrastructure.slurm.gres")
    match = re.search(r"(\d+)$", str(gres)) if gres else None
    return int(match.group(1)) if match else 1


def _tok(x):
    if x >= 1e9:
        return f"{x / 1e9:.2f}B"
    return f"{x / 1e6:.1f}M" if x >= 1e6 else f"{x:,}"


def summarize(config, point=0, gpu="h100", mfu=0.35):
    """A ~10-line summary of one grid point of `config` ("jm/h12_chinchilla")."""
    points = load(config)
    cfg = points[point][1]
    t, c = cfg.trainer, cfg.common
    n = t.n_steps
    row = c.sequence_length - 1  # trained tokens per row, as W&B's `tokens` counts
    stages, phases = _stages(t, c.batch_size), _phases(t)
    total, active, flops = _model_stats(cfg, row)
    tokens = _rows(stages, 0, n) * row

    swept = {}
    for overrides, _ in points:
        for override in overrides:
            key, _, value = override.partition("=")
            swept.setdefault(key.rsplit(".", 1)[-1], {})[value] = None
    sweep = "; ".join(f"{k} {', '.join(v)}" for k, v in swept.items())
    print(f"{config}: {len(points)} run(s){' over ' + sweep if sweep else ''}")
    print(
        f"model   h{c.h}, dff {c.dff / c.dmodel:g}x, qk_norm {c.attention.qk_norm} | "
        f"{total / 1e6:.1f}M params, {active / 1e6:.1f}M active | {flops / 1e9:.3f} GFLOPs/token"
    )
    print(
        f"tokens  {_tok(tokens)} in {n:_} steps, {tokens / active:.1f} per active param"
    )

    # one row per stretch of steps with the same batch and LR phase
    cuts = sorted({n} | {s for _, s, _ in stages} | {s for _, s, _ in phases})
    table = []
    for a, b in zip(cuts, cuts[1:]):
        part = _rows(stages, a, b) * row
        table.append(
            {
                "steps": f"{a}..{b - 1}",
                "batch": next(size for size, s, e in stages if s <= a < e),
                "LR": next(name for name, s, e in phases if s <= a < e),
                "tokens": _tok(part),
                "%": f"{part / tokens:.1%}",
            }
        )
    print(pd.DataFrame(table).to_string(index=False))

    interval = t.checkpoint.save.interval
    line = (
        f"ckpt    every {interval:_}"
        if t.checkpoint.save.path and interval > 0
        else "ckpt    final only"
    )
    if phases[-1][0] == "decay" and interval > 0:
        last_stable = phases[-1][1] - 1
        saved = (last_stable + 1) % interval == 0
        line += f", last stable step {last_stable} {'saved' if saved else 'NOT saved'}"
    evals = OmegaConf.select(cfg, "downstream_evaluator.eval_interval")
    if evals and evals != interval:
        line += f"; downstream eval every {evals:_}"
    print(line)

    n_gpus = _n_gpus(cfg)
    hours = flops * tokens / (mfu * PEAK_BF16[gpu] * n_gpus) / 3600
    limit = _slurm_hours(OmegaConf.select(cfg, "infrastructure.slurm.time"))
    print(
        f"time    {hours:.2f} h at {mfu:.0%} MFU on {n_gpus}x {gpu}"
        f"{f' (limit {limit:g} h)' if limit else ''}, {len(points) * hours * n_gpus:.1f} GPU-h in all"
    )
