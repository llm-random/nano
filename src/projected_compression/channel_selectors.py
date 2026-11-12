"""Functions that look at a model and return a ChannelSelection.

Convention: a selector is

    fn(model, *, <its own params>, source_model: str, notes: str = "") -> ChannelSelection

and ends by calling `selection_from_indices`, which reads the source dims from the model
and records the method and its params. A new selection criterion is just a new function;
code that computes indices elsewhere (a notebook, a one-off script) can call
`selection_from_indices` directly.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.projected_compression.channel_selection import ChannelSelection


def model_dims(model: nn.Module) -> dict:
    """Source dims of an (unpruned) nano LLM: {dmodel, dff, n_blocks}."""
    blocks = model.encoder.blocks
    return {
        "dmodel": model.embedding.weight.shape[1],
        "dff": blocks[0].ff_layer.layer.ff_pre_act.weight.shape[0],
        "n_blocks": len(blocks),
    }


def selection_from_indices(
    model: nn.Module,
    method: str,
    dmodel_keep=None,
    dff_keep=None,
    method_params: dict | None = None,
    source_model: str = "",
    notes: str = "",
) -> ChannelSelection:
    """Build a ChannelSelection for `model` from kept indices (any order, tensors or
    lists; None keeps the whole axis)."""

    def as_sorted(idx):
        return sorted(int(i) for i in torch.as_tensor(idx).flatten().tolist())

    return ChannelSelection(
        source_model=source_model,
        **model_dims(model),
        method=method,
        method_params=dict(method_params or {}),
        dmodel_keep=None if dmodel_keep is None else as_sorted(dmodel_keep),
        dff_keep=None if dff_keep is None else [as_sorted(b) for b in dff_keep],
        notes=notes,
    )


def topk_keep(importances, target_dmodel: int, target_dff: int):
    """Top-`target` channels per axis of an importances dict / `.pt` path, the same
    choice as pruning.prune(). Returns sorted (dmodel_keep, dff_keep); an axis whose
    target equals its size comes back as None (keep all)."""
    if not isinstance(importances, dict):
        importances = torch.load(importances, map_location="cpu")
    dmodel_imp = torch.as_tensor(importances["dmodel_importances"])
    dff_imp = importances["dff_importances"]
    # weight_importances.py saves a [n_blocks, dff] tensor; some callers a list of rows.
    if isinstance(dff_imp, (list, tuple)):
        dff_imp = torch.stack([torch.as_tensor(row) for row in dff_imp])

    def top(scores, k):
        return sorted(torch.topk(scores, k=k, dim=0, largest=True).indices.tolist())

    dmodel_keep = (
        None if target_dmodel == len(dmodel_imp) else top(dmodel_imp, target_dmodel)
    )
    dff_keep = (
        None
        if target_dff == dff_imp.shape[1]
        else [top(row, target_dff) for row in dff_imp]
    )
    return dmodel_keep, dff_keep


def select_by_importances(
    model: nn.Module,
    importances,
    target_dmodel: int,
    target_dff: int,
    *,
    source_model: str,
    notes: str = "",
) -> ChannelSelection:
    """Keep the most important channels of a precomputed importances file
    (src/projected_compression/weight_importances.py)."""
    dmodel_keep, dff_keep = topk_keep(importances, target_dmodel, target_dff)
    return selection_from_indices(
        model,
        "importance_topk",
        dmodel_keep,
        dff_keep,
        method_params={
            "importances_path": (
                None if isinstance(importances, dict) else str(importances)
            ),
            "target_dmodel": target_dmodel,
            "target_dff": target_dff,
        },
        source_model=source_model,
        notes=notes,
    )


@torch.no_grad()
def select_by_magnitude(
    model: nn.Module,
    target_dmodel: int,
    target_dff: int,
    *,
    source_model: str,
    notes: str = "",
) -> ChannelSelection:
    """Keep the channels with the largest weight norms
    (weight_importances._calculate_magnitude_dimension_importances)."""
    from src.projected_compression.weight_importances import (
        _calculate_magnitude_dimension_importances,
    )

    dmodel_imp, dff_imp = _calculate_magnitude_dimension_importances(model)
    dmodel_keep, dff_keep = topk_keep(
        {"dmodel_importances": dmodel_imp.cpu(), "dff_importances": dff_imp.cpu()},
        target_dmodel,
        target_dff,
    )
    return selection_from_indices(
        model,
        "magnitude_topk",
        dmodel_keep,
        dff_keep,
        method_params={"target_dmodel": target_dmodel, "target_dff": target_dff},
        source_model=source_model,
        notes=notes,
    )


def select_random_dff(
    model: nn.Module,
    drop_fraction: float,
    *,
    seed: int,
    multiple_of: int = 1,
    source_model: str,
    notes: str = "",
) -> ChannelSelection:
    """Drop `drop_fraction` of the FFN channels of every block, chosen uniformly at
    random and independently per block; dmodel is untouched. The same seed always gives
    the same selection. The kept count, dff - round(drop_fraction * dff), is rounded to
    the nearest multiple of `multiple_of`."""
    if not 0.0 <= drop_fraction < 1.0:
        raise ValueError(f"drop_fraction must be in [0, 1), got {drop_fraction}")
    dims = model_dims(model)
    n = dims["dff"]
    target_dff = kept_channel_count(n, (n - round(drop_fraction * n)) / n, multiple_of)

    generator = torch.Generator().manual_seed(seed)
    dff_keep = [
        torch.randperm(n, generator=generator)[:target_dff]
        for _ in range(dims["n_blocks"])
    ]
    return selection_from_indices(
        model,
        "random_dff",
        dff_keep=dff_keep,
        method_params={
            "drop_fraction": drop_fraction,
            "seed": seed,
            "multiple_of": multiple_of,
            "target_dff": target_dff,
        },
        source_model=source_model,
        notes=notes,
    )


@torch.no_grad()
def select_ffn_magnitude(
    model: nn.Module,
    keep_fraction: float,
    *,
    multiple_of: int = 1,
    source_model: str,
    notes: str = "",
) -> ChannelSelection:
    """Structured magnitude pruning of the FFN, as in section 4 of
    ~/bdd/notebooks/ffn_weight_distribution.ipynb: keep, in every block, the channels
    with the largest ||gate_i||^2 + ||up_i||^2 + ||down_i||^2 (row i of gate and up,
    column i of down); dmodel is untouched. The kept count is keep_fraction * dff rounded
    to the nearest multiple of `multiple_of` (see kept_channel_count)."""
    dims = model_dims(model)
    k = kept_channel_count(dims["dff"], keep_fraction, multiple_of)
    dff_keep = []
    for block in model.encoder.blocks:
        weights = _ffn_channel_weights(block)
        score = sum(weights[m].pow(2).sum(1) for m in _FFN_MATRICES)
        dff_keep.append(score.topk(k).indices)
    return selection_from_indices(
        model,
        "ffn_magnitude",
        dff_keep=dff_keep,
        method_params={
            "score": "||gate_i||^2 + ||up_i||^2 + ||down_i||^2",
            "keep_fraction": keep_fraction,
            "multiple_of": multiple_of,
            "target_dff": k,
        },
        source_model=source_model,
        notes=notes,
    )


# --- FFN channels closest to the init distribution ---------------------------------------
#
# Port of section 4 of ~/bdd/notebooks/ffn_weight_distribution.ipynb. For every block we
# pick K FFN channels (one index set shared by gate / up / down: channel i is row i of
# gate and up and column i of down) minimising
#
#     J(S) = 1/3 * sum_m W1(F_{m,S}, N(0, sigma_m)) / sigma_m,
#
# where F_{m,S} is the weight distribution of matrix m restricted to the channels in S.
# Every channel has the same number of weights, so F_{m,S} is the mean of the per-channel
# CDFs; those are tabulated once on a grid, which makes J of many subsets one matmul.
# Search: best of `n_random` uniform subsets, then local search by channel swaps. J is
# convex in the relaxation x in [0,1]^n, sum x = K, so a Frank-Wolfe gap gives a lower
# bound J* >= J(S) - gap on any K-subset, recorded with the result.

_FFN_MATRICES = ("gate", "up", "down")
INIT_REFERENCES = ("huggingface", "torchtitan")


def init_reference_std(reference: str, matrix: str, layer: int, std: float = 0.02):
    """Std of the init distribution a matrix is compared against.

    huggingface -- HF LlamaConfig: N(0, std) for gate, up and down.
    torchtitan  -- torchtitan depth_init: gate N(0, std), up/down N(0, std / sqrt(2 (layer + 1))).
    """
    if reference not in INIT_REFERENCES:
        raise ValueError(
            f"reference must be one of {INIT_REFERENCES}, got {reference!r}"
        )
    if reference == "huggingface" or matrix == "gate":
        return std
    return std / math.sqrt(2 * (layer + 1))


def _ffn_channel_weights(block) -> dict:
    """{gate, up, down} -> [dff, dmodel] float32 CPU tensors; row i holds channel i."""
    ff = block.ff_layer.layer
    if not hasattr(ff, "gate"):
        raise ValueError("init-distribution selection needs a gated (SwiGLU) FFN")
    return {
        "gate": ff.gate.weight.detach().float().cpu(),
        "up": ff.ff_pre_act.weight.detach().float().cpu(),
        "down": ff.ff_post_act.weight.detach().float().cpu().T.contiguous(),
    }


class _ChannelCdfs:
    """Per-channel CDFs of one matrix on a grid, and the reference N(0, sigma) CDF."""

    def __init__(self, w: torch.Tensor, sigma: float, n_quantiles: int = 1024):
        self.sigma = sigma
        rows = w.sort(dim=1).values
        flat = rows.flatten().sort().values
        data_q = flat[torch.linspace(0, flat.numel() - 1, n_quantiles).round().long()]
        ref_q = sigma * torch.linspace(-6, 6, n_quantiles)
        # sorted, and contains the data min and max
        edges = torch.cat([data_q, ref_q]).unique()
        # C[i, b] = F_i(e_b), the CDF of channel i at grid point b
        self.C = (
            torch.searchsorted(
                rows, edges.expand(rows.shape[0], -1).contiguous(), right=True
            ).float()
            / rows.shape[1]
        )
        self.F = torch.special.ndtr(edges.double() / sigma).float()
        self.D = torch.cat([edges.diff(), torch.zeros(1)])
        za, zb = edges[0].item() / sigma, edges[-1].item() / sigma
        phi = lambda z: math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
        Phi = lambda z: 0.5 * math.erfc(-z / math.sqrt(2))
        # Reference mass outside the grid, where the subset CDF is 0 (left) or 1 (right):
        # int_{-inf}^{e_0} Phi(x / sigma) dx + int_{e_last}^{inf} (1 - Phi(x / sigma)) dx
        self.tail = sigma * (phi(za) + za * Phi(za)) + sigma * (
            phi(zb) - zb * (1 - Phi(zb))
        )

    def w1(self, cdf: torch.Tensor) -> torch.Tensor:
        """W1 distance to the reference of subset CDFs `cdf` [..., n_edges]."""
        return (cdf - self.F).abs() @ self.D + self.tail


def _objective(problems, sums, k):
    """J for subsets given by their summed channel CDFs `sums` (one per matrix)."""
    return sum(p.w1(s / k) / p.sigma for p, s in zip(problems, sums)) / len(problems)


def _subgradient(problems, sums, k):
    return sum(
        p.C @ (torch.sign(s / k - p.F) * p.D) / p.sigma for p, s in zip(problems, sums)
    )


def _random_subsets(n, k, count, generator):
    idx = torch.rand(count, n, generator=generator).argsort(dim=1)[:, :k]
    return torch.zeros(count, n).scatter_(1, idx, 1.0)


def _local_search(
    problems, selected, t=256, t_max=1024, m_exact=64, max_steps=3000, sign=1.0
):
    """Improve a boolean channel mask by swaps, minimising sign * J (sign=-1 maximises J);
    returns (mask, J history)."""
    n = selected.numel()
    k = int(selected.sum())
    # never consider more exact-swap candidates than there are channels on each side
    m_in, m_out = min(m_exact, n - k), min(m_exact, k)
    sums = [selected.float() @ p.C for p in problems]
    cost = sign * _objective(problems, sums, k).item()
    history = [cost]
    while len(history) < max_steps and m_in > 0:
        g = sign * _subgradient(problems, sums, k)
        g_in = torch.where(selected, torch.inf, g)  # candidates to add: smallest g
        g_out = torch.where(selected, g, -torch.inf)  # candidates to drop: largest g
        ins = g_in.argsort()[:t]
        outs = g_out.argsort(descending=True)[:t]
        good = g_in[ins] < g_out[outs]
        ins, outs = ins[good], outs[good]
        new_cost = math.inf
        if len(ins):
            new_sums = [
                s + p.C[ins].sum(0) - p.C[outs].sum(0) for p, s in zip(problems, sums)
            ]
            new_cost = sign * _objective(problems, new_sums, k).item()
        if new_cost >= cost - 1e-9:
            # The linear step does not help (|.| in W1 is not smooth): try every single
            # swap among the best m_in x m_out candidates exactly and take the best one.
            ins = g_in.argsort()[:m_in]
            outs = g_out.argsort(descending=True)[:m_out]
            pair_sums = [
                s[None, None] + p.C[ins][:, None] - p.C[outs][None, :]
                for p, s in zip(problems, sums)
            ]
            costs = sign * _objective(problems, pair_sums, k)
            best = costs.argmin().item()
            if costs.flatten()[best].item() >= cost - 1e-9:
                break
            ins = ins[best // m_out : best // m_out + 1]
            outs = outs[best % m_out : best % m_out + 1]
            new_sums = [
                s + p.C[ins].sum(0) - p.C[outs].sum(0) for p, s in zip(problems, sums)
            ]
            new_cost = sign * _objective(problems, new_sums, k).item()
            t = max(t // 2, 1)
        else:
            t = min(2 * t, t_max)
        selected = selected.clone()
        selected[ins], selected[outs] = True, False
        sums, cost = new_sums, new_cost
        history.append(cost)
    return selected, [sign * c for c in history]


def _frank_wolfe_gap(problems, selected):
    """J(S) - gap is a lower bound on J of every subset of the same size."""
    k = int(selected.sum())
    sums = [selected.float() @ p.C for p in problems]
    g = _subgradient(problems, sums, k) / (len(problems) * k)
    return (g[selected].sum() - g.topk(k, largest=False).values.sum()).item()


def kept_channel_count(n: int, keep_fraction: float, multiple_of: int = 1) -> int:
    """keep_fraction * n rounded to the nearest multiple of `multiple_of` (ties go to the
    even multiple, as Python's round does), clamped to [multiple_of, n]."""
    if multiple_of < 1:
        raise ValueError(f"multiple_of must be >= 1, got {multiple_of}")
    k = round(keep_fraction * n / multiple_of) * multiple_of
    k = min(k, n)
    if k < 1:
        raise ValueError(
            f"keep_fraction {keep_fraction} of {n} channels keeps no channels"
        )
    return k


def _select_by_init_distance(
    model,
    keep_fraction,
    *,
    farthest,
    reference,
    init_std,
    n_random,
    seed,
    max_steps,
    multiple_of,
    source_model,
    notes,
) -> ChannelSelection:
    """Shared body of select_closest_to_init (farthest=False: minimise J) and
    select_farthest_from_init (farthest=True: maximise J)."""
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError(f"keep_fraction must be in (0, 1], got {keep_fraction}")
    init_reference_std(reference, "up", 0)  # validates `reference`
    dims = model_dims(model)
    n = dims["dff"]
    k = kept_channel_count(n, keep_fraction, multiple_of)
    sign = -1.0 if farthest else 1.0

    generator = torch.Generator().manual_seed(seed)
    dff_keep, per_block = [], []
    with torch.no_grad():
        for layer, block in enumerate(model.encoder.blocks):
            weights = _ffn_channel_weights(block)
            problems = [
                _ChannelCdfs(
                    weights[m], init_reference_std(reference, m, layer, init_std)
                )
                for m in _FFN_MATRICES
            ]
            masks = _random_subsets(n, k, n_random, generator)
            random_cost = _objective(problems, [masks @ p.C for p in problems], k)
            best = (sign * random_cost).argmin()
            selected, history = _local_search(
                problems, masks[best].bool(), max_steps=max_steps, sign=sign
            )
            dff_keep.append(selected.nonzero().squeeze(1))
            stats = {
                "all_channels": _objective(
                    problems, [p.C.sum(0) for p in problems], n
                ).item(),
                "random_median": random_cost.median().item(),
                "random_best": random_cost[best].item(),
                "optimized": history[-1],
            }
            if not farthest:
                # Convexity gives a lower bound on the minimum only; there is no
                # cheap certificate for the maximum.
                stats["lower_bound"] = history[-1] - _frank_wolfe_gap(
                    problems, selected
                )
            stats["swap_steps"] = len(history) - 1
            per_block.append(stats)

    return selection_from_indices(
        model,
        "farthest_from_init" if farthest else "closest_to_init",
        dff_keep=dff_keep,
        method_params={
            "objective": "mean over gate/up/down of W1(weights, N(0, sigma_ref)) / sigma_ref",
            "direction": "maximize" if farthest else "minimize",
            "keep_fraction": keep_fraction,
            "multiple_of": multiple_of,
            "target_dff": k,
            "reference": reference,
            "init_std": init_std,
            "n_random": n_random,
            "seed": seed,
            "max_steps": max_steps,
            "per_block": per_block,
        },
        source_model=source_model,
        notes=notes,
    )


def select_closest_to_init(
    model: nn.Module,
    keep_fraction: float,
    *,
    reference: str = "huggingface",
    init_std: float = 0.02,
    n_random: int = 2000,
    seed: int = 0,
    max_steps: int = 3000,
    multiple_of: int = 1,
    source_model: str,
    notes: str = "",
) -> ChannelSelection:
    """Keep, in every block, the FFN channels whose remaining gate / up / down weights are
    closest (mean W1 / sigma) to the init distribution `reference` (see
    init_reference_std); dmodel is untouched.

    The number of kept channels is keep_fraction * dff rounded to the nearest multiple
    of `multiple_of` (see kept_channel_count).

    Per-block objective values are recorded in method_params["per_block"]:
    all channels, median and best of the random subsets, the optimised subset, and the
    Frank-Wolfe lower bound on any subset of that size.
    """
    return _select_by_init_distance(
        model,
        keep_fraction,
        farthest=False,
        reference=reference,
        init_std=init_std,
        n_random=n_random,
        seed=seed,
        max_steps=max_steps,
        multiple_of=multiple_of,
        source_model=source_model,
        notes=notes,
    )


def select_farthest_from_init(
    model: nn.Module,
    keep_fraction: float,
    *,
    reference: str = "huggingface",
    init_std: float = 0.02,
    n_random: int = 2000,
    seed: int = 0,
    max_steps: int = 3000,
    multiple_of: int = 1,
    source_model: str,
    notes: str = "",
) -> ChannelSelection:
    """The opposite of select_closest_to_init: keep, in every block, the FFN channels
    whose remaining gate / up / down weights are farthest (mean W1 / sigma) from the
    init distribution `reference`. Same objective and search (best of `n_random` random
    subsets, then swap-based local search), maximising instead of minimising.

    Per-block objective values are recorded in method_params["per_block"]:
    all channels, median and best (largest) of the random subsets, and the optimised
    subset. No bound is recorded: maximising a convex function has no cheap certificate.
    """
    return _select_by_init_distance(
        model,
        keep_fraction,
        farthest=True,
        reference=reference,
        init_std=init_std,
        n_random=n_random,
        seed=seed,
        max_steps=max_steps,
        multiple_of=multiple_of,
        source_model=source_model,
        notes=notes,
    )
