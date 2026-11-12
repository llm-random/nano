"""CPU tests for ChannelSelection (channel_selection.py), the selectors
(channel_selectors.py) and pruning.prune_from_selection. Uses a tiny nano Llama composed from the real configs.
"""

import copy
import json
import os
import subprocess
import sys

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import resolver  # noqa: F401  (registers custom OmegaConf resolvers, e.g. `eval`)
from src.core.model import RMSNorm
from src.projected_compression.channel_selection import (
    ChannelSelection,
    describe,
    load_selection,
    save_selection,
)
from src.projected_compression.channel_selectors import (
    select_by_importances,
    init_reference_std,
    kept_channel_count,
    select_by_magnitude,
    select_closest_to_init,
    select_farthest_from_init,
    select_ffn_magnitude,
    select_random_dff,
    selection_from_indices,
)
from src.projected_compression.weight_importances import (
    _calculate_magnitude_dimension_importances,
)
from src.projected_compression.pruning import prune, prune_from_selection

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TINY = dict(
    dmodel=64,
    dff=96,
    datt=64,
    n_blocks=3,
    q_heads=4,
    kv_heads=2,
    vocab_size=101,
    sequence_length=32,
)


def _tiny_model(seed=0, **dims):
    t = {**TINY, **dims}
    with initialize_config_dir(
        version_base=None, config_dir=os.path.join(REPO_ROOT, "configs/pc_project")
    ):
        cfg = compose(
            config_name="llama_1B_prune",
            overrides=[f"common.{k}={v}" for k, v in t.items()],
        )
    torch.manual_seed(seed)
    model = instantiate(cfg.model, _convert_="all").to(torch.float64).eval()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, RMSNorm):
                m.weight.uniform_(0.5, 1.5)
    return model


@torch.no_grad()
def _logits(model, seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(
        0, TINY["vocab_size"], (2, TINY["sequence_length"]), generator=g
    )
    return model(ids)


def _importances(seed=3):
    g = torch.Generator().manual_seed(seed)
    return {
        "dmodel_importances": torch.rand(TINY["dmodel"], generator=g),
        "dff_importances": torch.rand(TINY["n_blocks"], TINY["dff"], generator=g),
    }


def _selection(**kw):
    base = dict(
        source_model="tiny",
        dmodel=8,
        dff=10,
        n_blocks=2,
        method="manual",
        dmodel_keep=[0, 3, 5],
        dff_keep=[[1, 2], [0, 9]],
    )
    return ChannelSelection(**{**base, **kw})


def test_save_load_roundtrip_is_plain_json(tmp_path):
    sel = _selection(method_params={"keep": 0.7}, notes="hello")
    path = tmp_path / "sel.json"
    save_selection(sel, path)
    raw = json.loads(path.read_text())
    assert raw["dff_keep"] == [[1, 2], [0, 9]] and raw["source_model"] == "tiny"
    assert load_selection(path) == sel
    assert sel.target_dmodel == 3 and sel.target_dff == 2
    assert "tiny" in describe(sel) and "hello" in describe(sel)


@pytest.mark.parametrize(
    "kw",
    [
        dict(dmodel_keep=[3, 0]),  # unsorted
        dict(dmodel_keep=[0, 0]),  # duplicate
        dict(dmodel_keep=[0, 8]),  # out of range
        dict(dmodel_keep=[]),  # empty
        dict(dff_keep=[[1, 2], [0]]),  # ragged
        dict(dff_keep=[[1, 2]]),  # wrong block count
        dict(format_version=99),
    ],
)
def test_rejects_bad_records(kw):
    with pytest.raises(ValueError):
        _selection(**kw)


def test_load_rejects_unknown_fields(tmp_path):
    path = tmp_path / "sel.json"
    save_selection(_selection(), path)
    d = json.loads(path.read_text())
    d["bogus"] = 1
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        load_selection(path)


def test_selection_from_indices_reads_dims_and_sorts():
    model = _tiny_model()
    sel = selection_from_indices(
        model,
        "manual",
        dff_keep=[torch.tensor([5, 1, 3])] * TINY["n_blocks"],
        source_model="tiny",
    )
    assert (sel.dmodel, sel.dff, sel.n_blocks) == (
        TINY["dmodel"],
        TINY["dff"],
        TINY["n_blocks"],
    )
    assert sel.dmodel_keep is None and sel.dff_keep[0] == [1, 3, 5]


def test_select_by_importances_matches_topk(tmp_path):
    imp = _importances()
    path = tmp_path / "imp.pt"
    torch.save(imp, path)
    sel = select_by_importances(_tiny_model(), path, 40, 50, source_model="tiny")
    assert sel.method == "importance_topk"
    assert sel.method_params["importances_path"] == str(path)
    assert set(sel.dmodel_keep) == set(
        torch.topk(imp["dmodel_importances"], 40).indices.tolist()
    )
    for keep, scores in zip(sel.dff_keep, imp["dff_importances"]):
        assert set(keep) == set(torch.topk(scores, 50).indices.tolist())
    # list-of-rows importances (as some callers save them) give the same record
    rows = {**imp, "dff_importances": list(imp["dff_importances"])}
    other = select_by_importances(_tiny_model(), rows, 40, 50, source_model="tiny")
    assert other.dff_keep == sel.dff_keep


def test_select_by_magnitude_matches_topk():
    model = _tiny_model()
    dmodel_imp, dff_imp = _calculate_magnitude_dimension_importances(model)
    sel = select_by_magnitude(model, 40, 50, source_model="tiny")
    assert sel.method == "magnitude_topk"
    assert set(sel.dmodel_keep) == set(torch.topk(dmodel_imp, 40).indices.tolist())
    for keep, scores in zip(sel.dff_keep, dff_imp):
        assert set(keep) == set(torch.topk(scores, 50).indices.tolist())


def test_prune_from_selection_matches_prune(tmp_path):
    imp_path = tmp_path / "imp.pt"
    torch.save(_importances(), imp_path)
    a = _tiny_model()
    b = copy.deepcopy(a)
    sel_path = tmp_path / "sel.json"
    save_selection(
        select_by_importances(b, imp_path, 40, 50, source_model="tiny"), sel_path
    )

    prune(a, imp_path, target_dmodel=40, target_dff=50)
    prune_from_selection(b, sel_path)

    # topk order vs sorted order permutes channels consistently: same function
    torch.testing.assert_close(_logits(a), _logits(b))
    assert b.embedding.weight.shape == (TINY["vocab_size"], 40)
    for block in b.encoder.blocks:
        ff = block.ff_layer.layer
        assert ff.ff_pre_act.weight.shape == (50, 40)
        assert ff.gate.weight.shape == (50, 40)
        assert ff.ff_post_act.weight.shape == (40, 50)
    # module size attributes follow the pruned weights, not the original sizes
    for module in b.modules():
        if isinstance(module, torch.nn.Linear):
            assert (module.out_features, module.in_features) == module.weight.shape
        elif isinstance(module, torch.nn.Embedding):
            assert module.embedding_dim == 40
    ff = b.encoder.blocks[0].ff_layer.layer
    assert (ff.ff_pre_act.out_features, ff.ff_post_act.in_features) == (50, 50)
    assert b.head.linear.in_features == 40


def test_prune_from_selection_ffn_only(tmp_path):
    model = _tiny_model()
    originals = [
        b.ff_layer.layer.ff_pre_act.weight.clone() for b in model.encoder.blocks
    ]
    keep = [list(range(0, TINY["dff"], 2))] * TINY["n_blocks"]
    path = tmp_path / "sel.json"
    save_selection(selection_from_indices(model, "every_other", dff_keep=keep), path)
    prune_from_selection(model, path)
    assert model.embedding.weight.shape[1] == TINY["dmodel"]
    for block, orig, idx in zip(model.encoder.blocks, originals, keep):
        assert torch.equal(block.ff_layer.layer.ff_pre_act.weight, orig[idx])
    _logits(model)  # runs


def test_prune_from_selection_rejects_other_model(tmp_path):
    path = tmp_path / "sel.json"
    save_selection(
        ChannelSelection(
            source_model="tiny",
            dmodel=TINY["dmodel"],
            dff=TINY["dff"] * 2,
            n_blocks=TINY["n_blocks"],
            method="manual",
        ),
        path,
    )
    with pytest.raises(ValueError, match="model has"):
        prune_from_selection(_tiny_model(), path)


def test_cli_from_importances_and_show(tmp_path):
    imp_path = tmp_path / "imp.pt"
    torch.save(_importances(), imp_path)
    out = tmp_path / "sel.json"
    script = os.path.join(REPO_ROOT, "scripts/channel_selection.py")
    subprocess.run(
        [
            sys.executable,
            script,
            "from-importances",
            str(imp_path),
            "--target-dmodel",
            "40",
            "--target-dff",
            "50",
            "--source-model",
            "tiny",
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    shown = subprocess.run(
        [sys.executable, script, "show", str(out)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "importance_topk" in shown and "50 / 96" in shown


def test_select_random_dff_drops_fraction_per_block():
    model = _tiny_model()
    sel = select_random_dff(model, 0.25, seed=7, source_model="tiny")
    assert sel.method == "random_dff"
    assert sel.method_params == {
        "drop_fraction": 0.25,
        "seed": 7,
        "multiple_of": 1,
        "target_dff": 72,
    }
    assert sel.dmodel_keep is None
    assert sel.target_dff == TINY["dff"] - round(0.25 * TINY["dff"])
    assert len({tuple(b) for b in sel.dff_keep}) > 1  # independent per block
    # same seed -> same selection, different seed -> different one
    assert (
        select_random_dff(model, 0.25, seed=7, source_model="tiny").dff_keep
        == sel.dff_keep
    )
    assert (
        select_random_dff(model, 0.25, seed=8, source_model="tiny").dff_keep
        != sel.dff_keep
    )
    # nothing dropped keeps every channel
    assert select_random_dff(model, 0.0, seed=0, source_model="tiny").dff_keep[
        0
    ] == list(range(TINY["dff"]))
    with pytest.raises(ValueError):
        select_random_dff(model, 1.0, seed=0, source_model="tiny")


def test_random_dff_selection_prunes_model(tmp_path):
    model = _tiny_model()
    path = tmp_path / "sel.json"
    save_selection(select_random_dff(model, 0.5, seed=0, source_model="tiny"), path)
    prune_from_selection(model, path)
    assert model.embedding.weight.shape[1] == TINY["dmodel"]
    for block in model.encoder.blocks:
        assert block.ff_layer.layer.ff_pre_act.weight.shape == (48, TINY["dmodel"])
    _logits(model)  # runs


def test_init_reference_std():
    assert init_reference_std("huggingface", "down", 15) == 0.02
    assert init_reference_std("torchtitan", "gate", 15) == 0.02
    assert init_reference_std("torchtitan", "up", 1) == pytest.approx(0.02 / 2)
    with pytest.raises(ValueError):
        init_reference_std("other", "up", 0)


def _plant_ffn_channels(model, good, sigma_good=0.02, sigma_bad=0.2, seed=0):
    """Draw the `good` FFN channels of every block from N(0, sigma_good), the rest from
    N(0, sigma_bad)."""
    g = torch.Generator().manual_seed(seed)
    std = torch.full((TINY["dff"],), sigma_bad, dtype=torch.float64)
    std[good] = sigma_good
    with torch.no_grad():
        for block in model.encoder.blocks:
            ff = block.ff_layer.layer
            for w in (ff.gate.weight, ff.ff_pre_act.weight):
                w.copy_(torch.randn(w.shape, generator=g, dtype=w.dtype) * std[:, None])
            down = ff.ff_post_act.weight
            down.copy_(
                torch.randn(down.shape, generator=g, dtype=down.dtype) * std[None, :]
            )


def test_select_closest_to_init_finds_planted_channels():
    model = _tiny_model()
    good = torch.randperm(TINY["dff"], generator=torch.Generator().manual_seed(1))[:48]
    _plant_ffn_channels(model, good)
    sel = select_closest_to_init(
        model, keep_fraction=0.5, n_random=200, seed=0, source_model="tiny"
    )
    assert sel.method == "closest_to_init"
    assert sel.dmodel_keep is None and sel.target_dff == 48
    for keep in sel.dff_keep:
        assert keep == sorted(good.tolist())
    for stats in sel.method_params["per_block"]:
        assert stats["optimized"] <= stats["random_best"] <= stats["random_median"]
        assert stats["optimized"] < stats["all_channels"]
        assert stats["lower_bound"] <= stats["optimized"] + 1e-6


def test_select_closest_to_init_is_deterministic_and_prunes(tmp_path):
    model = _tiny_model()
    kw = dict(
        keep_fraction=0.7,
        reference="torchtitan",
        n_random=100,
        seed=3,
        source_model="tiny",
    )
    sel = select_closest_to_init(model, **kw)
    assert select_closest_to_init(model, **kw).dff_keep == sel.dff_keep
    assert sel.target_dff == round(0.7 * TINY["dff"])
    path = tmp_path / "sel.json"
    save_selection(sel, path)
    assert load_selection(path) == sel
    prune_from_selection(model, path)
    for block in model.encoder.blocks:
        assert block.ff_layer.layer.gate.weight.shape == (
            sel.target_dff,
            TINY["dmodel"],
        )
    _logits(model)  # runs


def test_select_farthest_from_init_keeps_the_other_planted_channels():
    model = _tiny_model()
    good = torch.randperm(TINY["dff"], generator=torch.Generator().manual_seed(1))[:48]
    _plant_ffn_channels(model, good)
    sel = select_farthest_from_init(
        model, keep_fraction=0.5, n_random=200, seed=0, source_model="tiny"
    )
    assert sel.method == "farthest_from_init"
    assert sel.method_params["direction"] == "maximize"
    bad = sorted(set(range(TINY["dff"])) - set(good.tolist()))
    for keep in sel.dff_keep:
        assert keep == bad
    for stats in sel.method_params["per_block"]:
        assert stats["optimized"] >= stats["random_best"] >= stats["random_median"]
        assert stats["optimized"] > stats["all_channels"]
        assert "lower_bound" not in stats


def test_farthest_and_closest_to_init_disagree_on_a_real_init(tmp_path):
    model = _tiny_model()
    kw = dict(keep_fraction=0.5, n_random=100, seed=0, multiple_of=8)
    near = select_closest_to_init(model, source_model="tiny", **kw)
    far = select_farthest_from_init(model, source_model="tiny", **kw)
    assert far.target_dff == near.target_dff == 48
    for b_near, b_far in zip(
        near.method_params["per_block"], far.method_params["per_block"]
    ):
        assert b_far["optimized"] > b_near["optimized"]
    assert (
        select_farthest_from_init(model, source_model="tiny", **kw).dff_keep
        == far.dff_keep
    )
    path = tmp_path / "far.json"
    save_selection(far, path)
    prune_from_selection(model, path)
    _logits(model)  # runs


def test_kept_channel_count_rounds_to_multiple():
    counts = [kept_channel_count(8192, p / 100, 8) for p in range(30, 100, 10)]
    assert counts == [2456, 3280, 4096, 4912, 5736, 6552, 7376]
    assert kept_channel_count(8192, 0.7) == 5734
    assert kept_channel_count(256, 1.0, 8) == 256
    with pytest.raises(ValueError):
        kept_channel_count(256, 0.01, 8)


def test_select_closest_to_init_multiple_of():
    sel = select_closest_to_init(
        _tiny_model(), 0.7, n_random=50, multiple_of=8, source_model="tiny"
    )
    assert sel.target_dff == 64 and sel.method_params["multiple_of"] == 8


def test_select_random_dff_multiple_of():
    sel = select_random_dff(
        _tiny_model(), 0.3, seed=0, multiple_of=8, source_model="tiny"
    )
    # 96 - round(0.3 * 96) = 67 -> nearest multiple of 8 is 64
    assert sel.target_dff == 64 and sel.method_params["multiple_of"] == 8


def test_select_ffn_magnitude_keeps_largest_channels():
    model = _tiny_model()
    sel = select_ffn_magnitude(model, 0.7, multiple_of=8, source_model="tiny")
    assert sel.method == "ffn_magnitude" and sel.dmodel_keep is None
    assert sel.target_dff == kept_channel_count(TINY["dff"], 0.7, 8) == 64
    for block, keep in zip(model.encoder.blocks, sel.dff_keep):
        ff = block.ff_layer.layer
        score = (
            ff.gate.weight.pow(2).sum(1)
            + ff.ff_pre_act.weight.pow(2).sum(1)
            + ff.ff_post_act.weight.pow(2).sum(0)
        )
        assert set(keep) == set(score.topk(64).indices.tolist())
