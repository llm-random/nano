"""CPU unit tests for the SliceGPT port (src/projected_compression/slicegpt), see slicegpt.md.

Runs with pytest, or without it:  python test_slicegpt.py
Uses a tiny nano Llama composed from the real configs (dims overridden), so no GPU / data needed.
"""

import math
import os
import tempfile

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import resolver  # noqa: F401  (registers custom OmegaConf resolvers, e.g. `eval`)
from src.core.model import RMSNorm
from src.projected_compression.slicegpt import (
    finalize_norms,
    fuse_norms,
    prune_dff,
    rotate_and_slice,
)
from src.projected_compression.slicegpt.fusion import RMSN

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


def _tiny_overrides(dmodel=TINY["dmodel"], dff=TINY["dff"]):
    t = {**TINY, "dmodel": dmodel, "dff": dff}
    return [f"common.{k}={v}" for k, v in t.items()]


# hydra's initialize() resolves config_path relative to the calling file, which pinned
# this module to the repo root. initialize_config_dir takes an absolute path instead, so
# the file can live wherever the tests live.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _compose(config_dir, config_name, overrides=()):
    with initialize_config_dir(
        version_base=None, config_dir=os.path.join(REPO_ROOT, config_dir)
    ):
        return compose(config_name=config_name, overrides=list(overrides))


def _tiny_base_model(dtype=torch.float64, seed=0):
    cfg = _compose("configs/slicegpt", "llama_1B_slice_hyb_entropy", _tiny_overrides())
    torch.manual_seed(seed)
    model = instantiate(cfg.model, _convert_="all").to(dtype).eval()
    # non-trivial norm weights so that norm fusion is actually exercised
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, RMSNorm):
                m.weight.uniform_(0.5, 1.5)
    return model


def _batches(n=4, bsz=2, seq=TINY["sequence_length"], seed=1):
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randint(0, TINY["vocab_size"], (bsz, seq), generator=g) for _ in range(n)
    ]


@torch.no_grad()
def _logits(model, batches):
    return torch.cat([model(ids) for ids in batches])


def test_rmsn_matches_rmsnorm_with_sqrt_weight():
    """V2: SliceGPT's RMSN (divides by original D) == nano RMSNorm(d) with weight sqrt(D/d), up to eps."""
    D, d = 2048, 432
    x = torch.randn(4, 7, d, dtype=torch.float64)
    ref = RMSN(D, eps=1e-5)(x)
    norm = RMSNorm(d, eps=1e-5, elementwise_affine=True, dtype=torch.float64)
    norm.weight.data.fill_(math.sqrt(D / d))
    assert torch.allclose(norm(x), ref, rtol=1e-4, atol=1e-6)


def test_fusion_is_function_preserving():
    model = _tiny_base_model()
    batches = _batches()
    before = _logits(model, batches)
    fuse_norms(model)
    assert torch.allclose(_logits(model, batches), before, atol=1e-10)


def test_center_embeddings_changes_rmsnorm_model():
    """SliceGPT's unconditional embedding mean-centering is NOT function preserving for RMSNorm models."""
    model = _tiny_base_model()
    batches = _batches()
    before = _logits(model, batches)
    fuse_norms(model, center_embeddings=True)
    assert not torch.allclose(_logits(model, batches), before, atol=1e-6)


def test_rotation_without_slicing_is_invariant():
    """V1: fuse + rotate with d = D (no slicing) must give the same function (computational invariance)."""
    for orientation in ("pca", "random"):
        model = _tiny_base_model()
        batches = _batches()
        before = _logits(model, batches)
        fuse_norms(model)
        rotate_and_slice(
            model,
            _batches(seed=5),
            d=TINY["dmodel"],
            orientation=orientation,
            device=torch.device("cpu"),
        )
        finalize_norms(model)
        after = _logits(model, batches)
        assert torch.allclose(after, before, atol=1e-8), (
            orientation,
            (after - before).abs().max(),
        )
        for (
            block
        ) in (
            model.encoder.blocks
        ):  # residual rotations are orthogonal when nothing is sliced
            for res in (block.attention_layer, block.ff_layer):
                Q = res.shortcut_Q
                assert torch.allclose(
                    Q @ Q.T, torch.eye(Q.shape[0], dtype=Q.dtype), atol=1e-8
                )


def test_unsliced_head_mode():
    """`slice_head=False` (SliceGPT CLI default, used for the reference cross-check): invariance at d = D and
    the right shapes at d < D (last block output + head stay at width D)."""
    model = _tiny_base_model()
    batches = _batches()
    before = _logits(model, batches)
    fuse_norms(model)
    rotate_and_slice(
        model,
        _batches(seed=5),
        d=TINY["dmodel"],
        orientation="pca",
        device=torch.device("cpu"),
        slice_head=False,
    )
    finalize_norms(model)
    assert torch.allclose(_logits(model, batches), before, atol=1e-8)

    d, D = 40, TINY["dmodel"]
    model = _tiny_base_model()
    fuse_norms(model)
    rotate_and_slice(
        model,
        _batches(seed=5),
        d=d,
        orientation="pca",
        device=torch.device("cpu"),
        slice_head=False,
    )
    finalize_norms(model)
    last = model.encoder.blocks[-1]
    assert last.ff_layer.layer.ff_post_act.weight.shape[
        0
    ] == D and last.ff_layer.shortcut_Q.shape == (d, D)
    assert model.head.linear.weight.shape[
        1
    ] == D and model.head.norm.normalized_shape == (D,)
    assert model.encoder.blocks[0].ff_layer.shortcut_Q.shape == (d, d)
    _logits(model, batches)  # runs


def _fake_importances(path, n_blocks, D, dff, seed=3):
    g = torch.Generator().manual_seed(seed)
    torch.save(
        {
            "dmodel_importances": torch.rand(D, generator=g),
            "dff_importances": [torch.rand(dff, generator=g) for _ in range(n_blocks)],
        },
        path,
    )


def test_prune_dff_matches_hp_selection():
    model = _tiny_base_model()
    originals = [
        (
            b.ff_layer.layer.ff_pre_act.weight.clone(),
            b.ff_layer.layer.ff_post_act.weight.clone(),
        )
        for b in model.encoder.blocks
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "imp.pt")
        _fake_importances(path, TINY["n_blocks"], TINY["dmodel"], TINY["dff"])
        prune_dff(model, path, target_dff=40)
        imps = torch.load(path)["dff_importances"]
    for block, (up, down), imp in zip(model.encoder.blocks, originals, imps):
        idx = torch.topk(
            imp, dim=0, largest=True, k=40
        ).indices  # same selection as pruning.prune (HP)
        ff = block.ff_layer.layer
        assert torch.equal(ff.ff_pre_act.weight, up[idx, :])
        assert torch.equal(ff.ff_post_act.weight, down[:, idx])
        assert ff.gate.weight.shape == (40, TINY["dmodel"])


def test_sliced_model_saves_and_reloads_into_sliced_config():
    """V4: hybrid slicing (dff pruning + d slicing) -> state dict loads strictly into `llama_3_1_sliced`
    instantiated from the RFT config, and gives identical outputs; param counts agree.
    """
    d, dff = 40, 48
    model = _tiny_base_model(dtype=torch.float32)
    batches = _batches()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "imp.pt")
        _fake_importances(path, TINY["n_blocks"], TINY["dmodel"], TINY["dff"])
        prune_dff(model, path, target_dff=dff)
    fuse_norms(model)
    rotate_and_slice(
        model, _batches(seed=5), d=d, orientation="random", device=torch.device("cpu")
    )
    finalize_norms(model)
    sliced_out = _logits(model, batches)
    state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}

    cfg = _compose(
        "configs/pc_project_mas",
        "llama_1B_slicegpt_hyb_entropy",
        _tiny_overrides(dmodel=d, dff=dff),
    )
    fresh = instantiate(cfg.model, _convert_="all").eval()
    fresh.load_state_dict(state_dict, strict=True)
    assert torch.allclose(_logits(fresh, batches), sliced_out, atol=1e-6)
    assert sum(p.numel() for p in fresh.parameters()) == sum(
        p.numel() for p in model.parameters()
    )
    assert all(
        b.attention_layer.shortcut_Q.shape == (d, d) for b in fresh.encoder.blocks
    )


def _meta_params(config_dir, config_name, overrides):
    cfg = _compose(config_dir, config_name, overrides)
    with torch.device("meta"):
        model = instantiate(cfg.model, _convert_="all")
    return sum(p.numel() for p in model.parameters())


def test_param_counts_match_hp():
    """Hybrid widths (configs) give the same total params as the HP models (within 0.5%)."""
    for size, rft_cfg, hp_dims in (
        (
            "1B",
            "llama_1B_slicegpt_hyb_entropy",
            {"r50": (1344, 5376), "r70": (960, 3840), "r90": (448, 1728)},
        ),
        (
            "8B",
            "llama_8B_slicegpt_hyb_entropy",
            {"r50": (3072, 9216), "r70": (2160, 6480), "r90": (1056, 3168)},
        ),
    ):
        for ratio, (hp_d, hp_dff) in hp_dims.items():
            sg = _meta_params(
                "configs/pc_project_mas", rft_cfg, [f"slicegpt_ratio={ratio}"]
            )
            # the slicing config's model is the plain base Llama -> with HP dims it is exactly the HP model
            hp = _meta_params(
                "configs/slicegpt",
                f"llama_{size}_slice_hyb_entropy",
                [f"common.dmodel={hp_d}", f"common.dff={hp_dff}"],
            )
            assert abs(sg - hp) / hp < 0.005, (size, ratio, sg, hp)


def test_configs_compose():
    for config_dir, name in (
        ("configs/slicegpt", "llama_1B_slice_hyb_entropy"),
        ("configs/slicegpt", "llama_8B_slice_hyb_entropy"),
        ("configs/pc_project_mas", "llama_1B_slicegpt_hyb_entropy"),
        ("configs/pc_project_mas", "llama_1B_slicegpt_hyb_distillation_entropy"),
        ("configs/pc_project_mas", "llama_8B_slicegpt_hyb_entropy"),
        ("configs/pc_project_mas", "llama_8B_slicegpt_hyb_distillation_entropy"),
    ):
        for ratio in ("r50", "r70", "r90"):
            cfg = _compose(config_dir, name, [f"slicegpt_ratio={ratio}"])
            from omegaconf import OmegaConf

            assert not OmegaConf.missing_keys(cfg), (name, OmegaConf.missing_keys(cfg))
            if "slicegpt" in cfg:
                assert cfg.slicegpt.target_d == cfg.slicegpt_dims[ratio].d
            else:
                assert cfg.common.dmodel == cfg.slicegpt_dims[ratio].d
                assert cfg.model.encoder.block_fn.shortcut_dim == cfg.common.dmodel
                assert cfg.trainer.checkpoint.load.type == "sliced_nano"


if __name__ == "__main__":
    import traceback

    tests = [
        v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
