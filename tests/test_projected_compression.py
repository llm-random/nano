"""Tier 1 of the baseline audit: CPU unit tests for projected compression.

See documents/notes/main_baseline_audit.md. Built in the idiom of test_slicegpt.py --
a tiny model composed from the *real* configs with the dimensions overridden -- so the
config wiring is exercised too, on CPU, in seconds, with no data.

The central check is full-rank invariance. init_compression picks the top-k dimensions
by importance and builds one-hot selection matrices, with the auxiliary weight starting
at zero, so Wc = P1 W P2. When k equals the full dimension the selection is a
permutation of the residual stream rather than a reduction, and a permutation applied
consistently across the embedding, both norms, attention and the head is a no-op: the
logits must come out unchanged. If fusion or index bookkeeping is wrong anywhere, this
fails loudly. It is the analogue of the SliceGPT suite's s1_invariance.
"""

import os
import tempfile

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import resolver  # noqa: F401  (registers the repo's custom OmegaConf resolvers)
from src.projected_compression.compression import init_compression
from src.projected_compression.model import ProjectedLinear

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_DMODEL = 48
BASE_DFF = 96
N_BLOCKS = 2
VOCAB = 97
SEQ = 32

COMMON = {
    "base_dmodel": BASE_DMODEL,
    "base_dff": BASE_DFF,
    "dhead": 12,
    "n_blocks": N_BLOCKS,
    "q_heads": 4,
    "kv_heads": 2,
    "sequence_length": SEQ,
}


def _build(target_dmodel, target_dff, seed=0, dtype=torch.float32):
    """A tiny projected model from configs/pc_project/llama_1B_comp.yaml.

    Composed the way run_exp.py does: config_dir is the directory, config_name is bare.

    float32 on CPU because that is what production trains in. The projections now follow
    the model's own device and dtype, so other combinations work too -- see
    test_compression_follows_the_models_device_and_dtype.
    """
    overrides = [f"common.{k}={v}" for k, v in COMMON.items()]
    overrides += [
        f"common.dmodel={target_dmodel}",
        f"common.dff={target_dff}",
        f"model.embedding.num_embeddings={VOCAB}",
    ]
    with initialize_config_dir(
        version_base=None, config_dir=os.path.join(REPO_ROOT, "configs", "pc_project")
    ):
        cfg = compose(config_name="llama_1B_comp", overrides=overrides)
    torch.manual_seed(seed)
    model = instantiate(cfg.model, _convert_="all").to(dtype).eval()
    generator = torch.Generator().manual_seed(seed + 1000)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight"):
                # RMSNorm initialises to ones. A constant vector is invariant under
                # permutation, which would let a norm sliced with the wrong ordering
                # pass the invariance test unnoticed -- so give it real structure.
                param.copy_(
                    torch.rand(param.shape, generator=generator).mul(1.0).add(0.5)
                )
            else:
                # Weights come from torch.rand, so all positive and in [0, 1). Centre
                # them: a sign error or a transposed index cannot then pass unnoticed.
                param.sub_(0.5)
    return model


def _write_importances(path, order, seed=0):
    """Importance tensors whose top-k gives a known ordering.

    "identity" makes the selection come out as 0, 1, 2, ... so the projections are the
    identity matrix; "shuffled" makes it an arbitrary permutation.
    """
    if order == "identity":
        dmodel = torch.arange(BASE_DMODEL).flip(0).float()
        dff = [torch.arange(BASE_DFF).flip(0).float() for _ in range(N_BLOCKS)]
    else:
        generator = torch.Generator().manual_seed(seed)
        dmodel = torch.randperm(BASE_DMODEL, generator=generator).float()
        dff = [
            torch.randperm(BASE_DFF, generator=generator).float()
            for _ in range(N_BLOCKS)
        ]
    torch.save({"dmodel_importances": dmodel, "dff_importances": dff}, path)


def _compress(model, target_dmodel, target_dff, order, seed=0):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "importances.pt")
        _write_importances(path, order, seed=seed)
        return init_compression(model, path, target_dmodel, target_dff)


def _batch(seed=1, bsz=2):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (bsz, SEQ), generator=generator)


@torch.no_grad()
def _logits(model, ids):
    return model(ids)


# --------------------------------------------------------------------------------------


def test_full_rank_identity_projection_is_the_identity_matrix():
    """At full rank with importances in order, every selection matrix is I and the
    auxiliary weight starts at zero, so Wc = W exactly."""
    model = _build(BASE_DMODEL, BASE_DFF)
    _compress(model, BASE_DMODEL, BASE_DFF, "identity")

    checked = 0
    for name, module in model.named_modules():
        if not isinstance(module, ProjectedLinear):
            continue
        if module.projection_in_weight is not None:
            eye = torch.eye(
                module.base_in_features, dtype=module.projection_in_weight.dtype
            )
            assert torch.equal(
                module.projection_in_weight.detach(), eye
            ), f"{name}: projection_in is not the identity at full rank"
        if module.projection_out_weight is not None:
            eye = torch.eye(
                module.base_out_features, dtype=module.projection_out_weight.dtype
            )
            assert torch.equal(
                module.projection_out_weight.detach(), eye
            ), f"{name}: projection_out is not the identity at full rank"
        assert not module.auxiliary_weight.detach().any(), f"{name}: aux weight nonzero"
        checked += 1
    assert checked > 0, "no ProjectedLinear layers were found -- the model is wrong"


def test_full_rank_permutation_leaves_the_logits_unchanged():
    """The invariance check.

    Two models with identical base weights, compressed at full rank with two different
    dimension orderings. A permutation of the residual stream applied consistently
    everywhere is a no-op, so the logits must agree.
    """
    ids = _batch()

    identity = _build(BASE_DMODEL, BASE_DFF, seed=0)
    _compress(identity, BASE_DMODEL, BASE_DFF, "identity")
    reference = _logits(identity.double(), ids)

    shuffled = _build(BASE_DMODEL, BASE_DFF, seed=0)
    _compress(shuffled, BASE_DMODEL, BASE_DFF, "shuffled", seed=7)
    permuted = _logits(shuffled.double(), ids)

    assert reference.shape == (ids.shape[0], SEQ, VOCAB)
    # Compared in float64. A permutation reorders the summations inside every matmul,
    # so this is not bit-identical even when it is exactly right; in float32 that noise
    # reaches ~2e-5, which is too loose to catch a subtly wrong index. Converting after
    # init_compression (which can only build float32 projections) buys three orders of
    # magnitude of headroom and makes the check meaningful.
    torch.testing.assert_close(permuted, reference, rtol=1e-11, atol=1e-11)


def test_compressed_model_runs_and_has_the_right_shape():
    """A genuinely reduced model still produces finite logits of the expected shape."""
    model = _build(BASE_DMODEL // 2, BASE_DFF // 2)
    _compress(model, BASE_DMODEL // 2, BASE_DFF // 2, "shuffled")
    out = _logits(model, _batch())
    assert out.shape == (2, SEQ, VOCAB)
    assert torch.isfinite(out).all()


def test_init_compression_trains_only_projections_norms_and_auxiliary():
    """Pin down exactly what compression leaves trainable.

    init_compression freezes every parameter, then replaces the three RMSNorm weights
    with freshly sliced nn.Parameters -- which default to requires_grad=True. That is
    deliberate: a sliced norm belongs to the compressed model and cannot be expressed as
    a projection of the base norm, so it has to be learned. Everything else that is
    trainable must be a projection or the auxiliary term; no base weight may be.

    This is the invariant that makes a run projected compression rather than an ordinary
    fine-tune, so it is worth stating rather than leaving implicit.
    """
    model = _build(BASE_DMODEL // 2, BASE_DFF // 2)
    _compress(model, BASE_DMODEL // 2, BASE_DFF // 2, "identity")

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable, "nothing is trainable -- compression did not initialise"

    def allowed(name):
        return (
            "projection" in name
            or "auxiliary_weight" in name
            or name.endswith("norm.weight")
        )

    unexpected = sorted(n for n in trainable if not allowed(n))
    assert not unexpected, f"base weights left trainable: {unexpected}"

    # and the base weights really are frozen
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert any(
        n.endswith(".weight") and "projection" not in n and "auxiliary" not in n
        for n in frozen
    ), "no base weight was frozen"

    # every norm is trainable, not just some
    norms = {n for n in model.state_dict() if n.endswith("norm.weight")}
    assert norms <= trainable, f"some norms are frozen: {sorted(norms - trainable)}"


def test_gradients_reach_the_projections_and_not_the_base_weights():
    """One backward pass: projections get gradient, frozen base weights get none."""
    model = _build(BASE_DMODEL // 2, BASE_DFF // 2)
    _compress(model, BASE_DMODEL // 2, BASE_DFF // 2, "identity")

    out = model(_batch())
    out.square().mean().backward()

    got_grad = [n for n, p in model.named_parameters() if p.grad is not None]
    assert got_grad, "no parameter received a gradient"
    for name, param in model.named_parameters():
        if not param.requires_grad:
            assert param.grad is None, f"{name}: frozen weight received a gradient"


def test_importance_scores_have_the_shapes_the_method_assumes():
    """dmodel importances are one global vector; dff importances are per layer.

    The residual stream is shared across blocks, so a dimension dropped from it must be
    dropped everywhere -- one ranking for the whole model. Each block's feed-forward is
    independent, so dff is ranked per block. If these ever came back the other way round
    the compression would still run and still produce a model; it would just be
    compressing the wrong thing.
    """
    from src.projected_compression.compression import calculate_dimension_importances

    model = _build(BASE_DMODEL // 2, BASE_DFF // 2)
    topk_dmodel, topk_dff = BASE_DMODEL // 2, BASE_DFF // 2
    dmodel_indices, dff_indices = calculate_dimension_importances(
        model, topk_dmodel, topk_dff
    )

    assert dmodel_indices.shape == (topk_dmodel,), (
        "dmodel importances must be a single global ranking, got "
        f"{tuple(dmodel_indices.shape)}"
    )
    assert dff_indices.shape == (
        N_BLOCKS,
        topk_dff,
    ), f"dff importances must be one ranking per block, got {tuple(dff_indices.shape)}"

    # indices must be valid, unique and sorted -- the projections index with them
    assert dmodel_indices.min() >= 0 and dmodel_indices.max() < BASE_DMODEL
    assert len(set(dmodel_indices.tolist())) == topk_dmodel, "duplicate dmodel indices"
    assert torch.equal(dmodel_indices, dmodel_indices.sort().values), "not sorted"
    for block, row in enumerate(dff_indices):
        assert row.min() >= 0 and row.max() < BASE_DFF
        assert (
            len(set(row.tolist())) == topk_dff
        ), f"duplicate dff indices in block {block}"


def test_compression_follows_the_models_device_and_dtype():
    """The projections must be built where the model already is.

    They used to be created with no device or dtype -- always float32 on CPU -- while
    init_compression put the top-k indices on whatever get_device() returned. On any
    machine with a GPU that is cuda, so this path died with "indices should be either on
    cpu or on the same device as the indexed tensor". It is why the whole
    llama_1B_comp family could not run on a GPU, and it went unnoticed because the
    paper runs all use the mem-eff path instead.

    float64 stands in for "a dtype other than the default" so this is checkable without
    a GPU; the device half is asserted alongside it and is what actually broke.
    """
    model = _build(BASE_DMODEL // 2, BASE_DFF // 2, dtype=torch.float64)
    _compress(model, BASE_DMODEL // 2, BASE_DFF // 2, "identity")

    reference = next(model.parameters())
    checked = 0
    for name, module in model.named_modules():
        if not isinstance(module, ProjectedLinear):
            continue
        for attribute in (
            "projection_in_weight",
            "projection_out_weight",
            "auxiliary_weight",
        ):
            parameter = getattr(module, attribute)
            if parameter is None:
                continue
            assert (
                parameter.dtype == reference.dtype
            ), f"{name}.{attribute} is {parameter.dtype}, model is {reference.dtype}"
            assert (
                parameter.device == reference.device
            ), f"{name}.{attribute} is on {parameter.device}, model is on {reference.device}"
            checked += 1
    assert checked > 0, "no projections were checked"

    # and it still runs
    out = _logits(model, _batch())
    assert out.dtype == torch.float64 and torch.isfinite(out).all()
