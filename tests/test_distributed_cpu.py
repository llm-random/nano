"""Tier 3/5 of the baseline audit, as far as it can be taken without a GPU.

See documents/notes/main_baseline_audit.md. The framework's own distributed entry points
are hard-gated on torch.cuda.is_available() and setup_fsdp2_model builds its mesh with
init_device_mesh("cuda", ...), so none of them run here. But FSDP2 itself does:
fully_shard works on a CPU mesh with the gloo backend and produces real DTensor
parameters with a working full_tensor().

That is enough to cover the two things which otherwise have no test anywhere:

* `pc_finalize`, whose ProjectedLinear.finalize() calls .full_tensor() and reads
  os.environ["RANK"], so it is unreachable outside FSDP2 -- the audit first recorded it
  as untestable without a cluster, which turned out to be wrong.
* the sharded checkpoint path, which goes through torch.distributed.checkpoint rather
  than torch.save.

What this does NOT cover, and a cluster still must: more than one rank, NCCL, the
bfloat16 mixed-precision policy production uses, and any CUDA-specific behaviour.
"""

import os

import pytest
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import resolver  # noqa: F401
from src.projected_compression.model import ProjectedLinear
from src.projected_compression.compression import (
    finalize_projection_weights,
    init_compression,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_DMODEL, BASE_DFF, N_BLOCKS, VOCAB, SEQ = 32, 64, 2, 61, 16


@pytest.fixture(scope="module")
def cpu_process_group():
    """A single-rank gloo group, so fully_shard has a mesh to shard over."""
    already = dist.is_initialized()
    if not already:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29591")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    for key, value in dict(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0").items():
        os.environ.setdefault(key, value)
    os.environ["RANK"] = "0"
    yield
    if not already:
        dist.destroy_process_group()


def _shard(model):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    mesh = init_device_mesh("cpu", (dist.get_world_size(),))
    fully_shard(model, mesh=mesh)
    return model


def _pc_model(target_dmodel, target_dff, seed=0):
    overrides = [
        f"common.base_dmodel={BASE_DMODEL}",
        f"common.base_dff={BASE_DFF}",
        "common.dhead=8",
        f"common.n_blocks={N_BLOCKS}",
        "common.q_heads=4",
        "common.kv_heads=2",
        f"common.sequence_length={SEQ}",
        f"common.dmodel={target_dmodel}",
        f"common.dff={target_dff}",
        f"model.embedding.num_embeddings={VOCAB}",
    ]
    with initialize_config_dir(
        version_base=None, config_dir=os.path.join(REPO_ROOT, "configs", "pc_project")
    ):
        cfg = compose(config_name="llama_1B_comp", overrides=overrides)
    torch.manual_seed(seed)
    model = instantiate(cfg.model, _convert_="all").eval()
    generator = torch.Generator().manual_seed(seed + 7)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight"):
                param.copy_(torch.rand(param.shape, generator=generator).add(0.5))
            else:
                param.sub_(0.5)
    return model


def _compress(model, target_dmodel, target_dff, tmp_path):
    path = str(tmp_path / "importances.pt")
    generator = torch.Generator().manual_seed(3)
    torch.save(
        {
            "dmodel_importances": torch.randperm(
                BASE_DMODEL, generator=generator
            ).float(),
            "dff_importances": [
                torch.randperm(BASE_DFF, generator=generator).float()
                for _ in range(N_BLOCKS)
            ],
        },
        path,
    )
    return init_compression(model, path, target_dmodel, target_dff)


def _batch(seed=1):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (2, SEQ), generator=generator)


# --------------------------------------------------------------------------------------


def test_fully_shard_gives_dtensor_parameters(cpu_process_group):
    """The precondition everything else here relies on."""
    model = _shard(torch.nn.Sequential(torch.nn.Linear(8, 8)))
    param = next(iter(model.parameters()))
    assert hasattr(param, "full_tensor"), "parameters are not DTensors after sharding"
    assert tuple(param.full_tensor().shape) == (8, 8)


def test_pc_finalize_merges_the_projections_exactly(cpu_process_group, tmp_path):
    """Finalization must fold Wc = P_out W P_in + auxiliary into one weight, exactly.

    This is the whole point of the method: train projections, then merge them away into
    a smaller standard Transformer. If the merge is wrong, every finalized checkpoint is
    wrong, and nothing else in the suite can catch it -- finalize() calls .full_tensor()
    and reads os.environ["RANK"], so it is unreachable unless the parameters are
    DTensors.

    The comparison is on the weights rather than on a forward pass, because finalize()
    replaces the FSDP-managed parameters outright (and sets them to None off rank 0), so
    the module cannot be run afterwards. That is fine in production, where
    save_pc_finalized_checkpoint() finalizes, dumps the state dict and stops -- but it
    does mean nobody should expect to keep training a finalized model.
    """
    target_dmodel, target_dff = BASE_DMODEL // 2, BASE_DFF // 2
    model = _pc_model(target_dmodel, target_dff)
    _compress(model, target_dmodel, target_dff, tmp_path)

    # Stand in for a trained model. Straight after init_compression the projections are
    # one-hot and the auxiliary term is all zeros, so merging it in is a no-op and the
    # test cannot tell whether finalize() uses it at all -- dropping the auxiliary term
    # entirely still passed. After training both are arbitrary, which is the case that
    # has to be right.
    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, ProjectedLinear):
                continue
            for attribute in (
                "projection_in_weight",
                "projection_out_weight",
                "auxiliary_weight",
            ):
                parameter = getattr(module, attribute)
                if parameter is not None:
                    parameter.add_(
                        torch.rand(parameter.shape, generator=generator).mul(0.1)
                    )

    _shard(model)

    # what each ProjectedLinear should collapse to, computed before finalizing
    expected = {}
    for name, module in model.named_modules():
        if not isinstance(module, ProjectedLinear):
            continue
        weight = module.weight.full_tensor().float()
        if module.result_in_features is not None:
            weight = weight @ module.projection_in_weight.full_tensor().float()
        if module.result_out_features is not None:
            weight = module.projection_out_weight.full_tensor().float() @ weight
        if (
            module.result_in_features is not None
            or module.result_out_features is not None
        ):
            weight = weight + module.auxiliary_weight.full_tensor().float()
        expected[name] = weight

    assert expected, "no ProjectedLinear layers found"

    with torch.no_grad():
        finalize_projection_weights(model)

    checked = 0
    for name, module in model.named_modules():
        if name not in expected:
            continue
        assert module.weight is not None, f"{name}: weight dropped on rank 0"
        torch.testing.assert_close(
            module.weight.detach().float(), expected[name], rtol=1e-5, atol=1e-5
        )
        checked += 1
    assert checked == len(expected), f"only {checked}/{len(expected)} layers checked"


def test_pc_finalize_leaves_a_smaller_standard_model(cpu_process_group, tmp_path):
    """The point of finalizing: the projections are gone and the weights are compressed."""
    target_dmodel, target_dff = BASE_DMODEL // 2, BASE_DFF // 2
    model = _pc_model(target_dmodel, target_dff)
    _compress(model, target_dmodel, target_dff, tmp_path)
    _shard(model)
    with torch.no_grad():
        finalize_projection_weights(model)

    leftovers = [
        name
        for name, _ in model.named_parameters()
        if "projection" in name or "auxiliary" in name
    ]
    assert not leftovers, f"projections survived finalization: {leftovers}"

    block = model.encoder.blocks[0]
    gate = block.ff_layer.layer.gate.weight
    assert tuple(gate.shape) == (
        target_dff,
        target_dmodel,
    ), f"gate was not compressed: {tuple(gate.shape)}"


def test_sharded_checkpoint_round_trips(cpu_process_group, tmp_path):
    """The dcp save/load path, which plain torch.save checkpoints never exercise."""
    import torch.distributed.checkpoint as dcp

    from src.core.checkpointing import TrainingState

    torch.manual_seed(0)
    model = _shard(torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    model(torch.randn(4, 8)).sum().backward()
    optimizer.step()
    saved = next(iter(model.parameters())).full_tensor().clone()

    folder = str(tmp_path / "sharded")
    dcp.save({"app": TrainingState(model, optimizer, scheduler)}, checkpoint_id=folder)
    assert os.path.isdir(folder) and os.listdir(folder), "nothing was written"

    torch.manual_seed(99)
    fresh = _shard(torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)))
    fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    fresh_scheduler = torch.optim.lr_scheduler.ConstantLR(fresh_optimizer, factor=1.0)
    assert not torch.equal(
        next(iter(fresh.parameters())).full_tensor(), saved
    ), "the fresh model already matches; the test would prove nothing"

    dcp.load(
        state_dict={"app": TrainingState(fresh, fresh_optimizer, fresh_scheduler)},
        checkpoint_id=folder,
    )
    torch.testing.assert_close(
        next(iter(fresh.parameters())).full_tensor(), saved, rtol=0, atol=0
    )
