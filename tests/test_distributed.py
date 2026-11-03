"""FSDP2 and sharded checkpoints, on whatever hardware is present.

The framework's own distributed entry points are hard-gated on
torch.cuda.is_available() and setup_fsdp2_model builds its mesh with
init_device_mesh("cuda", ...), so none of them run here. But FSDP2 itself does:
fully_shard works on a CPU mesh with the gloo backend and produces real DTensor
parameters with a working full_tensor().

That covers the sharded checkpoint path, which goes through
torch.distributed.checkpoint rather than torch.save and otherwise has no test
anywhere.

The mesh follows the hardware: CPU with gloo on a workstation, CUDA with NCCL on a
cluster node, where these then exercise the real production configuration. Forcing a CPU
mesh where CUDA is present is not merely redundant, it breaks -- FSDP2's collectives pick
a CUDA stream and fail with "'Stream' object has no attribute 'record_event'". That was
found by running this file on helios.

What a single rank still cannot cover: more than one rank, real sharding across devices,
and the bfloat16 mixed-precision policy production uses.
"""

import os

import pytest
import torch
import torch.distributed as dist

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _mesh_device():
    """Shard on the device this machine actually has.

    On a workstation that is CPU with gloo, which is the point of this file. On a cluster
    node it is CUDA with NCCL, so the same tests then exercise the real production
    configuration rather than a CPU imitation of it -- and forcing a CPU mesh where CUDA
    is present does not work anyway: dcp save/load fails.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def cpu_process_group():
    """A single-rank process group, so fully_shard has a mesh to shard over."""
    from conftest import free_port

    already = dist.is_initialized()
    if not already:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = str(free_port())
        backend = "nccl" if _mesh_device() == "cuda" else "gloo"
        if backend == "nccl":
            torch.cuda.set_device(0)
        dist.init_process_group(backend=backend, rank=0, world_size=1)
    for key, value in dict(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0").items():
        os.environ.setdefault(key, value)
    os.environ["RANK"] = "0"
    yield
    if not already:
        dist.destroy_process_group()


def _shard(model):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    device = _mesh_device()
    model = model.to(device)
    mesh = init_device_mesh(device, (dist.get_world_size(),))
    fully_shard(model, mesh=mesh)
    return model


# --------------------------------------------------------------------------------------


def test_fully_shard_gives_dtensor_parameters(cpu_process_group):
    """The precondition everything else here relies on."""
    model = _shard(torch.nn.Sequential(torch.nn.Linear(8, 8)))
    param = next(iter(model.parameters()))
    assert hasattr(param, "full_tensor"), "parameters are not DTensors after sharding"
    assert tuple(param.full_tensor().shape) == (8, 8)


def test_sharded_checkpoint_round_trips(cpu_process_group, tmp_path):
    """The dcp save/load path, which plain torch.save checkpoints never exercise."""
    import torch.distributed.checkpoint as dcp

    from src.core.checkpointing import TrainingState

    torch.manual_seed(0)
    model = _shard(torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    model(torch.randn(4, 8, device=_mesh_device())).sum().backward()
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


def test_the_weight_decay_grouping_survives_sharding(cpu_process_group):
    """`parameter_groups` runs *after* `setup_distributed_training`, so it sees
    DTensors rather than plain tensors, and it decides membership with an
    `isinstance(p, nn.Parameter)` check. If fully_shard dropped that class the
    check would quietly match nothing, every AME scale would land back in the
    decayed group, and the only symptom would be alpha_2 drifting to zero on a
    multi-GPU run -- which is the exact signal the method is trying to measure.
    """
    import torch.nn as nn

    from src.core.ame import AMEAttention
    from src.core.model import parameter_groups

    torch.manual_seed(0)
    layer = AMEAttention(32, 4, 4, 16, vocab_size=64, dhead=16, backend="reference")
    expected = {
        name
        for name, _ in layer.named_parameters()
        if name in set(AMEAttention.no_weight_decay_parameters)
    }
    assert expected, "fixture declares no scales"

    layer = _shard(layer)
    assert all(isinstance(p, nn.Parameter) for p in layer.parameters())

    groups = parameter_groups(layer, 0.1)
    assert [group["weight_decay"] for group in groups] == [0.1, 0.0]
    undecayed = {id(p) for p in groups[1]["params"]}
    assert {n for n, p in layer.named_parameters() if id(p) in undecayed} == expected

    # And the exemption has to hold through a real step, not just at grouping.
    optimizer = torch.optim.AdamW(groups, lr=2**-6)
    before = layer.tau.detach().clone(), layer.alpha.detach().clone()
    for parameter in layer.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    assert (layer.tau - before[0]).abs().max().item() == 0.0
    assert (layer.alpha - before[1]).abs().max().item() == 0.0
