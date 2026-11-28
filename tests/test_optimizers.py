"""src/core/optimizers.py: parameter roles, param groups and their checkpoints."""

import copy
import os
import warnings

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import distribute_tensor

import resolver  # noqa: F401  (registers the repo's custom OmegaConf resolvers)
from conftest import free_port
from src.core.attention import build_attention
from src.core.checkpointing import TrainingState, load_checkpoint_from_file
from src.core.model import (
    RoPEAttention,
    TransformerBlock,
    TransformerEmbedding,
    TransformerHead,
)
from src.core.moe import MoE
from src.core.optimizers import (
    ROLES,
    MuonWithAdamW,
    add_qk_clip,
    build_optimizer_and_scheduler,
    newton_schulz,
    param_roles,
    qk_clip_,
)
from src.core.schedulers import get_cosine_scheduler_with_warmup
from test_distributed import _shard, cpu_process_group  # noqa: F401  (fixture)

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
MODEL_FILE = "model.pt"


def _cfg(optimizer, overrides=()):
    warnings.filterwarnings("ignore")
    with initialize_config_dir(version_base=None, config_dir=CONFIGS_DIR):
        return compose(
            "tiny_local", [f"_trainer/_optimizer@optimizer={optimizer}", *overrides]
        )


def _build(optimizer, seed=0, sharded=False, overrides=()):
    """The h4 tiny_local model, optimizer and scheduler, as main.py builds them."""
    cfg = _cfg(optimizer, overrides)
    torch.manual_seed(seed)
    model = instantiate(cfg.model, _convert_="all")
    if sharded:
        model = _shard(model)
    return (model, *build_optimizer_and_scheduler(cfg, model, 1e-3))


def _step(model, optimizer, scheduler):
    torch.manual_seed(1)
    tokens = torch.randint(0, 50304, (2, 16), device=next(model.parameters()).device)
    model(tokens).float().square().mean().backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()


def _save(folder, model, optimizer, scheduler, sharded):
    """As Trainer.save_checkpoint writes a nano checkpoint."""
    if sharded:
        dcp.save(
            {"app": TrainingState(model, optimizer, scheduler)}, checkpoint_id=folder
        )
    else:
        state = {"model": model.state_dict(), "optim": optimizer.state_dict()}
        torch.save(
            {**state, "scheduler": scheduler.state_dict()}, f"{folder}/{MODEL_FILE}"
        )


def _load(folder, model, optimizer, scheduler, only_weights):
    load_config = OmegaConf.create(
        {
            "path": folder,
            "model_checkpoint_filename": MODEL_FILE,
            "only_weights": only_weights,
        }
    )
    load_checkpoint_from_file(load_config, model, optimizer, scheduler)


def _params(model):
    return [
        p.full_tensor() if hasattr(p, "full_tensor") else p for p in model.parameters()
    ]


def _hyperparameters(optimizer):
    return [
        {k: v for k, v in g.items() if k != "params"} for g in optimizer.param_groups
    ]


def test_every_h4_parameter_gets_exactly_one_role():
    model, _, _ = _build("adamw")
    roles = param_roles(model)
    assert {r: len(ps) for r, ps in roles.items()} == {
        # 2 RMSNorm and 2 QK-norm gains per block, plus the head's RMSNorm
        "vector": 17,
        "embedding": 1,
        "head": 1,
        "hidden": 28,
    }
    ids = [id(p) for ps in roles.values() for p in ps]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    assert set(ids) == {id(p) for p in model.parameters()}
    assert roles["embedding"][0] is model.embedding.weight
    assert roles["head"][0] is model.head.linear.weight


def test_adamw_is_one_group():
    _, optimizer, _ = _build("adamw")
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["weight_decay"] == 0.1


def test_adamw_groups_has_one_group_per_nonempty_role():
    _, optimizer, _ = _build("adamw_groups")
    assert [(g["role"], g["weight_decay"]) for g in optimizer.param_groups] == [
        ("vector", 0.0),
        ("embedding", 0.0),
        ("head", 0.1),
        ("hidden", 0.1),
    ]
    no_head = torch.nn.Sequential(
        torch.nn.Embedding(8, 4), torch.nn.Linear(4, 4, bias=False)
    )
    optimizer, _ = build_optimizer_and_scheduler(_cfg("adamw_groups"), no_head, 1e-3)
    assert [g["role"] for g in optimizer.param_groups] == ["embedding", "hidden"]


def _two_groups(second_lr):
    a, b = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    return torch.optim.AdamW(
        [{"params": a.parameters()}, {"params": b.parameters(), "lr": second_lr}],
        lr=1e-3,
    )


def _lrs(optimizer, scheduler, n_steps):
    """Each group's LR at every step, then after the last one."""
    lrs = []
    for _ in range(n_steps):
        lrs.append([g["lr"] for g in optimizer.param_groups])
        optimizer.step()
        scheduler.step()
    return lrs + [[g["lr"] for g in optimizer.param_groups]]


def _stock_cosine(optimizer, warmup_steps, n_steps, final_lr_fraction):
    """The cosine schedule before per-group eta_min: torch's CosineAnnealingLR."""
    s = torch.optim.lr_scheduler
    eta_min = final_lr_fraction * optimizer.param_groups[0]["lr"]
    return s.SequentialLR(
        optimizer,
        schedulers=[
            s.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps),
            s.ConstantLR(optimizer, factor=1.0),
            s.CosineAnnealingLR(
                optimizer, T_max=n_steps - warmup_steps - 1, eta_min=eta_min
            ),
        ],
        milestones=[warmup_steps, warmup_steps + 1],
    )


def test_cosine_with_one_lr_is_bit_identical_to_torchs():
    ours, stock = _two_groups(1e-3), _two_groups(1e-3)
    scheduler = get_cosine_scheduler_with_warmup(
        ours, warmup_steps=3, n_steps=20, final_lr_fraction=0.1
    )
    reference = _stock_cosine(stock, warmup_steps=3, n_steps=20, final_lr_fraction=0.1)
    assert _lrs(ours, scheduler, 20) == _lrs(stock, reference, 20)


def test_cosine_decays_each_group_to_its_own_final_fraction():
    optimizer = _two_groups(1e-2)
    scheduler = get_cosine_scheduler_with_warmup(
        optimizer, warmup_steps=3, n_steps=20, final_lr_fraction=0.1
    )
    lrs = _lrs(optimizer, scheduler, 20)
    for first, second in lrs:
        assert second == pytest.approx(10 * first)
    assert lrs[-1] == pytest.approx([1e-4, 1e-3])


def _check_resume(folder, sharded, option="adamw_groups", overrides=()):
    model, optimizer, scheduler = _build(option, sharded=sharded, overrides=overrides)
    _step(model, optimizer, scheduler)
    _save(folder, model, optimizer, scheduler, sharded)
    resumed = _build(option, seed=99, sharded=sharded, overrides=overrides)
    _load(folder, *resumed, only_weights=False)
    # Same next step from both: model, AdamW moments, group decays and LR all came back.
    _step(model, optimizer, scheduler)
    _step(*resumed)
    for a, b in zip(_params(model), _params(resumed[0])):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert _hyperparameters(resumed[1]) == _hyperparameters(optimizer)


def test_adamw_groups_checkpoint_resumes(tmp_path):
    _check_resume(str(tmp_path), sharded=False)


def test_adamw_groups_sharded_checkpoint_resumes(cpu_process_group, tmp_path):
    _check_resume(str(tmp_path), sharded=True)


def _check_only_weights(folder, sharded):
    model, optimizer, scheduler = _build("adamw", sharded=sharded)
    _step(model, optimizer, scheduler)
    _save(folder, model, optimizer, scheduler, sharded)
    grouped = _build("adamw_groups", seed=99, sharded=sharded)
    _load(folder, *grouped, only_weights=True)
    for a, b in zip(_params(model), _params(grouped[0])):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not grouped[1].state, "only_weights must start a fresh optimizer"
    assert [g["role"] for g in grouped[1].param_groups] == list(ROLES)
    _step(*grouped)


def test_only_weights_loads_an_adamw_checkpoint_into_adamw_groups(tmp_path):
    _check_only_weights(str(tmp_path), sharded=False)


def test_only_weights_loads_a_sharded_adamw_checkpoint_into_adamw_groups(
    cpu_process_group, tmp_path
):
    _check_only_weights(str(tmp_path), sharded=True)


def _width(h):
    """Width h at depth 1."""
    return [f"common.h={h}", "common.n_blocks=1"]


@pytest.mark.parametrize("h, m", [(4, 1), (8, 2)])
def test_adamw_simplep_scales_only_the_hidden_and_head_lr(h, m):
    _, optimizer, _ = _build("adamw_simplep", overrides=_width(h))
    groups = [
        (g["role"], g["initial_lr"], g["weight_decay"]) for g in optimizer.param_groups
    ]
    assert groups == [
        ("vector", 1e-3, 0.0),
        ("embedding", 1e-3, 0.0),
        ("head", pytest.approx(1e-3 / m), 0.1),
        ("hidden", pytest.approx(1e-3 / m), 0.1),
    ]


def test_adamw_simplep_trains_every_moe_expert_and_the_router_as_hidden():
    moe = ["+_ff_layer@model.encoder.block_fn.ff_layer_fn=moe"]
    model, optimizer, _ = _build("adamw_simplep", overrides=_width(8) + moe)
    hidden = next(g for g in optimizer.param_groups if g["role"] == "hidden")
    in_hidden = {id(p) for p in hidden["params"]}
    moe_params = [
        p for m in model.modules() if isinstance(m, MoE) for p in m.parameters()
    ]
    assert moe_params and all(id(p) in in_hidden for p in moe_params)
    assert hidden["initial_lr"] == pytest.approx(1e-3 / 2)


def test_adamw_simplep_checkpoint_resumes(tmp_path):
    # Base h2 makes the h4 group LRs differ, so WSD's replay must restore each one.
    overrides = [
        "_trainer/_scheduler@trainer.scheduler=wsd",
        "optimizer.base_dmodel=128",
    ]
    _check_resume(
        str(tmp_path), sharded=False, option="adamw_simplep", overrides=overrides
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 32, 128), (1, 128, 32), (3, 32, 128)])
def test_newton_schulz_gives_singular_values_near_one(shape, dtype):
    torch.manual_seed(0)
    G = torch.randn(shape)
    orthogonal = newton_schulz(G.to(dtype), 5).float()
    singular_values = torch.linalg.svdvals(orthogonal)
    # Keller's coefficients trade exactness for speed: about [0.68, 1.15] here.
    assert singular_values.min() > 0.6 and singular_values.max() < 1.2
    U, _, Vh = torch.linalg.svd(G, full_matrices=False)
    cosine = torch.nn.functional.cosine_similarity(
        orthogonal.flatten(-2), (U @ Vh).flatten(-2), dim=-1
    )
    assert (cosine > 0.98).all()


def test_muon_puts_the_hidden_group_in_muon_and_the_rest_in_adamw():
    _, optimizer, _ = _build("muon")
    assert isinstance(optimizer, MuonWithAdamW)
    groups = [
        (g["role"], g["use_muon"], g["initial_lr"], g["weight_decay"])
        for g in optimizer.param_groups
    ]
    # One LR for all: Moonlight's scale lets Muon take AdamW's LR and decay.
    assert groups == [
        ("vector", False, 1e-3, 0.0),
        ("embedding", False, 1e-3, 0.0),
        ("head", False, 1e-3, 0.1),
        ("hidden", True, 1e-3, 0.1),
    ]


def test_muon_trains_every_moe_expert_and_the_router():
    moe = ["+_ff_layer@model.encoder.block_fn.ff_layer_fn=moe"]
    model, optimizer, _ = _build("muon", overrides=_width(4) + moe)
    muon = {id(p) for g in optimizer.param_groups if g["use_muon"] for p in g["params"]}
    moe_params = [
        p for m in model.modules() if isinstance(m, MoE) for p in m.parameters()
    ]
    assert moe_params and all(id(p) in muon for p in moe_params)


def test_muon_rejects_simplep():
    with pytest.raises(AssertionError, match="base_dmodel"):
        _build("muon", overrides=["optimizer.base_dmodel=128"])


def test_muon_with_only_adamw_groups_matches_torchs_adamw():
    reference = _build("adamw_groups")
    ours = _build("muon")
    for group in ours[1].param_groups:
        group["use_muon"] = False
    for _ in range(3):
        _step(*reference)
        _step(*ours)
    for a, b in zip(reference[0].parameters(), ours[0].parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_muon_update_is_orthogonal_at_adamw_rms():
    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(16, 48))
    start = weight.detach().clone()
    weight.grad = torch.randn(16, 48)
    lr, decay = 0.01, 0.1
    optimizer = MuonWithAdamW(
        [{"params": [weight], "use_muon": True}],
        lr=lr,
        weight_decay=decay,
        momentum=0.95,
        ns_steps=5,
        fused=False,
    )
    optimizer.step()
    update = start * (1 - lr * decay) - weight.detach()
    # From zero momentum: buffer (1 - 0.95) * grad, Nesterov step grad.lerp(buffer, 0.95).
    grad = weight.grad
    nesterov = grad.lerp(torch.zeros_like(grad).lerp(grad, 0.05), 0.95)
    expected = lr * 0.2 * 48**0.5 * newton_schulz(nesterov[None], 5)[0].float()
    torch.testing.assert_close(update, expected)
    # Moonlight's RMS match: about 0.2 * lr, like AdamW's update.
    assert 0.15 * lr < update.square().mean().sqrt() < 0.22 * lr


def test_muon_checkpoint_resumes(tmp_path):
    _check_resume(str(tmp_path), sharded=False, option="muon")


def test_muon_sharded_checkpoint_resumes(cpu_process_group, tmp_path):
    _check_resume(str(tmp_path), sharded=True, option="muon")


def _random_grads(model, sharded, seed):
    """The same full gradients on every rank; sharded params get their own rows."""
    torch.manual_seed(seed)
    for p in model.parameters():
        grad = torch.randn(p.shape)
        p.grad = (
            distribute_tensor(grad, p.device_mesh, p.placements) if sharded else grad
        )


def _assert_same(sharded, reference):
    for p, q in zip(sharded.parameters(), reference.parameters()):
        torch.testing.assert_close(p.full_tensor(), q, rtol=0, atol=0)


def _two_rank_muon(rank, port, folder):
    os.environ.update(MASTER_ADDR="localhost", MASTER_PORT=str(port), RANK=str(rank))
    dist.init_process_group("gloo", rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2,))
        cfg = _cfg("muon")

        def build(seed):
            """h4 with an odd-row matrix and a 1-row one: shards of 2 and 1, 1 and 0 rows."""
            torch.manual_seed(seed)
            model = instantiate(cfg.model, _convert_="all")
            model.odd = torch.nn.Sequential(
                torch.nn.Linear(4, 3, bias=False), torch.nn.Linear(3, 1, bias=False)
            )
            return model

        def shard(model):
            # As setup_fsdp2_model, on a CPU mesh.
            for m in model.modules():
                if isinstance(
                    m, (TransformerEmbedding, TransformerBlock, TransformerHead)
                ):
                    fully_shard(m, mesh=mesh)
            return fully_shard(model, mesh=mesh)

        reference = build(0)
        sharded = shard(copy.deepcopy(reference))
        runs = [
            (model, *build_optimizer_and_scheduler(cfg, model, 1e-2), is_sharded)
            for model, is_sharded in ((reference, False), (sharded, True))
        ]
        for seed in range(3):
            for model, optimizer, scheduler, is_sharded in runs:
                _random_grads(model, is_sharded, seed)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        _assert_same(sharded, reference)

        # DCP: a fresh sharded run loads the momentum and AdamW state and steps the same.
        _, optimizer, scheduler, _ = runs[1]
        dcp.save(
            {"app": TrainingState(sharded, optimizer, scheduler)}, checkpoint_id=folder
        )
        resumed = shard(build(99))
        resumed_optimizer, resumed_scheduler = build_optimizer_and_scheduler(
            cfg, resumed, 1e-2
        )
        dcp.load(
            {"app": TrainingState(resumed, resumed_optimizer, resumed_scheduler)},
            checkpoint_id=folder,
        )
        for model, optimizer in ((reference, runs[0][1]), (resumed, resumed_optimizer)):
            _random_grads(model, model is resumed, 3)
            optimizer.step()
        _assert_same(resumed, reference)
    finally:
        dist.destroy_process_group()


def test_muon_sharded_step_equals_unsharded_on_two_ranks(tmp_path):
    mp.spawn(_two_rank_muon, args=(free_port(), str(tmp_path)), nprocs=2)


def _attention(qk_clip, qk_norm=False):
    """One small GQA layer: 4 query heads of 16 dims, 2 kv heads."""
    torch.manual_seed(0)
    attention = {
        "type": "gqa",
        "qk_norm": qk_norm,
        "output_gate": False,
        "no_rope_layers": None,
        "qk_clip": qk_clip,
    }
    return build_attention(
        0, attention=attention, n_blocks=1, dmodel=64, q_heads=4, kv_heads=2, seq_len=16
    )


def test_qk_clip_tracks_each_heads_max_causal_logit():
    layer = _attention(qk_clip=100.0)
    x = torch.randn(2, 16, 64)
    layer(x)
    q = layer.rope(layer.q_proj(x).view(2, 16, 4, 16).transpose(1, 2))
    k = layer.rope(layer.k_proj(x).view(2, 16, 2, 16).transpose(1, 2))
    logits = q @ k.repeat_interleave(2, dim=1).transpose(-2, -1) / 4
    causal = torch.ones(16, 16, dtype=torch.bool).tril()
    expected = logits[..., causal].amax(dim=(0, 2))
    torch.testing.assert_close(layer.max_logits, expected)
    layer.eval()
    layer(3 * x)
    torch.testing.assert_close(layer.max_logits, expected)


def test_qk_clip_caps_every_heads_max_logit_at_the_threshold():
    layer = _attention(qk_clip=100.0)
    x = torch.randn(2, 16, 64)
    layer(x)
    before = layer.max_logits.clone()
    # Between heads 0 and 1, which share a kv head: one is clipped, its partner not.
    layer.qk_clip = before[:2].mean().item()
    assert before[:2].min() < layer.qk_clip < before[:2].max()
    assert qk_clip_([layer]) == before.max()
    assert (layer.max_logits == float("-inf")).all()
    layer(x)
    torch.testing.assert_close(layer.max_logits, before.clamp(max=layer.qk_clip))


def test_qk_clip_needs_qk_norm_off():
    with pytest.raises(ValueError, match="qk_norm"):
        _attention(qk_clip=100.0, qk_norm=True)


def test_qk_clip_keeps_the_init_and_the_checkpoint_keys():
    off, on = _attention(qk_clip=None), _attention(qk_clip=100.0)
    assert not hasattr(off, "max_logits")
    assert on.state_dict().keys() == off.state_dict().keys()
    for a, b in zip(off.parameters(), on.parameters(), strict=True):
        assert torch.equal(a, b)


class _Recorder:
    def __init__(self):
        self.logged = []

    def log(self, name, value):
        self.logged.append((name, value))


QK_CLIP = ["common.attention.qk_norm=false", "common.attention.qk_clip=0.5"]


def test_add_qk_clip_clips_every_step_and_logs_the_window_max():
    model, optimizer, scheduler = _build("muon", overrides=QK_CLIP)
    recorder = _Recorder()
    add_qk_clip(optimizer, model, recorder, log_every=2)
    layers = [m for m in model.modules() if isinstance(m, RoPEAttention)]
    for _ in range(4):
        _step(model, optimizer, scheduler)
        # qk_clip_ ran and reset the trackers after every step.
        assert all((m.max_logits == float("-inf")).all() for m in layers)
    assert [name for name, _ in recorder.logged] == ["train/max_attention_logit"] * 2
    # h4's logits top 0.5 from the start, so every step clipped.
    assert all(value > 0.5 for _, value in recorder.logged)


def test_add_qk_clip_does_nothing_when_off():
    model, optimizer, _ = _build("muon")
    add_qk_clip(optimizer, model, _Recorder())
    assert not optimizer._optimizer_step_post_hooks


def _two_rank_qk_clip(rank, port):
    os.environ.update(MASTER_ADDR="localhost", MASTER_PORT=str(port), RANK=str(rank))
    dist.init_process_group("gloo", rank=rank, world_size=2)
    try:
        cfg = _cfg(
            "muon", ["common.attention.qk_norm=false", "common.attention.qk_clip=1"]
        )
        torch.manual_seed(0)
        reference = instantiate(cfg.model, _convert_="all")
        sharded = copy.deepcopy(reference)
        mesh = init_device_mesh("cpu", (2,))
        for m in sharded.modules():
            if isinstance(m, TransformerBlock):
                fully_shard(m, mesh=mesh)
        fully_shard(sharded, mesh=mesh)
        ref_layers, layers = (
            [m for m in model.modules() if isinstance(m, RoPEAttention)]
            for model in (reference, sharded)
        )
        # Each rank saw its own batch; the clip must use the max over both.
        seen = [
            torch.rand(len(layers), 4, generator=torch.Generator().manual_seed(r)) * 4
            for r in range(2)
        ]
        for i, (ref_layer, layer) in enumerate(zip(ref_layers, layers)):
            layer.max_logits.copy_(seen[rank][i])
            ref_layer.max_logits.copy_(torch.maximum(seen[0][i], seen[1][i]))
        top = qk_clip_(layers)
        assert top == qk_clip_(ref_layers) and top > 1, "nothing was clipped"
        _assert_same(sharded, reference)
    finally:
        dist.destroy_process_group()


def test_qk_clip_sharded_equals_unsharded_on_two_ranks():
    mp.spawn(_two_rank_qk_clip, args=(free_port(),), nprocs=2)
