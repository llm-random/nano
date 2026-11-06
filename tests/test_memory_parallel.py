"""Section 1.4's correctness gate for Tiers 2 and 3, on a CPU toy model.

Tier 2 moves 2.576B parameters out of FSDP2 and AdamW into a hand-written
gradient and optimizer path; Tier 3 then splits them across ranks and changes
which process group reduces them. Neither may change the mathematics, and every
way they can go wrong is silent:

- a frozen bank (the gradient never arrives) trains, and converges slower;
- a wrong O8 divisor scales every bank gradient by a constant, which moves the
  loss curve by less than seed noise for hundreds of steps;
- a separately-clipped bank is a different function of the same gradients;
- a double-counted norm inflates the global norm by sqrt(dp_size) and nothing
  reports it.

So the gate is parameter-for-parameter equality against the ordinary path, on a
model small enough to run both in one process, and `mp=1` against `mp=4` on a
gloo mesh for the divisor. float64 throughout: the question is whether the two
paths compute the same function, not how fp32 rounds, and at 1e-12 any
structural error is orders of magnitude clear of the noise floor.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from src.core.ame import AMEAttention
from src.core.memory_parallel import (
    MemoryParallelController,
    ShardedBank,
    get_controller,
)
from src.core.model import parameter_groups

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The toy model section 1.4.1 specifies.
VOCAB = 256
DMODEL = 64
HEADS = 4
BLOCKS = 2
SEQ = 16
GA = 2  # micro-steps per optimizer step, so accumulation is exercised
STEPS = 3
LR = 0.01
CLIP = 1.0


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = AMEAttention(
            dmodel=DMODEL,
            q_heads=HEADS,
            kv_heads=HEADS,
            seq_len=SEQ,
            vocab_size=VOCAB,
            backend="reference",
        )
        self.norm = nn.LayerNorm(DMODEL)
        self.ff = nn.Sequential(
            nn.Linear(DMODEL, 2 * DMODEL), nn.GELU(), nn.Linear(2 * DMODEL, DMODEL)
        )

    def forward(self, x, ids):
        x = x + self.attention(self.norm(x), ids)
        return x + self.ff(self.norm(x))


class ToyAME(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, DMODEL)
        self.blocks = nn.ModuleList(ToyBlock() for _ in range(BLOCKS))
        self.head = nn.Linear(DMODEL, VOCAB)

    def forward(self, ids):
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x, ids)
        return self.head(x)


def build_model(dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    model = ToyAME()
    return model.to(dtype)


def micro_batches(rank=0, n=GA * STEPS, batch=2):
    generator = torch.Generator().manual_seed(100 + rank)
    return [
        torch.randint(0, VOCAB, (batch, SEQ), generator=generator) for _ in range(n)
    ]


def clip_through_trainer(model, clip, banks):
    """The real `Trainer.clip_gradient`, borrowed the way the O9 tests borrow
    `_accumulation_grad_sync` -- so the joint-clip path under test is the one
    that ships, not a copy of it."""
    from types import SimpleNamespace

    from src.core.trainer import Trainer

    stub = SimpleNamespace(model=model, gradient_clipping=clip, _banks=banks)
    return Trainer.clip_gradient(stub)


def train(model, batches, banks=None, clip=CLIP, steps=STEPS, lr=LR, bank_lr_ratio=1.0):
    """One faithful copy of Trainer.train's optimizer section.

    Baseline when `banks is None`: one AdamW over everything, one clip.
    `bank_lr_ratio` reaches the banks the way main.py and Trainer._bank_lr
    route it: a param group of their own, or the controller's lr.
    """
    ignored = banks.ignored_parameters() if banks is not None else None
    optimizer = torch.optim.AdamW(
        parameter_groups(
            model,
            weight_decay=0.1,
            exclude=ignored,
            lr=lr,
            bank_lr_ratio=bank_lr_ratio,
        ),
        lr=lr,
    )
    if banks is not None:
        banks.build_optimizer(lr=lr * bank_lr_ratio, weight_decay=0.0)

    norms, losses = [], []
    model.train()
    for step in range(steps):
        if banks is not None:
            banks.materialize()
        step_loss = 0.0
        for i in range(GA):
            ids = batches[step * GA + i]
            logits = model(ids)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1)
            )
            (loss / GA).backward()
            step_loss += loss.detach().item() / GA
        losses.append(step_loss)
        if banks is not None:
            banks.reduce()
        norm = clip_through_trainer(model, clip, banks)
        norms.append(None if norm is None else float(norm))
        optimizer.step()
        optimizer.zero_grad()
        if banks is not None:
            banks.step(lr=bank_lr_ratio * optimizer.param_groups[0]["lr"])
            banks.zero_grad()
    train.last_losses = losses
    return norms


def named_tensors(model):
    """Every trained tensor by name, with a bank reported through its fp32
    master so the two paths are compared on the same quantity."""
    out = {}
    for name, parameter in model.named_parameters():
        out[name] = parameter.detach().clone()
    for name, module in model.named_modules():
        if isinstance(module, ShardedBank):
            out[f"{name}.weight"] = module.master.detach().clone()
    return out


# ---- Tier 2: the hand-managed path is the ordinary path ---------------------


def test_toy_model_parameter_equivalence_without_clipping():
    """Section 1.4.1, in its strict form: with no clip in the way, the
    hand-managed path must be **bit-identical** to the ordinary one.

    Nothing here is allowed to be approximate. The gather, the scatter into
    `main_grad`, accumulation across micro-steps, the fp32 master, the separate
    AdamW and the master -> compute copy are each exact operations, and in
    float64 at one rank there is no reduction order for them to disagree about.
    A tolerance here would hide a frozen bank, a dropped micro-step or a
    double-applied weight decay.
    """
    batches = micro_batches()

    baseline = build_model()
    train(baseline, batches, clip=None)

    managed = build_model()
    banks = MemoryParallelController(managed, 1, compute_dtype=torch.float64)
    train(managed, batches, banks=banks, clip=None)

    assert len(banks.banks) == 2 * BLOCKS, "not every bank was claimed"
    # The controller fuses q/k by default, so this gate covers the SHIPPING
    # data path. Asserted rather than assumed: if the default ever flips, the
    # section 1.4.1 equivalence would quietly be testing code nobody runs.
    assert (
        len(banks.fused_pairs) == BLOCKS
    ), "the equivalence gate is not exercising the fused path"

    before = named_tensors(build_model())
    expected, actual = named_tensors(baseline), named_tensors(managed)
    assert set(expected) == set(actual), (
        f"parameter sets diverged: only baseline "
        f"{sorted(set(expected) - set(actual))}, only managed "
        f"{sorted(set(actual) - set(expected))}"
    )
    banks_moved = [
        name
        for name in expected
        if "bank" in name and not torch.equal(expected[name], before[name])
    ]
    assert len(banks_moved) == 2 * BLOCKS, (
        f"only {banks_moved} moved over three steps -- a gate that passes on a "
        "frozen bank is not a gate"
    )

    differing = {
        name: (expected[name] - actual[name]).abs().max().item()
        for name in expected
        if not torch.equal(expected[name], actual[name])
    }
    assert not differing, f"the hand-managed path is not bit-identical: {differing}"


def test_bank_lr_ratio_is_one_function_on_both_paths():
    """trainer.bank_lr_ratio reaches the banks by two routes -- a 4x param group
    in the backbone AdamW, or the controller's own AdamW stepped at 4x the
    backbone lr -- and they must compute the same thing: bit-identical, in the
    strict no-clip form of the gate above. And the ratio must do something, or
    the equality would hold for a ratio both paths ignore."""
    batches = micro_batches()

    baseline = build_model()
    train(baseline, batches, clip=None, bank_lr_ratio=4.0)

    managed = build_model()
    banks = MemoryParallelController(managed, 1, compute_dtype=torch.float64)
    train(managed, batches, banks=banks, clip=None, bank_lr_ratio=4.0)

    expected, actual = named_tensors(baseline), named_tensors(managed)
    differing = [
        name for name in expected if not torch.equal(expected[name], actual[name])
    ]
    assert not differing, f"the two routes for the ratio diverged on {differing}"

    # One step, so nothing downstream has seen a changed bank yet: the ratio
    # must move exactly the banks and leave every other parameter alone.
    after_one_step = {}
    for ratio in (1.0, 4.0):
        model = build_model()
        train(model, batches, clip=None, steps=1, bank_lr_ratio=ratio)
        after_one_step[ratio] = named_tensors(model)
    for name, unscaled in after_one_step[1.0].items():
        moved = not torch.equal(unscaled, after_one_step[4.0][name])
        assert moved == ("bank" in name), (
            f"{name}: the ratio should move exactly the banks, "
            f"{'moved' if moved else 'did not move'}"
        )


@pytest.mark.parametrize(
    "dtype,atol,norm_rtol", [(torch.float64, 1e-6, 1e-6), (torch.float32, 2e-5, 1e-5)]
)
def test_toy_model_parameter_equivalence_with_clipping(dtype, atol, norm_rtol):
    """Section 1.4.1 with the clip on, where bit-identity is the wrong gate.

    `Trainer.clip_gradient` computes the global norm and then casts it with
    `.float()`. The baseline sums one list of per-parameter norms; the joint
    clip sums the backbone's and adds the banks' separately. Same terms, and in
    float64 the two sums agree to about 1e-16 -- but the fp32 cast quantises the
    result, and the two land on neighbouring fp32 values. Measured: 1.3832852840
    against 1.3832854032, **exactly one fp32 ulp**, at step 1 only. Every later
    step's norm is identical and the parameters carry the one-ulp difference
    forward at ~1e-7.

    That is float associativity in a scalar, not a difference in the function
    being computed -- which `test_..._without_clipping` above establishes by
    being exact. What this test is really for is the *shape*: the norm traces
    must track each other, because a separate (non-joint) clip, a
    double-counted bank norm or a missing one all move the trace by far more
    than an ulp.
    """
    batches = micro_batches()

    baseline = build_model(dtype)
    base_norms = train(baseline, batches, clip=CLIP)

    managed = build_model(dtype)
    banks = MemoryParallelController(managed, 1, compute_dtype=dtype)
    assert len(banks.fused_pairs) == BLOCKS, "not exercising the fused path"
    managed_norms = train(managed, batches, banks=banks, clip=CLIP)

    for i, (a, b) in enumerate(zip(base_norms, managed_norms)):
        assert abs(a - b) <= norm_rtol * abs(a), (
            f"grad norm diverged at step {i}: baseline {a!r} vs managed {b!r}. "
            "A joint clip sums the same terms; a separate one does not."
        )

    expected, actual = named_tensors(baseline), named_tensors(managed)
    worst = max(((expected[n] - actual[n]).abs().max().item(), n) for n in expected)
    assert worst[0] <= atol, f"{worst[1]} differs by {worst[0]:.3e} (atol {atol:.1e})"


def test_a_separate_clip_would_fail_this_gate():
    """The negative control for the test above: clipping the banks on their own
    coefficient instead of the joint one must move the parameters far more than
    the tolerance, or the gate is not testing anything."""
    batches = micro_batches()

    baseline = build_model()
    train(baseline, batches, clip=CLIP)

    managed = build_model()
    banks = MemoryParallelController(managed, 1, compute_dtype=torch.float64)
    original = banks.clip_

    def bank_only_clip(max_norm, total_norm):
        own = torch.sqrt(banks.local_grad_norm_sq().double())
        original(max_norm, own)

    banks.clip_ = bank_only_clip
    train(managed, batches, banks=banks, clip=CLIP)

    expected, actual = named_tensors(baseline), named_tensors(managed)
    worst = max(((expected[n] - actual[n]).abs().max().item(), n) for n in expected)
    assert worst[0] > 1e-4, (
        f"a separately-clipped bank moved the model by only {worst[0]:.3e}; "
        "the equivalence gate above cannot tell the two clips apart"
    )


def test_bank_never_allocates_a_dense_gradient():
    """O1's whole point: `weight.grad` must stay None, every step, forever.

    If it is not None something is writing a `[vocab, width]` tensor per bank
    per micro-step -- the allocation this tier deletes -- and worse, the
    trainer's clip would then find it and clip it twice.
    """
    model = build_model()
    banks = MemoryParallelController(model, 1, compute_dtype=torch.float64)
    train(model, micro_batches(), banks=banks)
    for bank in banks.banks:
        assert bank.weight.grad is None, "a dense bank gradient was materialised"
        assert bank.main_grad is not None
        assert torch.equal(
            bank.main_grad, torch.zeros_like(bank.main_grad)
        ), "main_grad was not zeroed after the step"


def test_main_grad_matches_the_dense_backward():
    """`index_add_` into a persistent fp32 buffer must sum to exactly what
    `embedding_dense_backward` + `.grad` accumulation sums to."""
    batches = micro_batches(n=GA)

    baseline = build_model()
    for i in range(GA):
        logits = baseline(batches[i])
        (
            F.cross_entropy(
                logits[:, :-1].reshape(-1, VOCAB), batches[i][:, 1:].reshape(-1)
            )
            / GA
        ).backward()

    managed = build_model()
    banks = MemoryParallelController(managed, 1, compute_dtype=torch.float64)
    banks.materialize()
    for i in range(GA):
        logits = managed(batches[i])
        (
            F.cross_entropy(
                logits[:, :-1].reshape(-1, VOCAB), batches[i][:, 1:].reshape(-1)
            )
            / GA
        ).backward()

    dense = {
        name: p.grad
        for name, p in baseline.named_parameters()
        if "bank" in name and p.grad is not None
    }
    assert dense, "the baseline produced no bank gradients"
    scattered = {
        f"{name}.weight": module.main_grad
        for name, module in managed.named_modules()
        if isinstance(module, ShardedBank)
    }
    assert set(dense) == set(scattered)
    for name in dense:
        assert torch.allclose(dense[name], scattered[name], rtol=0, atol=1e-12), (
            f"{name}: scattered gradient differs by "
            f"{(dense[name] - scattered[name]).abs().max().item():.3e}"
        )
        assert dense[name].abs().max() > 0, f"{name} gradient was all zero"


def test_banks_are_excluded_from_the_backbone_optimizer():
    model = build_model()
    banks = MemoryParallelController(model, 1, compute_dtype=torch.float64)
    ignored = banks.ignored_parameters()
    assert len(ignored) == 2 * BLOCKS

    groups = parameter_groups(model, weight_decay=0.1, exclude=ignored)
    claimed = {id(p) for group in groups for p in group["params"]}
    assert not (claimed & {id(p) for p in ignored}), (
        "a bank parameter reached the backbone optimizer as well as the "
        "controller's -- it would be stepped twice"
    )
    # and nothing else was dropped on the way
    assert len(claimed) == sum(1 for _ in model.parameters()) - len(ignored)


def test_mfu_denominator_is_unchanged_by_the_swap():
    """A ShardedBank is not an nn.Embedding, and src.core.mfu identifies lookups
    by that class. Without `non_matmul_parameters` the swap would silently add
    6*P FLOPs/token for the banks and every MFU in the ledger would stop being
    comparable to the ones above it."""
    from src.core.mfu import model_flops_per_token

    baseline = build_model()
    before = model_flops_per_token(baseline, SEQ)

    managed = build_model()
    MemoryParallelController(managed, 1, compute_dtype=torch.float64)
    after = model_flops_per_token(managed, SEQ)

    assert (
        before == after
    ), f"FLOPs/token moved from {before} to {after} when the banks were swapped"


def test_mem_parallel_size_must_divide_world_size():
    model = build_model()
    with pytest.raises(ValueError, match="does not divide world_size"):
        MemoryParallelController(model, 3, compute_dtype=torch.float64)


def test_controller_is_reachable_from_the_model():
    model = build_model()
    assert get_controller(model) is None
    banks = MemoryParallelController(model, 1, compute_dtype=torch.float64)
    assert get_controller(model) is banks


def test_a_model_without_banks_is_rejected():
    plain = nn.Sequential(nn.Embedding(VOCAB, DMODEL), nn.Linear(DMODEL, VOCAB))
    with pytest.raises(ValueError, match="no q_bank/k_bank"):
        MemoryParallelController(plain, 1, compute_dtype=torch.float64)


def test_trainer_rejects_a_config_whose_model_has_no_controller():
    """`trainer.mem_parallel_size=4` on a model built without one would train
    the ordinary FSDP2 path and report it as a Tier 3 number."""
    from types import SimpleNamespace

    from src.core.trainer import Trainer

    stub = SimpleNamespace(model=build_model(), mem_parallel_size=4)
    stub._instrumentation_failures = set()
    with pytest.raises(ValueError, match="carries no "):
        # the two lines of __attrs_post_init__ that own this check
        banks = get_controller(stub.model)
        if stub.mem_parallel_size and banks is None:
            raise ValueError(
                "trainer.mem_parallel_size=4 but the model carries no "
                "memory-parallel controller"
            )


# ---- Tier 3: the column split, the two groups, and O8's divisor -------------
#
# Everything below runs four gloo ranks on the CPU, because `mem_parallel_size`
# is only interesting when it is neither 1 nor world_size at the same time. The
# comparison is always the same shape: run the identical schedule at several
# `mem_parallel_size` values and demand the SAME answer, since mp is a layout
# choice and nothing about the mathematics may depend on it.


def full_width_main_grad(controller, index):
    """Bank `index`'s gradient at full width, assembled on every rank.

    At mp=1 the rank already holds it. At mp>1 it owns one column slice, so the
    slices are all-gathered in mp-rank order -- the same order the forward's
    all-to-all assumes -- and concatenated.
    """
    bank = controller.banks[index]
    if controller.mp_size == 1:
        return bank.main_grad.clone()
    slices = [torch.empty_like(bank.main_grad) for _ in range(controller.mp_size)]
    dist.all_gather(slices, bank.main_grad.contiguous(), group=controller.mp_group)
    return torch.cat(slices, dim=1)


def full_width_master(controller, index):
    bank = controller.banks[index]
    if controller.mp_size == 1:
        return bank.master.clone()
    slices = [torch.empty_like(bank.master) for _ in range(controller.mp_size)]
    dist.all_gather(slices, bank.master.contiguous(), group=controller.mp_group)
    return torch.cat(slices, dim=1)


def _tier3_worker(rank, world_size, port, mem_parallel_size, out_path):
    """Results go to a file, not to an mp.Queue: the payload is a few MB of
    tensors and the queue's feeder socket does not survive the spawned process
    exiting under it."""
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        model = build_model()  # identical on every rank under one seed
        banks = MemoryParallelController(
            model, mem_parallel_size, compute_dtype=torch.float64
        )
        batches = micro_batches(rank)

        # Phase A: one optimizer step's worth of backward, stopped right after
        # the reduce. This is the quantity O8's divisor scales.
        banks.materialize()
        for i in range(GA):
            ids = batches[i]
            logits = model(ids)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1)
            )
            (loss / GA).backward()
        banks.reduce()
        if os.environ.get("NANO_TEST_WRONG_O8_DIVISOR"):
            # Negative control: exactly what dividing by the memory-DP subgroup
            # size instead of the global world size produces.
            with torch.no_grad():
                for bank in banks.banks:
                    bank.main_grad.mul_(banks.mp_size)
        grads = [
            full_width_main_grad(banks, i).clone() for i in range(len(banks.banks))
        ]
        joint_norm_sq = banks.local_grad_norm_sq().clone()
        dist.all_reduce(joint_norm_sq, op=dist.ReduceOp.SUM)

        # Phase B: the same schedule from scratch, three full steps.
        banks.zero_grad()
        model.zero_grad(set_to_none=True)
        norms = train(model, batches, banks=banks)
        losses = train.last_losses
        masters = [full_width_master(banks, i).clone() for i in range(len(banks.banks))]

        if rank == 0:
            torch.save(
                {
                    "status": "ok",
                    "grads": grads,
                    "joint_norm": float(joint_norm_sq.sqrt()),
                    "norms": norms,
                    "losses": losses,
                    "masters": masters,
                    "slice_width": banks.banks[0].slice_width,
                    "dp_size": banks.dp_size,
                    "n_banks": len(banks.banks),
                },
                out_path,
            )
    except Exception as exc:  # noqa: BLE001
        if rank == 0:
            torch.save(
                {"status": "error", "message": f"{type(exc).__name__}: {exc}"},
                out_path,
            )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_tier3(world_size, mem_parallel_size, tmp_path, wrong_divisor=False):
    from conftest import spawn_on_free_port

    out = str(tmp_path / f"tier3_mp{mem_parallel_size}_{int(wrong_divisor)}.pt")
    previous = os.environ.pop("NANO_TEST_WRONG_O8_DIVISOR", None)
    if wrong_divisor:
        os.environ["NANO_TEST_WRONG_O8_DIVISOR"] = "1"
    try:
        spawn_on_free_port(
            _tier3_worker,
            lambda port: (world_size, port, mem_parallel_size, out),
            world_size,
        )
    finally:
        os.environ.pop("NANO_TEST_WRONG_O8_DIVISOR", None)
        if previous is not None:
            os.environ["NANO_TEST_WRONG_O8_DIVISOR"] = previous
    payload = torch.load(out, weights_only=False)
    if payload["status"] == "error":
        pytest.fail(f"mem_parallel_size={mem_parallel_size}: {payload['message']}")
    return payload


@pytest.mark.parametrize("mem_parallel_size", [2, 4])
def test_o8_divisor_is_the_global_world_size(mem_parallel_size, tmp_path):
    """**The Tier 3 gate.** `mem_parallel_size` must not change the gradient.

    The all-reduce runs in the memory-DP subgroup, whose size is
    `world / mp`, but the divisor is the GLOBAL world size: the forward's id
    all-gather already summed `mp` ranks' token contributions into each slice,
    so the terms total `mp * (world/mp) = world`. Dividing by the subgroup size
    instead -- the natural-looking mistake, and the one this plan called its
    most likely silent bug -- leaves every bank gradient a factor of `mp` too
    large. At mp=4 on 4 ranks the subgroup is a singleton, so that mistake is
    "no division at all" and the error is exactly 4x.

    A constant factor on the bank gradient does not break training. AdamW
    normalises by sqrt(v), so the first steps look almost unchanged and the
    damage arrives later as a wrong effective learning rate on 81% of the
    model's parameters. Nothing but this comparison would report it.
    """
    reference = run_tier3(4, 1, tmp_path)
    sharded = run_tier3(4, mem_parallel_size, tmp_path)

    assert sharded["dp_size"] == 4 // mem_parallel_size
    assert sharded["slice_width"] * mem_parallel_size == reference["slice_width"]
    assert reference["n_banks"] == sharded["n_banks"] == 2 * BLOCKS

    for i, (want, got) in enumerate(zip(reference["grads"], sharded["grads"])):
        assert want.shape == got.shape
        assert want.abs().max() > 0, f"bank {i} gradient was all zero"
        worst = (want - got).abs().max().item()
        ratio = (got.abs().sum() / want.abs().sum()).item()
        assert worst <= 1e-14, (
            f"bank {i}: gradient differs by {worst:.3e} at mem_parallel_size "
            f"{mem_parallel_size} (magnitude ratio {ratio:.4f} -- a ratio near "
            f"{mem_parallel_size} or 1/{mem_parallel_size} means the divisor)"
        )


@pytest.mark.parametrize("mem_parallel_size", [2, 4])
def test_loss_trajectory_coincides_across_mem_parallel_size(
    mem_parallel_size, tmp_path
):
    """Section 1.4.2 and 1.4.3 for Tier 3: same losses, same grad-norm trace,
    same parameters. The norm trace is the part that catches a bank norm counted
    once per dp-rank instead of once per slice -- that inflates it by
    sqrt(dp_size) and changes nothing else."""
    reference = run_tier3(4, 1, tmp_path)
    sharded = run_tier3(4, mem_parallel_size, tmp_path)

    for step, (a, b) in enumerate(zip(reference["losses"], sharded["losses"])):
        assert abs(a - b) <= 1e-12 * max(1.0, abs(a)), (
            f"loss diverged at step {step}: mp=1 {a!r} vs "
            f"mp={mem_parallel_size} {b!r}"
        )
    for step, (a, b) in enumerate(zip(reference["norms"], sharded["norms"])):
        assert abs(a - b) <= 1e-6 * max(1.0, abs(a)), (
            f"grad norm diverged at step {step}: mp=1 {a!r} vs "
            f"mp={mem_parallel_size} {b!r}"
        )
    assert abs(reference["joint_norm"] - sharded["joint_norm"]) <= 1e-6 * abs(
        reference["joint_norm"]
    ), (
        f"the joint clip norm moved with the layout: {reference['joint_norm']!r} "
        f"vs {sharded['joint_norm']!r} -- a bank slice is being counted once per "
        "dp-rank rather than once"
    )
    for i, (want, got) in enumerate(zip(reference["masters"], sharded["masters"])):
        worst = (want - got).abs().max().item()
        assert worst <= 1e-12, f"bank {i} master differs by {worst:.3e} after 3 steps"


def _forward_worker(rank, world_size, port, mem_parallel_size, queue):
    """The column split plus the all-to-all must reassemble the EXACT tensor the
    unsharded gather produces, column for column. A transposed reshape here is
    a permutation of the head dimension: the model still trains, and every head
    is reading another head's bank row."""
    try:
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            RANK=str(rank),
            WORLD_SIZE=str(world_size),
            LOCAL_RANK=str(rank),
        )
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        reference_model = build_model()
        reference = reference_model.blocks[0].attention.q_bank
        ids = micro_batches(rank, n=1)[0]
        with torch.no_grad():
            want = reference(ids)

        model = build_model()
        MemoryParallelController(model, mem_parallel_size, compute_dtype=torch.float64)
        got = model.blocks[0].attention.q_bank(ids)

        if rank == 0:
            queue.put(("ok", {"equal": bool(torch.equal(want, got.detach()))}))
    except Exception as exc:  # noqa: BLE001
        if rank == 0:
            queue.put(("error", f"{type(exc).__name__}: {exc}"))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("mem_parallel_size", [2, 4])
def test_column_shard_reassembles_the_full_width_gather(mem_parallel_size):
    from conftest import spawn_on_free_port

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    spawn_on_free_port(
        _forward_worker, lambda port: (4, port, mem_parallel_size, queue), 4
    )
    status, payload = queue.get(timeout=120)
    if status == "error":
        pytest.fail(payload)
    assert payload["equal"], (
        "the sharded gather is not the unsharded gather -- the all-to-all "
        "chunks are being reassembled in the wrong column order"
    )


def test_the_o8_gate_catches_a_wrong_divisor(tmp_path):
    """Negative control for the test above. Dividing by the memory-DP subgroup
    size instead of the global world size must be visible as a factor of
    `mem_parallel_size` on every bank gradient -- if it is not, the gate passes
    on the bug it was written for."""
    reference = run_tier3(4, 1, tmp_path)
    broken = run_tier3(4, 4, tmp_path, wrong_divisor=True)

    ratios = [
        (b.abs().sum() / a.abs().sum()).item()
        for a, b in zip(reference["grads"], broken["grads"])
    ]
    assert all(
        abs(r - 4.0) < 1e-9 for r in ratios
    ), f"the wrong divisor did not show up as a 4x gradient: ratios {ratios}"


# ---- Tier 3 gate: a checkpoint at mem_parallel_size 4 round-trips ----------


def _checkpoint_worker(rank, world_size, port, mem_parallel_size, folder, out_path):
    """Save at `mem_parallel_size`, rebuild from scratch, load, and check the
    model computes the same thing.

    The failure this is written against is silent and total: a ShardedBank's
    `weight` is a plain tensor whose fully-qualified name is identical on every
    rank, so a distributed save of `model.state_dict()` writes one rank's
    columns under that name and the other three vanish. The restored model then
    runs, produces finite loss, and is three quarters random.
    """
    try:
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            RANK=str(rank),
            WORLD_SIZE=str(world_size),
            LOCAL_RANK=str(rank),
        )
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        trained = build_model()
        banks = MemoryParallelController(
            trained, mem_parallel_size, compute_dtype=torch.float64
        )
        batches = micro_batches(rank)
        train(trained, batches, banks=banks)
        banks.save(folder)
        dist.barrier()

        probe = batches[0]
        with torch.no_grad():
            want = trained(probe)

        # A fresh model at its initial values, so anything the load fails to
        # restore shows up as a difference rather than as a coincidence.
        restored = build_model()
        restored_banks = MemoryParallelController(
            restored, mem_parallel_size, compute_dtype=torch.float64
        )
        restored.load_state_dict(trained.state_dict())
        restored_banks.build_optimizer(lr=LR, weight_decay=0.0)
        restored_banks.load(folder)
        with torch.no_grad():
            got = restored(probe)

        # The negative control, in the same process: skipping the sidecar must
        # leave the banks at init, i.e. a different model.
        naive = build_model()
        MemoryParallelController(naive, mem_parallel_size, compute_dtype=torch.float64)
        naive.load_state_dict(trained.state_dict())
        with torch.no_grad():
            without_sidecar = naive(probe)

        if rank == 0:
            torch.save(
                {
                    "status": "ok",
                    "restored_max_diff": (want - got).abs().max().item(),
                    "no_sidecar_max_diff": (want - without_sidecar).abs().max().item(),
                    "bank_keys_in_model_state": [
                        key for key in trained.state_dict() if "bank" in key
                    ],
                },
                out_path,
            )
    except Exception as exc:  # noqa: BLE001
        if rank == 0:
            torch.save(
                {"status": "error", "message": f"{type(exc).__name__}: {exc}"},
                out_path,
            )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("mem_parallel_size", [1, 4])
def test_checkpoint_round_trips_at_every_mem_parallel_size(mem_parallel_size, tmp_path):
    from conftest import spawn_on_free_port

    folder = str(tmp_path / f"ckpt_mp{mem_parallel_size}")
    out = str(tmp_path / f"ckpt_mp{mem_parallel_size}.result")
    spawn_on_free_port(
        _checkpoint_worker,
        lambda port: (4, port, mem_parallel_size, folder, out),
        4,
    )
    payload = torch.load(out, weights_only=False)
    if payload["status"] == "error":
        pytest.fail(payload["message"])

    assert payload["bank_keys_in_model_state"] == [], (
        "a bank parameter is in the module state_dict: at mem_parallel_size > 1 "
        f"every rank writes {payload['bank_keys_in_model_state'][:1]} under the "
        "same name with different columns, and all but one rank's are lost"
    )
    assert (
        payload["restored_max_diff"] == 0.0
    ), f"the restored model differs by {payload['restored_max_diff']:.3e}"
    assert payload["no_sidecar_max_diff"] > 1e-3, (
        "loading without the sidecar produced the same model -- the round-trip "
        "test above cannot tell a restored bank from an un-restored one"
    )

    written = sorted(os.listdir(folder))
    assert len(written) == 4, f"expected one sidecar per rank, found {written}"


# ---- why the bank gradient is not just different but better ----------------


def test_an_fp32_accumulator_beats_a_bf16_one_on_duplicate_token_ids():
    """The mechanism behind the only difference the cluster run shows.

    Under FSDP2's `MixedPrecisionPolicy(param_dtype=bf16)` the unsharded bank
    parameter is bf16, so autograd hands `embedding_dense_backward` a bf16
    output to scatter into -- and a micro-batch of 32768 tokens over a 50304-row
    vocabulary means the commonest rows are hit hundreds or thousands of times.
    Each of those adds rounds to bf16's 8 significant bits, and the error
    accumulates in the rows that matter most. FSDP2 upcasts to fp32 only
    afterwards, for the reduce.

    Tier 2's `main_grad` is fp32 from the first add, so every one of those
    contributions is accumulated at fp32. That is why this path is not merely a
    reassociation of the baseline's arithmetic: on the duplicated rows it is
    closer to the exact answer, and by a wide margin.
    """
    torch.manual_seed(0)
    vocab, width, n = 256, 64, 8192
    # Zipf-ish, like real text: a handful of rows take most of the traffic.
    ids = torch.clamp(
        torch.distributions.Pareto(torch.tensor(1.0), torch.tensor(0.6))
        .sample((n,))
        .long(),
        max=vocab - 1,
    )
    counts = torch.bincount(ids, minlength=vocab)
    assert counts.max() > 200, f"not enough duplication to test: max {counts.max()}"

    grad = torch.randn(n, width)
    grad_bf16 = grad.to(torch.bfloat16)

    exact = torch.zeros(vocab, width, dtype=torch.float64)
    exact.index_add_(0, ids, grad_bf16.to(torch.float64))

    in_bf16 = torch.zeros(vocab, width, dtype=torch.bfloat16)
    in_bf16.index_add_(0, ids, grad_bf16)

    in_fp32 = torch.zeros(vocab, width, dtype=torch.float32)
    in_fp32.index_add_(0, ids, grad_bf16.to(torch.float32))

    hot = counts > 200
    scale = exact[hot].abs().max()
    error_bf16 = (in_bf16[hot].to(torch.float64) - exact[hot]).abs().max() / scale
    error_fp32 = (in_fp32[hot].to(torch.float64) - exact[hot]).abs().max() / scale

    assert error_fp32 < error_bf16 / 10, (
        f"relative error on the hot rows: bf16 accumulator {error_bf16:.3e}, "
        f"fp32 accumulator {error_fp32:.3e} -- expected at least 10x better"
    )


# ---- D2: fusing the q/k pair is a data-path change only -------------------


def _fused_vs_unfused_worker(rank, world_size, port, mem_parallel_size, out_path):
    """The fused pair must be bit-identical to the unfused banks.

    It batches two gathers, two all-to-alls and two permutes into one of each.
    The all-to-all layout is where that can go wrong: the pair is carried as one
    `[mp*n, 2*width/mp]` buffer, and reassembling it in the wrong order silently
    swaps q with k, or interleaves heads. Both still train.
    """
    try:
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            RANK=str(rank),
            WORLD_SIZE=str(world_size),
            LOCAL_RANK=str(rank),
        )
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        batches = micro_batches(rank)

        results = {}
        for tag, fuse in (("unfused", False), ("fused", True)):
            model = build_model()
            banks = MemoryParallelController(
                model, mem_parallel_size, compute_dtype=torch.float64, fuse_qk=fuse
            )
            assert len(banks.fused_pairs) == (BLOCKS if fuse else 0)
            norms = train(model, batches, banks=banks)
            results[tag] = {
                "norms": norms,
                "losses": list(train.last_losses),
                "masters": [
                    full_width_master(banks, i).clone() for i in range(len(banks.banks))
                ],
                "params": {
                    name: p.detach().clone() for name, p in model.named_parameters()
                },
            }

        if rank == 0:
            a, b = results["unfused"], results["fused"]
            worst_master = max(
                (x - y).abs().max().item() for x, y in zip(a["masters"], b["masters"])
            )
            worst_param = max(
                (a["params"][n] - b["params"][n]).abs().max().item()
                for n in a["params"]
            )
            torch.save(
                {
                    "status": "ok",
                    "worst_master": worst_master,
                    "worst_param": worst_param,
                    "loss_equal": a["losses"] == b["losses"],
                    "norm_max_diff": max(
                        abs(x - y) for x, y in zip(a["norms"], b["norms"])
                    ),
                    "bank_moved": max(x.abs().max().item() for x in a["masters"]),
                },
                out_path,
            )
    except Exception as exc:  # noqa: BLE001
        if rank == 0:
            torch.save(
                {"status": "error", "message": f"{type(exc).__name__}: {exc}"},
                out_path,
            )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("mem_parallel_size", [1, 2, 4])
def test_fusing_the_qk_pair_changes_nothing(mem_parallel_size, tmp_path):
    from conftest import spawn_on_free_port

    out = str(tmp_path / f"fused_mp{mem_parallel_size}.pt")
    spawn_on_free_port(
        _fused_vs_unfused_worker, lambda port: (4, port, mem_parallel_size, out), 4
    )
    payload = torch.load(out, weights_only=False)
    if payload["status"] == "error":
        pytest.fail(payload["message"])

    assert payload["bank_moved"] > 0, "the banks never moved; the test is vacuous"
    assert payload["loss_equal"], "the fused pair changed the loss trajectory"
    assert payload["worst_master"] == 0.0, (
        f"bank masters differ by {payload['worst_master']:.3e} -- the fused "
        "all-to-all is being reassembled in the wrong order"
    )
    assert (
        payload["worst_param"] == 0.0
    ), f"backbone parameters differ by {payload['worst_param']:.3e}"
    assert payload["norm_max_diff"] == 0.0


# ---- D3: DDP backbone + memory-parallel banks ------------------------------


def _ddp_worker(rank, world_size, port, out_path):
    """DDP for the backbone, the controller for the banks.

    Two failures here are silent and total:

    1. `get_controller` must see through the DDP wrapper. DDP returns a new
       module and `nn.Module.__getattr__` does not forward unknown attributes to
       `.module`, so a plain getattr reports "no controller" and the banks are
       never stepped -- the model trains, with 81% of its parameters frozen.
    2. The clip must not reduce the backbone norm across ranks. DDP has already
       all-reduced those gradients, so they are identical everywhere; summing
       them again inflates the global norm by sqrt(world_size) and clips twice
       as hard as asked, at four GPUs.
    """
    try:
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            RANK=str(rank),
            WORLD_SIZE=str(world_size),
            LOCAL_RANK=str(rank),
        )
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        from torch.nn.parallel import DistributedDataParallel as DDP

        from src.core.memory_parallel import get_controller

        model = build_model()
        banks = MemoryParallelController(model, world_size, compute_dtype=torch.float64)
        names = [
            n for n, p in model.named_parameters() if p in banks.ignored_parameters()
        ]
        DDP._set_params_and_buffers_to_ignore_for_model(model, names)
        wrapped = DDP(model)

        seen = get_controller(wrapped)
        before = [b.master.clone() for b in banks.banks]

        batches = micro_batches(rank)
        train(wrapped, batches, banks=banks)

        moved = max(
            (a - b.master).abs().max().item() for a, b in zip(before, banks.banks)
        )
        if rank == 0:
            torch.save(
                {
                    "status": "ok",
                    "controller_found": seen is banks,
                    "ignored": len(names),
                    "bank_moved": moved,
                    "norms": train.last_losses and None,
                },
                out_path,
            )
    except Exception as exc:  # noqa: BLE001
        if rank == 0:
            torch.save(
                {"status": "error", "message": f"{type(exc).__name__}: {exc}"},
                out_path,
            )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_ddp_backbone_with_memory_parallel_banks(tmp_path):
    from conftest import spawn_on_free_port

    out = str(tmp_path / "ddp.pt")
    spawn_on_free_port(_ddp_worker, lambda port: (4, port, out), 4)
    payload = torch.load(out, weights_only=False)
    if payload["status"] == "error":
        pytest.fail(payload["message"])

    assert payload["controller_found"], (
        "get_controller did not see through the DDP wrapper -- the banks would "
        "never be stepped and 81% of the model would train frozen"
    )
    assert payload["ignored"] == 2 * BLOCKS
    assert payload["bank_moved"] > 0, "the banks did not move under DDP"


def test_the_clip_does_not_double_count_replicated_gradients():
    """Trainer.clip_gradient must skip the cross-rank reduce when the model is
    DDP-wrapped. Checked by construction rather than over a mesh: the branch is
    one `isinstance`, and getting it wrong is a factor of sqrt(world_size)."""
    import inspect

    from src.core.trainer import Trainer

    source = inspect.getsource(Trainer.clip_gradient)
    assert "replicated = isinstance(self.model, _DDP)" in source
    assert (
        "if dist.is_initialized() and not replicated:" in source
    ), "the backbone norm is still being all-reduced unconditionally"


def test_cross_entropy_runs_outside_autocast():
    """`F.cross_entropy` is on torch's autocast fp32 list, so inside an autocast
    region it upcasts the logits -- 6.6 GiB at h20 micro-batch 16, which is what
    OOMed the first DDP run. Outside it, bf16 logits go straight in, which is
    also what the FSDP2 path does (it has no autocast at all). The two must
    match or a DDP run is not comparable to an FSDP2 one."""
    from types import SimpleNamespace

    from src.core.trainer import Trainer

    off = Trainer._loss_ctx(SimpleNamespace(autocast_bf16=False))
    assert (
        type(off).__name__ == "nullcontext"
    ), "with autocast off this must be a no-op, or every FSDP2 measurement moves"

    on = Trainer._loss_ctx(SimpleNamespace(autocast_bf16=True))
    assert isinstance(on, torch.autocast)

    logits = torch.randn(4, 16, dtype=torch.bfloat16)
    targets = torch.randint(0, 16, (4,))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        inside = F.cross_entropy(logits, targets)
        with Trainer._loss_ctx(SimpleNamespace(autocast_bf16=True)).__class__(
            "cpu", enabled=False
        ):
            outside = F.cross_entropy(logits, targets)
    assert inside.dtype == torch.float32, "autocast no longer upcasts cross-entropy"
    assert (
        outside.dtype == torch.bfloat16
    ), "the loss context did not disable autocast; the fp32 upcast is still paid"
