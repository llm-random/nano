"""The MoE layer (src/core/moe.py): capacity, dropped tokens, eval routing, balance loss."""

from functools import partial

import pytest
import torch
import torch.nn.functional as F

from src.core.model import Linear, Residual, RMSNorm, llm_random_weight_init
from src.core.moe import MoE, load_balancing_loss

DMODEL = 8


def _moe(num_experts, topk, capacity_factor):
    def linear(in_features, out_features):
        init = partial(llm_random_weight_init, scale=1)
        return partial(Linear, in_features, out_features, partial_init_fn=init)

    torch.manual_seed(0)
    return MoE(
        ff_pre_act_fn=linear(DMODEL, 16),
        ff_post_act_fn=linear(16, DMODEL),
        gate_fn=linear(DMODEL, 16),
        num_experts=num_experts,
        topk=topk,
        capacity_factor=capacity_factor,
        load_balancing_loss_factor=0.01,
        router_z_loss_factor=0.001,
    )


def _everyone_wants_expert_0():
    """16 tokens whose router logit for expert 0 grows with feature 0, which is positive."""
    moe = _moe(num_experts=4, topk=1, capacity_factor=1.0)
    with torch.no_grad():
        moe.router_weight.zero_()
        moe.router_weight[0, 0] = 1.0
    x = torch.randn(2, 8, DMODEL)
    x[..., 0] = x[..., 0].abs() + 1
    return moe, x


def test_routing_respects_capacity():
    moe, x = _everyone_wants_expert_0()
    capacity = moe.capacity(16)
    assert capacity == 4  # ceil(1.0 * 16 tokens * top-1 / 4 experts)
    served = (moe(x).reshape(16, DMODEL) != 0).any(-1)
    assert served.sum() == capacity
    # the slots go to the highest router weights
    assert set(served.nonzero().flatten().tolist()) == set(
        x[..., 0].flatten().topk(capacity).indices.tolist()
    )
    assert moe.stats[2].item() == pytest.approx(1 - capacity / 16)


def test_slots_count_up_within_each_expert_by_router_weight():
    moe = _moe(num_experts=4, topk=2, capacity_factor=1.0)
    expert_ids = torch.randint(0, 4, (64,))
    weights = torch.rand(64)
    slots = moe.slot_in_expert(expert_ids, weights)
    for expert in range(4):
        mine = expert_ids == expert
        by_weight = weights[mine].argsort(descending=True)
        assert slots[mine][by_weight].tolist() == list(range(int(mine.sum())))


def test_a_dropped_token_comes_out_zero_so_the_residual_passes_it_unchanged():
    moe, x = _everyone_wants_expert_0()
    block = Residual(norm=RMSNorm(DMODEL), layer=moe, log_name="ff")
    unchanged = (block(x) == x).all(-1)
    assert unchanged.sum() == 16 - moe.capacity(16)


def test_eval_drops_nothing_so_right_padding_cannot_change_real_tokens():
    moe, x = _everyone_wants_expert_0()
    moe.eval()
    assert (moe(x).reshape(16, DMODEL) != 0).any(-1).all()
    alone = moe(x[:1, :5])
    padded = torch.cat([x[:1, :5], torch.full((1, 3, DMODEL), 5.0)], dim=1)
    batched = moe(torch.cat([padded, x[1:]]))
    torch.testing.assert_close(batched[:1, :5], alone)


def test_the_balance_loss_is_at_its_minimum_for_uniform_routing():
    experts, tokens = 4, 32
    logits = torch.zeros(tokens, experts, requires_grad=True)
    round_robin = torch.arange(tokens) % experts
    uniform = load_balancing_loss(logits.softmax(-1), round_robin, experts)
    assert uniform.item() == pytest.approx(1.0)
    uniform.backward()
    assert logits.grad.abs().max() < 1e-7  # a stationary point
    # a skew costs more; full collapse onto one expert costs num_experts
    all_zero = torch.zeros(tokens, dtype=torch.long)
    skewed = torch.zeros(tokens, experts)
    skewed[:, 0] = 0.5
    assert load_balancing_loss(skewed.softmax(-1), all_zero, experts) > 1.0
    collapsed = F.one_hot(all_zero, experts).float()
    assert load_balancing_loss(collapsed, all_zero, experts).item() == pytest.approx(
        experts
    )
