"""Token superposition (arXiv 2605.06546): bags and targets, bag loss, bag schedule."""

import os
import warnings

import pytest
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir

import resolver  # noqa: F401  (registers the repo's custom OmegaConf resolvers)
from src.core.model import TransformerEmbedding
from src.core.trainer import Trainer
from src.definitions import check_point

FEATURE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "configs", "jm", "feature"
)


def test_each_bag_targets_the_next_bag():
    # Two spliced rows: 3 loader rows of 5 tokens each, ids = positions in the row.
    batch = torch.arange(2 * 15).view(2, 15)
    input_ids, target_ids = Trainer._preprocess_bags(batch, 3)
    assert input_ids.shape == target_ids.shape == (2, 4, 3)
    assert torch.equal(input_ids[0, 1], torch.tensor([3, 4, 5]))
    assert torch.equal(target_ids, input_ids + 3)


def test_bag_loss_is_the_mean_of_one_cross_entropy_per_bag_token():
    """Eq. 3, computed the paper's way (Listing 3: one F.cross_entropy per bag slot)."""
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 11)
    target_ids = torch.randint(11, (2, 4, 3))
    expected = torch.stack(
        [
            F.cross_entropy(logits.flatten(0, 1), target_ids[..., i].flatten())
            for i in range(3)
        ]
    ).mean()
    assert torch.allclose(Trainer._bag_cross_entropy(logits, target_ids), expected)


def test_a_bag_embeds_as_the_mean_of_its_tokens():
    torch.manual_seed(0)
    embedding = TransformerEmbedding(11, 8, init_fn=None)
    ids = torch.randint(11, (2, 4, 3))
    per_token = embedding(ids.flatten(1)).view(2, 4, 3, 8)
    assert torch.allclose(embedding(ids), per_token.mean(dim=2), atol=1e-6)


class _ScheduleStub:
    _bag_size = Trainer._bag_size
    _batches_from_start_step = Trainer._batches_from_start_step
    _rewarmup_factor = Trainer._rewarmup_factor
    _rewarmed_lr = Trainer._rewarmed_lr


def test_bag_size_holds_for_the_superposition_steps_then_drops_to_one():
    stub = _ScheduleStub()
    stub.token_superposition = {"bag_size": 4, "steps": 3}
    assert [stub._bag_size(step) for step in range(5)] == [4, 4, 4, 1, 1]
    # The configs/_trainer default: off.
    stub.token_superposition = {"bag_size": 1, "steps": 0}
    assert stub._bag_size(0) == 1


def test_a_resume_skips_bag_size_batches_per_superposition_step():
    stub = _ScheduleStub()
    stub.token_superposition = {"bag_size": 4, "steps": 3}
    stub.batch_size_warmup = None
    stub.train_dataloader = range(100)
    stub.start_step = 5
    # 3 superposition steps took 4 batches each, steps 3 and 4 took one each.
    assert next(iter(stub._batches_from_start_step())) == 3 * 4 + 2


def test_lr_rewarms_linearly_over_the_steps_after_the_switch():
    stub = _ScheduleStub()
    stub.token_superposition = {"bag_size": 6, "steps": 3, "rewarmup_steps": 4}
    factors = [stub._rewarmup_factor(step) for step in range(9)]
    assert factors == [1.0, 1.0, 1.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0]
    stub.token_superposition = {"bag_size": 1, "steps": 0, "rewarmup_steps": 0}
    assert stub._rewarmup_factor(0) == 1.0


def test_rewarmup_scales_one_optimizer_step_and_restores_the_scheduled_lr():
    stub = _ScheduleStub()
    stub.token_superposition = {"bag_size": 6, "steps": 3, "rewarmup_steps": 4}
    weight = torch.nn.Parameter(torch.ones(1))
    stub.optimizer = torch.optim.SGD([weight], lr=0.1)
    weight.grad = torch.ones(1)
    with stub._rewarmed_lr(3):
        stub.optimizer.step()
    assert weight.item() == pytest.approx(1 - 0.1 * 0.25)
    assert stub.optimizer.param_groups[0]["lr"] == 0.1


def _compose(name, overrides=()):
    warnings.filterwarnings("ignore")
    with initialize_config_dir(version_base=None, config_dir=FEATURE_DIR):
        return compose(config_name=name, overrides=list(overrides))


def test_shipped_token_superposition_configs_pass_check_point():
    check_point(_compose("tiny_local_token_superposition"))
    check_point(_compose("h12_token_superposition"))


def test_token_superposition_with_batch_size_warmup_fails_check_point():
    """Superposition steps splice loader batches and warmup steps slice them."""
    cfg = _compose(
        "tiny_local_batch_warmup",
        [
            "trainer.token_superposition.bag_size=4",
            "trainer.token_superposition.steps=10",
        ],
    )
    with pytest.raises(ValueError, match="can't both be on"):
        check_point(cfg)
