"""The memory-efficient chunked distillation loss must be numerically identical
to the naive full-vocab reference, for both loss values and gradients."""

import pytest
import torch

from src.core.trainer_distillation import TrainerDistillation


def _make_trainer(temperature, num_chunks):
    # Bypass the heavy attrs __init__; the loss methods only read these two fields.
    trainer = TrainerDistillation.__new__(TrainerDistillation)
    trainer.distillation_temperature = temperature
    trainer.distillation_loss_num_chunks = num_chunks
    return trainer


def _random_inputs(num_tokens, vocab, seed):
    gen = torch.Generator().manual_seed(seed)
    student = torch.randn(num_tokens, vocab, dtype=torch.float64, generator=gen)
    teacher = torch.randn(num_tokens, vocab, dtype=torch.float64, generator=gen)
    targets = torch.randint(0, vocab, (num_tokens,), generator=gen)
    return student, teacher, targets


@pytest.mark.parametrize("temperature", [1.0, 2.0])
@pytest.mark.parametrize("num_chunks", [1, 3, 7, 100])
@pytest.mark.parametrize("num_tokens", [64, 65])  # 65 -> chunks don't divide evenly
def test_chunked_matches_naive_values_and_grads(temperature, num_chunks, num_tokens):
    vocab = 37
    student, teacher, targets = _random_inputs(num_tokens, vocab, seed=0)

    naive_trainer = _make_trainer(temperature, num_chunks=1)
    chunked_trainer = _make_trainer(temperature, num_chunks=num_chunks)

    # Reference
    s_naive = student.clone().requires_grad_(True)
    ce_naive, distill_naive = naive_trainer._ce_distill_losses_naive(
        s_naive, teacher, targets
    )
    (ce_naive + distill_naive).backward()

    # Chunked (checkpointed)
    s_chunked = student.clone().requires_grad_(True)
    ce_chunked, distill_chunked = chunked_trainer._ce_distill_losses_chunked(
        s_chunked, teacher, targets
    )
    (ce_chunked + distill_chunked).backward()

    assert torch.allclose(ce_naive, ce_chunked, rtol=0, atol=1e-10)
    assert torch.allclose(distill_naive, distill_chunked, rtol=0, atol=1e-10)
    assert torch.allclose(s_naive.grad, s_chunked.grad, rtol=0, atol=1e-10)


def test_num_chunks_one_is_bitwise_close_to_naive():
    # num_chunks=1 is the same computation reorganized; should match extremely tightly.
    student, teacher, targets = _random_inputs(48, 29, seed=1)
    trainer = _make_trainer(temperature=1.5, num_chunks=1)

    ce_naive, distill_naive = trainer._ce_distill_losses_naive(
        student, teacher, targets
    )
    ce_chunked, distill_chunked = trainer._ce_distill_losses_chunked(
        student, teacher, targets
    )
    assert torch.allclose(ce_naive, ce_chunked, rtol=0, atol=1e-12)
    assert torch.allclose(distill_naive, distill_chunked, rtol=0, atol=1e-12)
