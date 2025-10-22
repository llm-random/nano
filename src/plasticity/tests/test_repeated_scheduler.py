"""Test RepeatedScheduler with various base schedulers"""

import torch
import torch.nn as nn
from functools import partial
from ..schedulers import WSDScheduler, CosineScheduler, RepeatedScheduler


def test_repeated_wsd():
    """Test RepeatedScheduler with WSDScheduler"""
    print("=" * 60)
    print("Testing RepeatedScheduler + WSDScheduler")
    print("=" * 60)

    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Factory function for WSD scheduler
    def wsd_factory(optimizer, n_steps, warmup_steps):
        return WSDScheduler(
            optimizer=optimizer,
            n_steps=n_steps,
            warmup_steps=warmup_steps,
            decay_fraction=0.1,
            final_lr_fraction=0.1,
        )

    scheduler = RepeatedScheduler(
        optimizer=optimizer,
        base_scheduler_factory=wsd_factory,
        num_cycles=3,
        n_steps=1000,
        warmup_steps=100,
    )

    lrs = []
    for step in range(1000):
        lr = scheduler.get_last_lr()[0]
        lrs.append(lr)
        scheduler.step()

    # Check for negative values
    negative = [lr for lr in lrs if lr < 0]
    print(f"Total steps: {len(lrs)}")
    print(f"Negative LRs: {len(negative)}")
    print(f"Min LR: {min(lrs):.6f}, Max LR: {max(lrs):.6f}")

    # Sample some LR values
    print(f"\nSample LRs:")
    for i in [0, 99, 100, 333, 334, 666, 667, 999]:
        if i < len(lrs):
            print(f"  Step {i:4d}: {lrs[i]:.6f}")

    assert len(negative) == 0, f"Found {len(negative)} negative LRs"
    print("✓ PASS\n")


def test_repeated_cosine():
    """Test RepeatedScheduler with CosineScheduler"""
    print("=" * 60)
    print("Testing RepeatedScheduler + CosineScheduler")
    print("=" * 60)

    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Factory function for Cosine scheduler
    def cosine_factory(optimizer, n_steps, warmup_steps):
        return CosineScheduler(
            optimizer=optimizer,
            n_steps=n_steps,
            warmup_steps=warmup_steps,
            final_lr_fraction=0.1,
        )

    scheduler = RepeatedScheduler(
        optimizer=optimizer,
        base_scheduler_factory=cosine_factory,
        num_cycles=3,
        n_steps=1000,
        warmup_steps=100,
    )

    lrs = []
    for step in range(1000):
        lr = scheduler.get_last_lr()[0]
        lrs.append(lr)
        scheduler.step()

    # Check for negative values
    negative = [lr for lr in lrs if lr < 0]
    print(f"Total steps: {len(lrs)}")
    print(f"Negative LRs: {len(negative)}")
    print(f"Min LR: {min(lrs):.6f}, Max LR: {max(lrs):.6f}")

    # Sample some LR values
    print(f"\nSample LRs:")
    for i in [0, 99, 100, 333, 334, 666, 667, 999]:
        if i < len(lrs):
            print(f"  Step {i:4d}: {lrs[i]:.6f}")

    assert len(negative) == 0, f"Found {len(negative)} negative LRs"
    print("✓ PASS\n")


def test_with_lambda():
    """Test RepeatedScheduler with lambda factory"""
    print("=" * 60)
    print("Testing RepeatedScheduler with lambda factory")
    print("=" * 60)

    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Use lambda as factory
    scheduler = RepeatedScheduler(
        optimizer=optimizer,
        base_scheduler_factory=lambda **kw: WSDScheduler(
            decay_fraction=0.1, final_lr_fraction=0.1, **kw
        ),
        num_cycles=3,
        n_steps=1000,
        warmup_steps=100,
    )

    lrs = []
    for step in range(1000):
        lr = scheduler.get_last_lr()[0]
        lrs.append(lr)
        scheduler.step()

    # Check for negative values
    negative = [lr for lr in lrs if lr < 0]
    print(f"Total steps: {len(lrs)}")
    print(f"Negative LRs: {len(negative)}")
    print(f"Min LR: {min(lrs):.6f}, Max LR: {max(lrs):.6f}")

    assert len(negative) == 0, f"Found {len(negative)} negative LRs"
    print("✓ PASS\n")


if __name__ == "__main__":
    test_repeated_wsd()
    test_repeated_cosine()
    test_with_lambda()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
