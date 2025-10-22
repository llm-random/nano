"""Comprehensive test to verify RepeatedScheduler behavior"""

import torch
import torch.nn as nn
from functools import partial
from ..schedulers import WSDScheduler, CosineScheduler, RepeatedScheduler


def test_cycle_boundaries():
    """Verify that cycles reset at the correct boundaries"""
    print("=" * 60)
    print("Testing Cycle Boundaries")
    print("=" * 60)

    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Simple WSD: 300 total steps, 3 cycles, no warmup in base
    scheduler = RepeatedScheduler(
        optimizer=optimizer,
        base_scheduler_factory=lambda **kw: WSDScheduler(
            decay_fraction=0.1, final_lr_fraction=0.1, **kw
        ),
        num_cycles=3,
        n_steps=300,
        warmup_steps=0,
    )

    print(f"Total n_steps: 300")
    print(f"Num cycles: 3")
    print(f"Expected cycle length: 100")
    print(f"Actual cycle length: {scheduler.cycle_steps}")
    print()

    lrs = []
    for step in range(300):
        lr = scheduler.get_last_lr()[0]
        lrs.append(lr)
        scheduler.step()

    # Check cycle resets
    print("Learning rates at cycle boundaries:")
    print(f"  Step 0 (start cycle 1): {lrs[0]:.6f}")
    print(f"  Step 99 (end cycle 1): {lrs[99]:.6f}")
    print(f"  Step 100 (start cycle 2): {lrs[100]:.6f}")
    print(f"  Step 199 (end cycle 2): {lrs[199]:.6f}")
    print(f"  Step 200 (start cycle 3): {lrs[200]:.6f}")
    print(f"  Step 299 (end cycle 3): {lrs[299]:.6f}")
    print()

    # Verify cycles reset (LR should jump back to peak after decay)
    assert (
        abs(lrs[100] - 1e-3) < 1e-6
    ), f"Cycle 2 should start at peak LR, got {lrs[100]}"
    assert (
        abs(lrs[200] - 1e-3) < 1e-6
    ), f"Cycle 3 should start at peak LR, got {lrs[200]}"
    print("✓ PASS: Cycles reset correctly\n")


def test_warmup_override():
    """Test that warmup_steps override works correctly"""
    print("=" * 60)
    print("Testing Warmup Override")
    print("=" * 60)

    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # WSD scheduler with warmup override
    scheduler = RepeatedScheduler(
        optimizer=optimizer,
        base_scheduler_factory=lambda **kw: WSDScheduler(
            decay_fraction=0.1, final_lr_fraction=0.1, **kw
        ),
        num_cycles=3,
        n_steps=1000,
        warmup_steps=50,
    )

    print(f"RepeatedScheduler warmup_steps: 50")
    print(f"Expected first cycle warmup: 50 steps")
    print(f"Actual warmup: {scheduler.warmup_steps}")
    print()

    lrs = []
    for step in range(1000):
        lr = scheduler.get_last_lr()[0]
        lrs.append(lr)
        scheduler.step()

    print("Learning rates during first warmup:")
    print(f"  Step 0: {lrs[0]:.6f}")
    print(f"  Step 25: {lrs[25]:.6f}")
    print(f"  Step 49: {lrs[49]:.6f}")
    print(f"  Step 50: {lrs[50]:.6f} (should be at peak)")
    print()

    # Verify warmup is 50 steps
    assert abs(lrs[50] - 1e-3) < 1e-6, f"Should reach peak at step 50, got {lrs[50]}"
    print("✓ PASS: Warmup override works correctly\n")


def test_no_warmup_after_first_cycle():
    """Verify that subsequent cycles have no warmup"""
    print("=" * 60)
    print("Testing No Warmup After First Cycle")
    print("=" * 60)

    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    scheduler = RepeatedScheduler(
        optimizer=optimizer,
        base_scheduler_factory=lambda **kw: WSDScheduler(
            decay_fraction=0.1, final_lr_fraction=0.1, **kw
        ),
        num_cycles=3,
        n_steps=900,
        warmup_steps=100,  # Only first cycle
    )

    lrs = []
    for step in range(900):
        lr = scheduler.get_last_lr()[0]
        lrs.append(lr)
        scheduler.step()

    print("Learning rates at start of each cycle:")
    print(f"  Cycle 1, Step 0: {lrs[0]:.6f} (warmup)")
    print(f"  Cycle 2, Step 300: {lrs[300]:.6f} (should be peak)")
    print(f"  Cycle 3, Step 600: {lrs[600]:.6f} (should be peak)")
    print()

    # Verify no warmup after first cycle
    assert abs(lrs[300] - 1e-3) < 1e-6, f"Cycle 2 should start at peak, got {lrs[300]}"
    assert abs(lrs[600] - 1e-3) < 1e-6, f"Cycle 3 should start at peak, got {lrs[600]}"
    print("✓ PASS: Subsequent cycles have no warmup\n")


def test_cosine_scheduler():
    """Test RepeatedScheduler with CosineScheduler"""
    print("=" * 60)
    print("Testing with CosineScheduler")
    print("=" * 60)

    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    scheduler = RepeatedScheduler(
        optimizer=optimizer,
        base_scheduler_factory=lambda **kw: CosineScheduler(
            final_lr_fraction=0.1, **kw
        ),
        num_cycles=3,
        n_steps=600,
        warmup_steps=0,
    )

    lrs = []
    for step in range(600):
        lr = scheduler.get_last_lr()[0]
        lrs.append(lr)
        scheduler.step()

    print(f"Cycle length: {scheduler.cycle_steps}")
    print(f"Learning rates at cycle boundaries:")
    print(f"  Step 0: {lrs[0]:.6f}")
    print(f"  Step 199: {lrs[199]:.6f}")
    print(f"  Step 200: {lrs[200]:.6f}")
    print(f"  Step 399: {lrs[399]:.6f}")
    print(f"  Step 400: {lrs[400]:.6f}")
    print(f"  Step 599: {lrs[599]:.6f}")
    print()

    # Verify cosine behavior
    assert min(lrs) >= 0, "No negative learning rates"
    assert abs(lrs[200] - 1e-3) < 1e-6, f"Cycle 2 should reset to peak, got {lrs[200]}"
    print("✓ PASS: CosineScheduler works correctly\n")


if __name__ == "__main__":
    test_cycle_boundaries()
    test_warmup_override()
    test_no_warmup_after_first_cycle()
    test_cosine_scheduler()
    print("=" * 60)
    print("All comprehensive tests passed!")
    print("=" * 60)
