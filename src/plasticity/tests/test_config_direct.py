"""Direct test of schedulers without config interpolation"""

import torch
import torch.nn as nn
from ..schedulers import WSDScheduler, CosineScheduler, RepeatedScheduler


def test_repeated_wsd_direct():
    """Test repeated WSD scheduler directly"""
    print("=" * 70)
    print("Testing Repeated WSD (Direct)")
    print("=" * 70)

    # Create dummy model and optimizer
    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    n_steps = 300
    num_cycles = 3
    warmup_steps = int(n_steps * 0.1)  # 30 steps

    # Create scheduler as config would
    scheduler = RepeatedScheduler(
        optimizer=optimizer,
        base_scheduler_factory=lambda **kw: WSDScheduler(
            decay_fraction=0.1, final_lr_fraction=0.1, **kw
        ),
        num_cycles=num_cycles,
        n_steps=n_steps,
        warmup_steps=warmup_steps,
    )

    print(f"✓ Scheduler created successfully!")
    print(f"  Total steps: {scheduler.n_steps}")
    print(f"  Num cycles: {scheduler.num_cycles}")
    print(f"  Warmup steps: {scheduler.warmup_steps}")
    print(f"  Cycle steps: {scheduler.cycle_steps}")
    print()

    # Run and check for negative LRs
    lrs = []
    negative_count = 0

    for step in range(n_steps):
        lr = scheduler.get_last_lr()[0]
        lrs.append(lr)

        if lr < 0:
            negative_count += 1
            print(f"  ❌ ERROR: Negative LR at step {step}: {lr:.6f}")

        scheduler.step()

    # Results
    if negative_count == 0:
        print(f"  ✓ PASS: No negative LR values found")
    else:
        print(f"  ✗ FAIL: {negative_count} negative LR values found")

    print(f"  Min LR: {min(lrs):.6f}")
    print(f"  Max LR: {max(lrs):.6f}")

    # Show some sample LRs
    print(f"\n  Sample LR values:")
    for i in [0, 30, 60, 100, 150, 200, 250, 299]:
        if i < len(lrs):
            print(f"    Step {i:3d}: {lrs[i]:.6f}")

    # Check cycles reset properly
    print(f"\n  Checking cycle resets:")
    cycle_len = scheduler.cycle_steps
    for cycle in range(num_cycles):
        if cycle > 0:  # After first cycle
            step_idx = cycle * cycle_len
            if step_idx < len(lrs):
                print(
                    f"    Cycle {cycle+1} start (step {step_idx}): {lrs[step_idx]:.6f} (should be ~0.001)"
                )

    print()
    return negative_count == 0


def test_repeated_cosine_direct():
    """Test repeated Cosine scheduler directly"""
    print("=" * 70)
    print("Testing Repeated Cosine (Direct)")
    print("=" * 70)

    # Create dummy model and optimizer
    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    n_steps = 250
    num_cycles = 5
    warmup_steps = int(n_steps * 0.01)  # 2 steps

    # Create scheduler as config would
    scheduler = RepeatedScheduler(
        optimizer=optimizer,
        base_scheduler_factory=lambda **kw: CosineScheduler(final_lr_fraction=0.1, **kw),
        num_cycles=num_cycles,
        n_steps=n_steps,
        warmup_steps=warmup_steps,
    )

    print(f"✓ Scheduler created successfully!")
    print(f"  Total steps: {scheduler.n_steps}")
    print(f"  Num cycles: {scheduler.num_cycles}")
    print(f"  Warmup steps: {scheduler.warmup_steps}")
    print(f"  Cycle steps: {scheduler.cycle_steps}")
    print()

    # Run and check for negative LRs
    lrs = []
    negative_count = 0

    for step in range(n_steps):
        lr = scheduler.get_last_lr()[0]
        lrs.append(lr)

        if lr < 0:
            negative_count += 1
            print(f"  ❌ ERROR: Negative LR at step {step}: {lr:.6f}")

        scheduler.step()

    # Results
    if negative_count == 0:
        print(f"  ✓ PASS: No negative LR values found")
    else:
        print(f"  ✗ FAIL: {negative_count} negative LR values found")

    print(f"  Min LR: {min(lrs):.6f}")
    print(f"  Max LR: {max(lrs):.6f}")

    # Show some sample LRs
    print(f"\n  Sample LR values:")
    for i in [0, 25, 50, 100, 150, 200, 249]:
        if i < len(lrs):
            print(f"    Step {i:3d}: {lrs[i]:.6f}")

    # Check that cosine is actually varying (not flat)
    lr_range = max(lrs) - min(lrs)
    print(f"\n  LR range (max - min): {lr_range:.6f}")

    if lr_range > 1e-4:
        print(f"  ✓ Cosine schedule is varying as expected")
    else:
        print(f"  ⚠ Warning: LR appears flat")

    # Check cycles reset properly
    print(f"\n  Checking cycle resets:")
    cycle_len = scheduler.cycle_steps
    for cycle in range(min(3, num_cycles)):  # Show first 3 cycles
        if cycle > 0:  # After first cycle
            step_idx = cycle * cycle_len
            if step_idx < len(lrs):
                print(
                    f"    Cycle {cycle+1} start (step {step_idx}): {lrs[step_idx]:.6f} (should be ~0.001)"
                )

    print()
    return negative_count == 0


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DIRECT SCHEDULER TESTS (Config-equivalent)")
    print("=" * 70 + "\n")

    wsd_pass = test_repeated_wsd_direct()
    cosine_pass = test_repeated_cosine_direct()

    print("=" * 70)
    if wsd_pass and cosine_pass:
        print("✓ ALL TESTS PASSED!")
        print("  Configs should work correctly with these same parameters")
    else:
        print("✗ SOME TESTS FAILED")
        if not wsd_pass:
            print("  - Repeated WSD: FAILED")
        if not cosine_pass:
            print("  - Repeated Cosine: FAILED")
    print("=" * 70)
