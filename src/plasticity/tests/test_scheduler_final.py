"""Final verification: ensure no negative values across full range"""

import torch
import torch.nn as nn
from ..schedulers import WSDScheduler, RepeatedScheduler

# Create a dummy model and optimizer
model = nn.Linear(10, 10)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Test multiple configurations
configs = [
    {"n_steps": 1000, "num_cycles": 3, "warmup_steps": 100},
    {"n_steps": 1000, "num_cycles": 5, "warmup_steps": 50},
    {"n_steps": 500, "num_cycles": 2, "warmup_steps": 100},
]

print("Testing RepeatedScheduler + WSDScheduler for negative values...\n")

for i, config in enumerate(configs):
    print(
        f"=== Test {i+1}: n_steps={config['n_steps']}, cycles={config['num_cycles']}, warmup={config['warmup_steps']} ==="
    )

    # Create fresh optimizer for each test
    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Create repeated scheduler with WSD
    scheduler = RepeatedScheduler(
        optimizer=optimizer,
        base_scheduler_factory=lambda **kw: WSDScheduler(
            decay_fraction=0.1, final_lr_fraction=0.1, **kw
        ),
        num_cycles=config["num_cycles"],
        n_steps=config["n_steps"],
        warmup_steps=config["warmup_steps"],
    )

    # Track learning rates
    lrs = []
    min_lr = float("inf")
    max_lr = float("-inf")
    negative_count = 0

    # Simulate training
    for step in range(config["n_steps"]):
        lr = scheduler.get_last_lr()[0]
        lrs.append(lr)
        min_lr = min(min_lr, lr)
        max_lr = max(max_lr, lr)

        if lr < 0:
            negative_count += 1
            print(f"  ERROR: Negative LR at step {step}: {lr}")

        scheduler.step()

    # Results
    if negative_count == 0:
        print(f"  ✓ PASS: No negative values found")
    else:
        print(f"  ✗ FAIL: {negative_count} negative values found")

    print(f"  Min LR: {min_lr:.6f}, Max LR: {max_lr:.6f}")
    print()

print("All tests completed!")
