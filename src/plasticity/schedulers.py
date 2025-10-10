import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineScheduler(_LRScheduler):
    """
    Cosine annealing learning rate scheduler.
    Decays learning rate from initial value to final_lr_fraction * base_lr following a cosine curve.

    Note: last_epoch in PyTorch's _LRScheduler is actually the step count (poor naming).
    """

    def __init__(self, optimizer, n_steps, final_lr_fraction=0, warmup_steps=0, last_epoch=-1):
        """
        Args:
            optimizer: Wrapped optimizer
            n_steps: Total number of steps for the schedule
            final_lr_fraction: Final learning rate as a fraction of base_lr (default: 0)
            warmup_steps: Number of warmup steps with linear increase (default: 0)
            last_epoch: Current step count, -1 means start from beginning (default: -1)
        """
        self.n_steps = n_steps
        self.final_lr_fraction = final_lr_fraction
        self.warmup_steps = warmup_steps
        # Initialize min_lr before super().__init__() which calls get_lr()
        # We'll use optimizer.defaults['lr'] as a fallback for base_lr
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.min_lr = [base_lr * final_lr_fraction for base_lr in base_lrs]
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate based on current step (self.last_epoch is actually step count)."""
        step = self.last_epoch

        if self.warmup_steps > 0 and step < self.warmup_steps:
            # Warmup: linear increase from 0 to base_lr
            alpha = step / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        elif step >= self.n_steps:
            return self.min_lr

        # Cosine annealing (adjusted for warmup)
        progress = (step - self.warmup_steps) / (self.n_steps - self.warmup_steps)
        return [
            min_lr + (base_lr - min_lr) * (1 + math.cos(math.pi * progress)) / 2
            for base_lr, min_lr in zip(self.base_lrs, self.min_lr)
        ]


class WSDScheduler(_LRScheduler):
    """
    Warmup-Stable-Decay (WSD) scheduler with linear decay.
    - Warmup: Linear increase from 0 to peak_lr
    - Stable: Constant at peak_lr
    - Decay: Linear decrease to final_lr_fraction * base_lr

    Note: last_epoch in PyTorch's _LRScheduler is actually the step count (poor naming).
    """

    def __init__(
        self,
        optimizer,
        n_steps,
        warmup_fraction=0.1,
        decay_fraction=0.1,
        final_lr_fraction=0,
        warmup_steps=None,
        decay_steps=None,
        last_epoch=-1,
    ):
        """
        Args:
            optimizer: Wrapped optimizer
            n_steps: Total number of training steps
            warmup_fraction: Fraction of n_steps for warmup (default: 0.1)
            decay_fraction: Fraction of n_steps for decay (default: 0.1)
            final_lr_fraction: Final learning rate as a fraction of base_lr (default: 0)
            warmup_steps: Override warmup_fraction with explicit step count (default: None)
            decay_steps: Override decay_fraction with explicit step count (default: None)
            last_epoch: Current step count, -1 means start from beginning (default: -1)
        """
        # Allow explicit step counts to override fractions
        self.n_steps = n_steps
        self.warmup_steps = (
            warmup_steps if warmup_steps is not None else int(n_steps * warmup_fraction)
        )
        self.decay_steps = decay_steps if decay_steps is not None else int(n_steps * decay_fraction)
        self.stable_steps = n_steps - self.warmup_steps - self.decay_steps

        self.final_lr_fraction = final_lr_fraction
        super().__init__(optimizer, last_epoch)
        # Compute min_lr as fraction of base_lr
        self.min_lr = [base_lr * final_lr_fraction for base_lr in self.base_lrs]

    def get_lr(self):
        """Calculate learning rate based on current step (self.last_epoch is actually step count)."""
        step = self.last_epoch

        if step < self.warmup_steps:
            # Warmup: linear increase from 0 to peak_lr
            alpha = step / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]

        elif step < self.warmup_steps + self.stable_steps:
            # Stable: constant at peak_lr
            return self.base_lrs

        elif step < self.n_steps:
            # Decay: linear decrease from peak_lr to min_lr
            decay_progress = (step - self.warmup_steps - self.stable_steps) / self.decay_steps
            return [
                base_lr - (base_lr - min_lr) * decay_progress
                for base_lr, min_lr in zip(self.base_lrs, self.min_lr)
            ]

        else:
            # After n_steps, stay at min_lr
            return self.min_lr


class RepeatedScheduler(_LRScheduler):
    """
    Repeats a base scheduler for multiple cycles.
    After each cycle completes, the scheduler resets to its initial state.

    First cycle can have warmup, subsequent cycles start at peak LR.
    """

    def __init__(
        self, optimizer, base_scheduler_factory, num_cycles, n_steps, warmup_steps=0, last_epoch=-1
    ):
        """
        Args:
            optimizer: Wrapped optimizer
            base_scheduler_factory: Function/partial that creates a scheduler given (optimizer, n_steps, warmup_steps)
            num_cycles: Number of times to repeat the scheduler
            n_steps: Total number of training steps
            warmup_steps: Warmup steps for first cycle only (default: 0)
            last_epoch: Current step (default: -1)
        """
        self.num_cycles = num_cycles
        self.n_steps = n_steps
        self.warmup_steps = warmup_steps

        # Calculate steps per cycle
        cycle_steps = (n_steps - warmup_steps) // num_cycles

        # Create base scheduler for first cycle with warmup
        self.base_scheduler = base_scheduler_factory(
            optimizer=optimizer, n_steps=cycle_steps, warmup_steps=warmup_steps
        )

        # Store attributes from base scheduler
        if hasattr(self.base_scheduler, "final_lr_fraction"):
            self.final_lr_fraction = self.base_scheduler.final_lr_fraction
        if hasattr(self.base_scheduler, "decay_steps"):
            self.decay_steps = self.base_scheduler.decay_steps
        if hasattr(self.base_scheduler, "stable_steps"):
            self.stable_steps = self.base_scheduler.stable_steps

        self.cycle_steps = cycle_steps
        self.current_cycle = 0
        self.step_in_cycle = 0

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Delegate to base scheduler"""
        return self.base_scheduler.get_lr()

    def get_last_lr(self):
        """Delegate to base scheduler"""
        return self.base_scheduler.get_last_lr()

    def step(self, epoch=None):
        """Step the scheduler, resetting at cycle boundaries"""
        self.base_scheduler.step()
        self.step_in_cycle += 1

        # Check if cycle complete
        if self.step_in_cycle >= self.cycle_steps:
            self.current_cycle += 1
            self.step_in_cycle = 0

            # Reset for next cycle (if not last)
            if self.current_cycle < self.num_cycles:
                # Reset base scheduler with no warmup for subsequent cycles
                self.base_scheduler.last_epoch = -1
                self.base_scheduler.__init__(
                    self.optimizer,
                    n_steps=self.cycle_steps,
                    warmup_steps=0,
                    **{
                        k: v
                        for k, v in self.base_scheduler.__dict__.items()
                        if k in ["final_lr_fraction", "decay_fraction", "warmup_fraction"]
                    }
                )

        self.last_epoch = self.base_scheduler.last_epoch

    def state_dict(self):
        """Return state for checkpointing"""
        return {
            "base_scheduler": self.base_scheduler.state_dict(),
            "current_cycle": self.current_cycle,
            "step_in_cycle": self.step_in_cycle,
        }

    def load_state_dict(self, state_dict):
        """Load state from checkpoint"""
        self.base_scheduler.load_state_dict(state_dict["base_scheduler"])
        self.current_cycle = state_dict["current_cycle"]
        self.step_in_cycle = state_dict["step_in_cycle"]
