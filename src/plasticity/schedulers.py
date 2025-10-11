from torch.optim.lr_scheduler import (
    SequentialLR,
    LinearLR,
    CosineAnnealingLR,
    ConstantLR,
    _LRScheduler,
)


def _resolve_warmup(n_steps, warmup_fraction=None, warmup_steps=None):
    """
    Resolve warmup parameters with priority: warmup_fraction > warmup_steps.

    Args:
        n_steps: Total number of steps
        warmup_fraction: Fraction of n_steps for warmup (optional)
        warmup_steps: Absolute number of warmup steps (optional)

    Returns:
        tuple: (warmup_fraction, warmup_steps)
    """
    if warmup_fraction is not None:
        # warmup_fraction has priority
        return warmup_fraction, int(n_steps * warmup_fraction)
    elif warmup_steps is not None:
        # Use warmup_steps if warmup_fraction not specified
        return warmup_steps / n_steps if n_steps > 0 else 0.0, warmup_steps
    else:
        # Both are None, default to no warmup
        return 0.0, 0


def _resolve_repeated_warmup(
    n_steps,
    num_cycles,
    repeated_warmup_fraction=None,
    repeated_warmup_steps=None,
    base_warmup_fraction=None,
    base_warmup_steps=None,
):
    """
    Resolve warmup for RepeatedScheduler with 4-level priority.

    Priority (highest to lowest):
    1. repeated_warmup_fraction
    2. repeated_warmup_steps
    3. base_warmup_fraction
    4. base_warmup_steps
    """

    warmup_fraction = None
    warmup_steps = None

    if (repeated_warmup_fraction is not None) or (repeated_warmup_steps is not None):
        warmup_fraction, warmup_steps = _resolve_warmup(
            n_steps=n_steps,
            warmup_fraction=repeated_warmup_fraction,
            warmup_steps=repeated_warmup_steps,
        )

        if base_warmup_fraction is not None:
            cycle_n_steps = (
                n_steps * (1 - warmup_fraction + base_warmup_fraction)
            ) // num_cycles

        elif base_warmup_steps is not None:
            cycle_n_steps = (n_steps - warmup_steps + base_warmup_steps) // num_cycles

        else:
            cycle_n_steps = (n_steps - warmup_steps) // num_cycles
    else:
        cycle_n_steps = n_steps // num_cycles

    base_warmup_fraction, base_warmup_steps = _resolve_warmup(
        cycle_n_steps, base_warmup_fraction, base_warmup_steps
    )

    if warmup_fraction is None:
        warmup_fraction, warmup_steps = base_warmup_fraction, base_warmup_steps

    return (
        warmup_fraction,
        warmup_steps,
        cycle_n_steps,
        base_warmup_fraction,
        base_warmup_steps,
    )


class LinearWarmupLR(_LRScheduler):
    """
    Linear warmup scheduler that starts from 0 (not just near 0).
    PyTorch's LinearLR doesn't allow start_factor=0, so we need a custom implementation.
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return [0.0 for _ in self.base_lrs]
        elif self.last_epoch >= self.total_iters:
            return self.base_lrs
        else:
            return [
                base_lr * self.last_epoch / self.total_iters
                for base_lr in self.base_lrs
            ]


class CosineScheduler(SequentialLR):
    """
    Cosine annealing learning rate scheduler with optional warmup.
    Decays learning rate from initial value to final_lr_fraction * base_lr following a cosine curve.

    Uses SequentialLR to compose LinearWarmupLR and CosineAnnealingLR schedulers.
    """

    def __init__(
        self,
        optimizer,
        n_steps,
        final_lr_fraction=0,
        warmup_fraction=0.0,
        warmup_steps=None,
        last_epoch=-1,
    ):
        """
        Args:
            optimizer: Wrapped optimizer
            n_steps: Total number of steps for the schedule
            final_lr_fraction: Final learning rate as a fraction of base_lr (default: 0)
            warmup_fraction: Fraction of n_steps for warmup (default: 0.0)
            warmup_steps: Absolute number of warmup steps (optional, used only if warmup_fraction is not specified)
            last_epoch: Current step count, -1 means start from beginning (default: -1)
        """
        self.n_steps = n_steps
        self.final_lr_fraction = final_lr_fraction

        # Resolve warmup parameters
        self.warmup_fraction, self.warmup_steps = _resolve_warmup(
            n_steps, warmup_fraction, warmup_steps
        )

        # Get base learning rate for eta_min calculation
        optimizer_lr = optimizer.param_groups[0]["lr"]

        schedulers = []
        milestones = []

        if self.warmup_steps > 0:
            # Warmup phase: linear increase from 0 to base_lr
            warmup_scheduler = LinearWarmupLR(
                optimizer,
                total_iters=self.warmup_steps,
            )
            schedulers.append(warmup_scheduler)
            milestones.append(self.warmup_steps)

        # Cosine annealing phase
        cosine_steps = n_steps - self.warmup_steps
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=final_lr_fraction * optimizer_lr,
        )
        schedulers.append(cosine_scheduler)

        super().__init__(
            optimizer,
            schedulers=schedulers,
            milestones=milestones,
            last_epoch=last_epoch,
        )

    def step(self, epoch=None):
        """Override step to ignore epoch parameter for compatibility with nested SequentialLR"""
        super().step()

    def print_config(self):
        """Print scheduler configuration for debugging"""
        print(f"CosineScheduler Configuration:")
        print(f"  n_steps: {self.n_steps}")
        print(f"  warmup_fraction: {self.warmup_fraction}")
        print(f"  warmup_steps: {self.warmup_steps}")
        print(f"  cosine_steps: {self.n_steps - self.warmup_steps}")
        print(f"  final_lr_fraction: {self.final_lr_fraction}")


class WSDScheduler(SequentialLR):
    """
    Warmup-Stable-Decay (WSD) scheduler with linear decay.
    - Warmup: Linear increase from 0 to peak_lr
    - Stable: Constant at peak_lr
    - Decay: Linear decrease to final_lr_fraction * base_lr

    Uses SequentialLR to compose LinearWarmupLR (warmup), ConstantLR (stable), and LinearLR (decay) schedulers.
    """

    def __init__(
        self,
        optimizer,
        n_steps,
        warmup_fraction=None,
        decay_fraction=None,
        final_lr_fraction=0,
        warmup_steps=None,
        decay_steps=None,
        last_epoch=-1,
    ):
        """
        Args:
            optimizer: Wrapped optimizer
            n_steps: Total number of training steps
            warmup_fraction: Fraction of n_steps for warmup (optional, default: None)
            decay_fraction: Fraction of n_steps for decay (optional, default: None)
            final_lr_fraction: Final learning rate as a fraction of base_lr (default: 0)
            warmup_steps: Absolute number of warmup steps (optional, used only if warmup_fraction is not specified)
            decay_steps: Absolute number of decay steps (optional, used only if decay_fraction is not specified)
            last_epoch: Current step count, -1 means start from beginning (default: -1)
        """
        self.n_steps = n_steps
        self.final_lr_fraction = final_lr_fraction

        # Resolve warmup parameters
        self.warmup_fraction, self.warmup_steps = _resolve_warmup(
            n_steps, warmup_fraction, warmup_steps
        )

        # Resolve decay parameters
        self.decay_fraction, self.decay_steps = _resolve_warmup(
            n_steps, decay_fraction, decay_steps
        )

        self.stable_steps = n_steps - self.warmup_steps - self.decay_steps

        schedulers = []
        milestones = []
        current_step = 0

        # Warmup phase: linear increase from 0 to base_lr
        if self.warmup_steps > 0:
            warmup_scheduler = LinearWarmupLR(
                optimizer,
                total_iters=self.warmup_steps,
            )
            schedulers.append(warmup_scheduler)
            current_step += self.warmup_steps

        # Stable phase: constant at peak_lr
        if self.stable_steps > 0:
            if schedulers:  # Only add milestone if there's a previous scheduler
                milestones.append(current_step)
            stable_scheduler = ConstantLR(
                optimizer,
                factor=1.0,
                total_iters=self.stable_steps,
            )
            schedulers.append(stable_scheduler)
            current_step += self.stable_steps

        # Decay phase: linear decrease to final_lr_fraction
        if self.decay_steps > 0:
            if schedulers:  # Only add milestone if there's a previous scheduler
                milestones.append(current_step)
            decay_scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=final_lr_fraction,
                total_iters=self.decay_steps,
            )
            schedulers.append(decay_scheduler)

        super().__init__(
            optimizer,
            schedulers=schedulers,
            milestones=milestones,
            last_epoch=last_epoch,
        )

    def step(self, epoch=None):
        """Override step to ignore epoch parameter for compatibility with nested SequentialLR"""
        super().step()

    def print_config(self):
        """Print scheduler configuration for debugging"""
        print(f"WSDScheduler Configuration:")
        print(f"  n_steps: {self.n_steps}")
        print(f"  warmup_fraction: {self.warmup_fraction}")
        print(f"  warmup_steps: {self.warmup_steps}")
        print(f"  stable_steps: {self.stable_steps}")
        print(f"  decay_fraction: {self.decay_fraction}")
        print(f"  decay_steps: {self.decay_steps}")
        print(f"  final_lr_fraction: {self.final_lr_fraction}")


class RepeatedScheduler(_LRScheduler):
    """
    Repeats a base scheduler for multiple cycles.
    After each cycle completes, the scheduler resets to its initial state.

    Allows overriding the first cycle's warmup with custom warmup_fraction or warmup_steps.
    Subsequent cycles use the base scheduler's own warmup parameters.

    Warmup priority (highest to lowest):
    1. RepeatedScheduler.warmup_fraction
    2. RepeatedScheduler.warmup_steps
    3. base_scheduler.warmup_fraction (from base_scheduler_kwargs)
    4. base_scheduler.warmup_steps (from base_scheduler_kwargs)

    Note: This uses a delegation pattern rather than SequentialLR because we need
    to reset the base scheduler at cycle boundaries, which SequentialLR doesn't support
    (SequentialLR schedulers continue from the optimizer's current LR, not the base LR).
    """

    def __init__(
        self,
        optimizer,
        base_scheduler_factory,
        num_cycles,
        n_steps,
        warmup_fraction=None,
        warmup_steps=None,
        last_epoch=-1,
        **base_scheduler_kwargs,
    ):
        """
        Args:
            optimizer: Wrapped optimizer
            base_scheduler_factory: Scheduler class or factory that creates a scheduler
            num_cycles: Number of times to repeat the scheduler
            n_steps: Total number of training steps
            warmup_fraction: Override warmup fraction for first cycle only (default: None, uses base scheduler's warmup_fraction)
            warmup_steps: Override warmup steps for first cycle only (default: None, uses base scheduler's warmup_steps)
            last_epoch: Current step (default: -1)
            **base_scheduler_kwargs: Arguments to pass to base_scheduler_factory (e.g., warmup_fraction, decay_fraction)
        """
        self.num_cycles = num_cycles
        self.n_steps = n_steps
        self.base_scheduler_kwargs = base_scheduler_kwargs
        self.base_scheduler_factory = base_scheduler_factory

        # Extract warmup parameters from base scheduler kwargs
        base_warmup_fraction = base_scheduler_kwargs.get("warmup_fraction", None)
        base_warmup_steps = base_scheduler_kwargs.get("warmup_steps", None)

        # Resolve warmup for first cycle with 4-level priority
        (
            self.warmup_fraction,
            self.warmup_steps,
            self.cycle_n_steps,
            self.base_warmup_fraction,
            self.base_warmup_steps,
        ) = _resolve_repeated_warmup(
            n_steps,
            num_cycles,
            warmup_fraction,
            warmup_steps,
            base_warmup_fraction,
            base_warmup_steps,
        )

        # Calculate cycle parameters
        self.cycle_steps = (
            self.cycle_n_steps + self.warmup_steps - self.base_warmup_steps
        )

        # Create first cycle's base scheduler with resolved warmup
        first_cycle_kwargs = dict(base_scheduler_kwargs)
        first_cycle_kwargs["warmup_fraction"] = self.warmup_fraction
        first_cycle_kwargs["warmup_steps"] = None  # Use fraction instead

        self.base_scheduler = base_scheduler_factory(
            optimizer=optimizer, n_steps=self.cycle_steps, **first_cycle_kwargs
        )

        # Store attributes from base scheduler for external access
        if hasattr(self.base_scheduler, "final_lr_fraction"):
            self.final_lr_fraction = self.base_scheduler.final_lr_fraction
        if hasattr(self.base_scheduler, "decay_steps"):
            self.decay_steps = self.base_scheduler.decay_steps
        if hasattr(self.base_scheduler, "stable_steps"):
            self.stable_steps = self.base_scheduler.stable_steps

        # Initialize cycle tracking
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
                # Reset base scheduler with its original warmup_fraction
                self.base_scheduler = self.base_scheduler_factory(
                    optimizer=self.optimizer,
                    n_steps=self.cycle_n_steps,
                    **self.base_scheduler_kwargs,
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

    def print_config(self):
        """Print scheduler configuration for debugging"""
        print(f"RepeatedScheduler Configuration:")
        print(f"  n_steps: {self.n_steps}")
        print(f"  num_cycles: {self.num_cycles}")
        print(f"  current_cycle: {self.current_cycle}")
        print(f"  step_in_cycle: {self.step_in_cycle}")
        print(f"  First cycle warmup:")
        print(f"    warmup_fraction: {self.warmup_fraction}")
        print(f"    warmup_steps: {self.warmup_steps}")
        print(f"  Subsequent cycles warmup:")
        print(f"    base_warmup_fraction: {self.base_warmup_fraction}")
        print(f"    base_warmup_steps: {self.base_warmup_steps}")
        print(f"  Cycle parameters:")
        print(f"    cycle_n_steps: {self.cycle_n_steps}")
        print(f"    cycle_steps: {self.cycle_steps}")
        if hasattr(self, "final_lr_fraction"):
            print(f"  final_lr_fraction: {self.final_lr_fraction}")
        if hasattr(self, "decay_steps"):
            print(f"  decay_steps: {self.decay_steps}")
        if hasattr(self, "stable_steps"):
            print(f"  stable_steps: {self.stable_steps}")
