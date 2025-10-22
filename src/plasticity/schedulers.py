from torch.optim.lr_scheduler import (
    SequentialLR,
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


class CustomLinearLR(_LRScheduler):
    """
    Custom linear LR scheduler that transitions from start_lr to end_lr over a specified number of steps.
    If steps=1, immediately uses end_lr.

    Unlike PyTorch's LinearLR which uses factors, this allows explicit start and end learning rates.
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1, start_lr=None, end_lr=None):
        """
        Args:
            optimizer: Wrapped optimizer
            total_iters: Number of steps for the linear transition
            last_epoch: Current step count, -1 means start from beginning (default: -1)
            start_lr: Starting learning rate (default: None, which means 0.0)
            end_lr: Ending learning rate (default: None, which means base_lr from optimizer)
        """
        self.total_iters = total_iters
        self.start_lr = start_lr if start_lr is not None else 0.0
        super().__init__(optimizer, last_epoch)
        # Resolve end_lr: use base_lr if not specified
        if end_lr is None:
            self.end_lr = self.base_lrs[0]  # Assumes all param groups have same lr
        else:
            self.end_lr = end_lr

    def get_lr(self):
        # Special case: if total_iters is 1, use end_lr immediately
        if self.total_iters == 1:
            return [self.end_lr for _ in self.base_lrs]

        if self.last_epoch == 0:
            return [self.start_lr for _ in self.base_lrs]
        elif self.last_epoch >= self.total_iters - 1:
            return [self.end_lr for _ in self.base_lrs]
        else:
            # Linear interpolation from start_lr to end_lr
            progress = self.last_epoch / (self.total_iters - 1)
            return [
                self.start_lr + (self.end_lr - self.start_lr) * progress
                for _ in self.base_lrs
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
        final_lr_fraction=0.1,
        warmup_fraction=None,
        warmup_steps=None,
        start_lr=None,
        last_epoch=-1,
    ):
        """
        Args:
            optimizer: Wrapped optimizer
            n_steps: Total number of steps for the schedule
            final_lr_fraction: Final learning rate as a fraction of base_lr (default: 0)
            warmup_fraction: Fraction of n_steps for warmup (default: 0.0)
            warmup_steps: Absolute number of warmup steps (optional, used only if warmup_fraction is not specified)
            start_lr: Starting learning rate for warmup (default: None, which means 0.0)
            last_epoch: Current step count, -1 means start from beginning (default: -1)
        """
        self.n_steps = n_steps
        self.final_lr_fraction = final_lr_fraction

        # Resolve warmup parameters
        self.warmup_fraction, self.warmup_steps = _resolve_warmup(
            n_steps, warmup_fraction, warmup_steps
        )

        # Get base learning rate for eta_min calculation
        # this is is a fix for repeated scheduler
        if "initial_lr" in optimizer.param_groups[0]:
            initial_lr = optimizer.param_groups[0]["initial_lr"]
        else:
            initial_lr = optimizer.param_groups[0]["lr"]

        schedulers = []
        milestones = []

        if self.warmup_steps > 0:
            # Warmup phase: linear increase from start_lr (or 0) to base_lr
            warmup_scheduler = CustomLinearLR(
                optimizer,
                total_iters=self.warmup_steps,
                start_lr=start_lr,
                end_lr=initial_lr,
            )
            schedulers.append(warmup_scheduler)
            milestones.append(self.warmup_steps)

        # Cosine annealing phase
        cosine_steps = n_steps - self.warmup_steps
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=final_lr_fraction * initial_lr,
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
        start_lr=None,
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
            start_lr: Starting learning rate for warmup (default: None, which means 0.0)
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

        # Get base learning rate for end_lr calculation
        if "initial_lr" in optimizer.param_groups[0]:
            initial_lr = optimizer.param_groups[0]["initial_lr"]
        else:
            initial_lr = optimizer.param_groups[0]["lr"]

        # Warmup phase: linear increase from start_lr (or 0) to base_lr
        if self.warmup_steps > 0:
            warmup_scheduler = CustomLinearLR(
                optimizer,
                total_iters=self.warmup_steps,
                start_lr=start_lr,
                end_lr=initial_lr,
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
            decay_scheduler = CustomLinearLR(
                optimizer,
                total_iters=self.decay_steps,
                start_lr=initial_lr,
                end_lr=final_lr_fraction * initial_lr,
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

    Warmup behavior across cycles:
    - First cycle: warmup starts from start_lr (default 0.0 if not specified)
    - Subsequent cycles: warmup starts from where the previous cycle ended

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
        final_lr_fraction=None,
        start_lr=None,
        last_epoch=-1,
    ):
        """
        Args:
            optimizer: Wrapped optimizer
            base_scheduler_factory: Scheduler class or factory (partial or lambda) that creates a scheduler
            num_cycles: Number of times to repeat the scheduler
            n_steps: Total number of training steps
            warmup_fraction: Override warmup fraction for first cycle only (default: None, uses base scheduler's warmup_fraction)
            warmup_steps: Override warmup steps for first cycle only (default: None, uses base scheduler's warmup_steps)
            final_lr_fraction: Override final_lr_fraction for the final cycle only (default: None, uses base scheduler's final_lr_fraction)
            start_lr: Starting learning rate for warmup in first cycle (default: None, which means 0.0)
            last_epoch: Current step (default: -1)
        """
        self.num_cycles = num_cycles
        self.n_steps = n_steps
        self.base_scheduler_factory = base_scheduler_factory
        self.final_lr_fraction = final_lr_fraction
        self.start_lr = start_lr  # Store for first cycle
        self.current_total_steps = 0

        # Extract warmup parameters from base scheduler factory (if it's a partial)
        base_warmup_fraction = None
        base_warmup_steps = None
        if hasattr(base_scheduler_factory, "keywords"):
            # It's a functools.partial
            base_warmup_fraction = base_scheduler_factory.keywords.get(
                "warmup_fraction", None
            )
            base_warmup_steps = base_scheduler_factory.keywords.get(
                "warmup_steps", None
            )

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

        # 1st cycle may have more steps
        self.current_cycle_steps = (
            self.cycle_n_steps + self.warmup_steps - self.base_warmup_steps
        )
        # to make sure the whole training has self.n_steps
        self.current_cycle_steps = self.n_steps - (
            (self.num_cycles - 1) * self.cycle_n_steps
        )

        # Create first cycle's base scheduler
        # Override warmup, use longer steps, and pass start_lr
        self.base_scheduler = base_scheduler_factory(
            optimizer=optimizer,
            n_steps=self.current_cycle_steps,
            warmup_steps=self.warmup_steps,
            start_lr=self.start_lr,
        )

        # Store attributes from base scheduler for external access
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
        self.current_total_steps += 1

        # Check if cycle complete

        if self.step_in_cycle >= self.current_cycle_steps:
            # Capture the final LR from this cycle to use as start_lr for next cycle
            cycle_final_lr = self.get_last_lr()[0]  # Get LR from first param group

            # after 1st cycle all cycles have equal number of steps
            self.current_cycle_steps = self.cycle_n_steps
            self.current_cycle += 1
            self.step_in_cycle = 0

            # Reset for next cycle (if not last)
            if self.current_cycle < self.num_cycles:
                # Check if this is the final cycle and we have a final_lr_fraction override
                is_final_cycle = self.current_cycle == self.num_cycles - 1

                # Try passing warmup_steps, start_lr, and final_lr_fraction (works for lambda factories)
                kwargs = {
                    "optimizer": self.optimizer,
                    "n_steps": self.cycle_n_steps,
                    "warmup_steps": self.base_warmup_steps,
                    "start_lr": cycle_final_lr,  # Start next cycle from where this one ended
                }

                # Override final_lr_fraction for the final cycle if specified
                if is_final_cycle and self.final_lr_fraction is not None:
                    kwargs["final_lr_fraction"] = self.final_lr_fraction
                    # Calculate remaining steps for final cycle using the running step counter
                    kwargs["n_steps"] = self.n_steps - self.current_total_steps

                self.base_scheduler = self.base_scheduler_factory(**kwargs)

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
