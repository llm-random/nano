from torch.optim.lr_scheduler import (
    SequentialLR,
    LinearLR,
    CosineAnnealingLR,
    ConstantLR,
    _LRScheduler,
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
                base_lr * self.last_epoch / self.total_iters for base_lr in self.base_lrs
            ]


class CosineScheduler(SequentialLR):
    """
    Cosine annealing learning rate scheduler with optional warmup.
    Decays learning rate from initial value to final_lr_fraction * base_lr following a cosine curve.

    Uses SequentialLR to compose LinearWarmupLR and CosineAnnealingLR schedulers.
    """

    def __init__(
        self, optimizer, n_steps, final_lr_fraction=0, warmup_fraction=0.0, warmup_steps=None, last_epoch=-1
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

        # If warmup_fraction is specified (not default), use it; otherwise use warmup_steps
        if warmup_fraction is not None:
            self.warmup_fraction = warmup_fraction
            self.warmup_steps = int(n_steps * warmup_fraction)
        else:
            self.warmup_steps = warmup_steps
            self.warmup_fraction = warmup_steps / n_steps if n_steps > 0 else 0.0

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
        warmup_fraction=0.0,
        decay_fraction=0.0,
        final_lr_fraction=0,
        warmup_steps=None,
        last_epoch=-1,
    ):
        """
        Args:
            optimizer: Wrapped optimizer
            n_steps: Total number of training steps
            warmup_fraction: Fraction of n_steps for warmup (default: 0.0)
            decay_fraction: Fraction of n_steps for decay (default: 0.0)
            final_lr_fraction: Final learning rate as a fraction of base_lr (default: 0)
            warmup_steps: Absolute number of warmup steps (optional, used only if warmup_fraction is not specified)
            last_epoch: Current step count, -1 means start from beginning (default: -1)
        """
        self.n_steps = n_steps
        self.decay_fraction = decay_fraction
        self.final_lr_fraction = final_lr_fraction

        # If warmup_fraction is specified (not default), use it; otherwise use warmup_steps
        if warmup_fraction is not None:
            self.warmup_fraction = warmup_fraction
            self.warmup_steps = int(n_steps * warmup_fraction)
        else:
            self.warmup_steps = warmup_steps
            self.warmup_fraction = warmup_steps / n_steps if n_steps > 0 else 0.0


        self.decay_steps = int(n_steps * decay_fraction)
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


class RepeatedScheduler(_LRScheduler):
    """
    Repeats a base scheduler for multiple cycles.
    After each cycle completes, the scheduler resets to its initial state.

    Allows overriding the first cycle's warmup with a longer warmup_steps.
    Subsequent cycles use the base scheduler's own warmup_fraction.

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
        warmup_steps = None,
        last_epoch=-1,
        **base_scheduler_kwargs
    ):
        """
        Args:
            optimizer: Wrapped optimizer
            base_scheduler_factory: Scheduler class or factory that creates a scheduler
            num_cycles: Number of times to repeat the scheduler
            n_steps: Total number of training steps
            warmup_fraction: Override warmup fraction for first cycle only (default: None, uses base scheduler's warmup_fraction)
            last_epoch: Current step (default: -1)
            **base_scheduler_kwargs: Arguments to pass to base_scheduler_factory (e.g., warmup_fraction, decay_fraction)
        """
        self.num_cycles = num_cycles
        self.n_steps = n_steps
        self.base_scheduler_kwargs = base_scheduler_kwargs

        # Get base scheduler's warmup_fraction
        base_warmup_fraction = base_scheduler_kwargs.get('warmup_fraction', 0.0)

        # Use base warmup_fraction if repeated warmup_fraction is not specified
        self.warmup_fraction = warmup_fraction if warmup_fraction is not None else base_warmup_fraction

        # Calculate cycle steps: (n_steps - repeated_warmup + base_warmup) // num_cycles
        # This accounts for replacing base warmup with repeated warmup in first cycle
        base_warmup_steps = int((n_steps / num_cycles) * base_warmup_fraction)
        warmup_steps = int((n_steps / num_cycles) * self.warmup_fraction)
        cycle_steps = (n_steps - warmup_steps + base_warmup_steps) // num_cycles

        # Create base scheduler for first cycle
        first_cycle_kwargs = dict(base_scheduler_kwargs)
        if self.warmup_fraction != base_warmup_fraction:
            # Override base scheduler's warmup with RepeatedScheduler's warmup for first cycle
            first_cycle_kwargs['warmup_fraction'] = self.warmup_fraction

        self.base_scheduler = base_scheduler_factory(
            optimizer=optimizer,
            n_steps=cycle_steps + warmup_steps - base_warmup_steps,
            **first_cycle_kwargs
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
        self.base_scheduler_factory = base_scheduler_factory
        self.base_warmup_fraction = base_warmup_fraction
        self.warmup_steps = warmup_steps
        self.base_warmup_steps = base_warmup_steps

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
        # First cycle length is adjusted for the custom warmup
        first_cycle_length = self.cycle_steps + self.warmup_steps - self.base_warmup_steps

        cycle_length = first_cycle_length if self.current_cycle == 0 else self.cycle_steps

        if self.step_in_cycle >= cycle_length:
            self.current_cycle += 1
            self.step_in_cycle = 0

            # Reset for next cycle (if not last)
            if self.current_cycle < self.num_cycles:
                # Reset base scheduler with its original warmup_fraction
                self.base_scheduler = self.base_scheduler_factory(
                    optimizer=self.optimizer,
                    n_steps=self.cycle_steps,
                    **self.base_scheduler_kwargs
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
