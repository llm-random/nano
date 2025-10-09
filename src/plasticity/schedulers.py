import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineScheduler(_LRScheduler):
    """
    Cosine annealing learning rate scheduler.
    Decays learning rate from initial value to final_lr_fraction * base_lr following a cosine curve.
    """

    def __init__(self, optimizer, n_steps, final_lr_fraction=0, warmup_steps=0, last_epoch=-1):
        """
        Args:
            optimizer: Wrapped optimizer
            n_steps: Total number of steps for the schedule
            final_lr_fraction: Final learning rate as a fraction of base_lr (default: 0)
            warmup_steps: Number of warmup steps with linear increase (default: 0)
            last_epoch: The index of last epoch (default: -1)
        """
        self.n_steps = n_steps
        self.final_lr_fraction = final_lr_fraction
        self.warmup_steps = warmup_steps
        self.total_iters = n_steps  # For compatibility with RepeatedScheduler
        super().__init__(optimizer, last_epoch)
        # Compute min_lr as fraction of base_lr
        self.min_lr = [base_lr * final_lr_fraction for base_lr in self.base_lrs]

    def get_lr(self):
        if self.warmup_steps > 0 and self.last_epoch < self.warmup_steps:
            # Warmup phase: linear increase from 0 to base_lr
            alpha = self.last_epoch / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        elif self.last_epoch >= self.n_steps:
            return self.min_lr

        # Cosine annealing formula (adjusted for warmup)
        progress = (self.last_epoch - self.warmup_steps) / (self.n_steps - self.warmup_steps)
        return [
            min_lr + (base_lr - min_lr) *
            (1 + math.cos(math.pi * progress)) / 2
            for base_lr, min_lr in zip(self.base_lrs, self.min_lr)
        ]


class WSDScheduler(_LRScheduler):
    """
    Warmup-Stable-Decay (WSD) scheduler with linear decay.
    - Warmup: Linear increase from 0 to peak_lr
    - Stable: Constant at peak_lr
    - Decay: Linear decrease to final_lr_fraction * base_lr
    """

    def __init__(self, optimizer, warmup_steps, stable_steps, decay_steps,
                 final_lr_fraction=0, n_steps=None, last_epoch=-1):
        """
        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Number of warmup steps
            stable_steps: Number of stable steps at peak learning rate
            decay_steps: Number of decay steps
            final_lr_fraction: Final learning rate as a fraction of base_lr (default: 0)
            n_steps: Total steps (ignored, calculated from warmup+stable+decay) (default: None)
            last_epoch: The index of last epoch (default: -1)
        """
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.n_steps = warmup_steps + stable_steps + decay_steps
        self.final_lr_fraction = final_lr_fraction
        self.total_iters = self.n_steps  # For compatibility with RepeatedScheduler
        super().__init__(optimizer, last_epoch)
        # Compute min_lr as fraction of base_lr
        self.min_lr = [base_lr * final_lr_fraction for base_lr in self.base_lrs]

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup phase: linear increase from 0 to peak_lr
            alpha = self.last_epoch / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]

        elif self.last_epoch < self.warmup_steps + self.stable_steps:
            # Stable phase: constant at peak_lr
            return self.base_lrs

        elif self.last_epoch < self.n_steps:
            # Decay phase: linear decrease from peak_lr to min_lr
            decay_progress = (self.last_epoch - self.warmup_steps - self.stable_steps) / self.decay_steps
            return [
                base_lr - (base_lr - min_lr) * decay_progress
                for base_lr, min_lr in zip(self.base_lrs, self.min_lr)
            ]

        else:
            # After n_steps, stay at min_lr
            return self.min_lr


class RepeatedScheduler:
    """
    Wraps a learning rate scheduler and repeats it for a specified number of cycles.
    After each cycle completes, the scheduler is reset to its initial state.
    Automatically divides the base scheduler's n_steps by num_cycles.
    """

    def __init__(self, base_scheduler, num_cycles, warmup_steps=None, **kwargs):
        """
        Args:
            base_scheduler: A partial scheduler (not yet instantiated) or scheduler instance
            num_cycles: Number of times to repeat the scheduler cycle
            warmup_steps: Warmup steps for the first cycle only (default: None, uses base_scheduler's warmup)
            **kwargs: Additional arguments (e.g., n_steps, optimizer from main.py), passed to base_scheduler if it's a partial
        """
        self.num_cycles = num_cycles

        # If base_scheduler is a partial, we need to instantiate it with modified n_steps
        if hasattr(base_scheduler, 'func'):
            # It's a partial, divide n_steps by num_cycles
            if 'n_steps' in base_scheduler.keywords:
                base_scheduler.keywords['n_steps'] = base_scheduler.keywords['n_steps'] // num_cycles

            # Handle warmup_steps override for first cycle
            if warmup_steps is not None:
                base_scheduler.keywords['warmup_steps'] = warmup_steps

            # Store the warmup value for later
            self.first_cycle_warmup = base_scheduler.keywords.get('warmup_steps', 0)

            # Only pass optimizer from kwargs if it's there
            if 'optimizer' in kwargs:
                base_scheduler.keywords['optimizer'] = kwargs['optimizer']

            # Now instantiate the partial
            base_scheduler = base_scheduler()
        elif hasattr(base_scheduler, 'n_steps'):
            # It's already instantiated, modify n_steps directly
            base_scheduler.n_steps = base_scheduler.n_steps // num_cycles
            base_scheduler.total_iters = base_scheduler.n_steps

            # Handle warmup_steps override
            if warmup_steps is not None and hasattr(base_scheduler, 'warmup_steps'):
                base_scheduler.warmup_steps = warmup_steps

            self.first_cycle_warmup = getattr(base_scheduler, 'warmup_steps', 0)

            # Recompute min_lr if it exists
            if hasattr(base_scheduler, 'final_lr_fraction'):
                base_scheduler.min_lr = [base_lr * base_scheduler.final_lr_fraction
                                         for base_lr in base_scheduler.base_lrs]

        self.base_scheduler = base_scheduler
        self.optimizer = base_scheduler.optimizer

        # Store initial state
        self.initial_state = base_scheduler.state_dict()

        # Track current cycle and total steps
        self.current_cycle = 0
        self.total_steps = 0

        # Get the cycle length from the base scheduler
        # This assumes the scheduler has a total_iters or T_max attribute
        if hasattr(base_scheduler, 'total_iters'):
            self.cycle_length = base_scheduler.total_iters
        elif hasattr(base_scheduler, 'T_max'):
            self.cycle_length = base_scheduler.T_max
        else:
            # Try to infer from milestones for SequentialLR
            if hasattr(base_scheduler, 'milestones') and base_scheduler.milestones:
                self.cycle_length = max(base_scheduler.milestones) + 1
            else:
                raise ValueError("Unable to determine cycle length from scheduler")

    def step(self):
        """Take a step with the scheduler, resetting if cycle is complete"""
        self.base_scheduler.step()
        self.total_steps += 1

        # Check if we've completed a cycle
        if self.total_steps % self.cycle_length == 0:
            self.current_cycle += 1

            # Reset scheduler if we haven't exhausted repeats
            if self.current_cycle < self.num_cycles:
                self.base_scheduler.load_state_dict(self.initial_state)
                # After first cycle, disable warmup for subsequent cycles
                if self.current_cycle == 1 and hasattr(self.base_scheduler, 'warmup_steps'):
                    self.base_scheduler.warmup_steps = 0

    def get_last_lr(self):
        """Get the last computed learning rate"""
        return self.base_scheduler.get_last_lr()

    def state_dict(self):
        """Return state dict for checkpointing"""
        return {
            'base_scheduler': self.base_scheduler.state_dict(),
            'current_cycle': self.current_cycle,
            'total_steps': self.total_steps,
            'cycle_length': self.cycle_length,
            'num_cycles': self.num_cycles,
        }

    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint"""
        self.base_scheduler.load_state_dict(state_dict['base_scheduler'])
        self.current_cycle = state_dict['current_cycle']
        self.total_steps = state_dict['total_steps']
        self.cycle_length = state_dict['cycle_length']
        self.num_cycles = state_dict['num_cycles']