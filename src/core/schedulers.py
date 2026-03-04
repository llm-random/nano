import math
import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR


class TrapezoidalLR(SequentialLR):
    def __init__(
        self,
        optimizer,
        warmup_steps,
        constant_steps,
        decay_steps,
    ):
        self.warmup_steps = warmup_steps
        self.constant_steps = constant_steps
        self.decay_steps = decay_steps

        # Define individual schedulers
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        constant_scheduler = ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=constant_steps,
        )

        linear_decay_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=decay_steps,
        )

        schedulers = [warmup_scheduler, constant_scheduler, linear_decay_scheduler]
        milestones = [warmup_steps, warmup_steps + constant_steps]

        super(TrapezoidalLR, self).__init__(
            optimizer, schedulers=schedulers, milestones=milestones
        )

    def load_state_dict(self, loaded_state):

        if loaded_state["last_epoch"] < self.last_epoch:
            raise RuntimeError(
                "Loaded scheduler checkpoint should have more steps than current one."
            )

        """
        It is a workaround for the problem with loading state_dict of different schedulers.
        The problem is that the state_dict of SequentialLR is not obvious. State of particular scheduler changes after each step. 
        But when it is milestone step, both schedulers are updated. It is easier to just load state and then step to the last_epoch.
        """
        while loaded_state["last_epoch"] > self.last_epoch:
            self.step()


def get_cosine_scheduler_with_warmup(
    optimizer, warmup_steps: int, n_steps: int, final_lr_fraction: float
):
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    # Cosine scheduler phase starts after warmup and 1 constant step
    cosine_start_step = warmup_steps + 1
    T_max = n_steps - cosine_start_step

    def cosine_lambda(step):
        # Calculate progress t within the cosine phase
        if step < cosine_start_step:
            return 1.0
        t = step - cosine_start_step
        if t >= T_max:
            return final_lr_fraction
        # Decay from 1.0 to final_lr_fraction
        return final_lr_fraction + 0.5 * (1 - final_lr_fraction) * (
            1 + math.cos(math.pi * t / T_max)
        )

    constant_scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=1.0, total_iters=1
    )
    # Use LambdaLR to allow different base/min LRs per parameter group (proportional decay)
    cosine_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=cosine_lambda
    )

    training_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, constant_scheduler, cosine_scheduler],
        milestones=[warmup_steps, warmup_steps + 1],
    )
    return training_scheduler
