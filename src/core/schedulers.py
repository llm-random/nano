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
    after_warmup_steps = n_steps - warmup_steps - 1
    constant_scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=1.0
    )  # TODO this is only because of a bug in llm-random
    # Factor-based cosine (identical to CosineAnnealingLR with eta_min=fraction*base_lr)
    # so each param group decays to final_lr_fraction of its own base lr.
    cosine_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: final_lr_fraction
        + (1 - final_lr_fraction)
        * 0.5
        * (1 + math.cos(math.pi * min(step / after_warmup_steps, 1.0))),
    )
    training_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, constant_scheduler, cosine_scheduler],
        milestones=[warmup_steps, warmup_steps + 1],
    )
    return training_scheduler
