import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR


class WSDScheduler(SequentialLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        n_steps: int,
        warmup_steps: int,
        decay_steps: int,
    ):
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.constant_steps = n_steps - warmup_steps - decay_steps

        # Define individual schedulers
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-12,  # can't use 0.0
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        constant_scheduler = ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=self.constant_steps,
        )

        linear_decay_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=decay_steps,
        )

        schedulers = [warmup_scheduler, constant_scheduler, linear_decay_scheduler]
        milestones = [warmup_steps, warmup_steps + self.constant_steps]

        super(WSDScheduler, self).__init__(
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
    assert (
        len(optimizer.param_groups) == 1
    ), "Cosine scheduler only supports one param group"
    optimizer_lr = optimizer.param_groups[0][
        "lr"
    ]  # param_groups changes when applying scheduler
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
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=after_warmup_steps,
        eta_min=final_lr_fraction * optimizer_lr,
    )
    training_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, constant_scheduler, cosine_scheduler],
        milestones=[warmup_steps, warmup_steps + 1],
    )
    return training_scheduler
