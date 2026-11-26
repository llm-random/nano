import math

import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR


class WSDScheduler(SequentialLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        n_steps: int,
        warmup_steps: int,
        decay_steps: int,
        final_lr_fraction: float,
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

        schedulers = [warmup_scheduler, constant_scheduler]
        milestones = [warmup_steps]

        # Skip the decay branch when decay_steps=0: LinearLR(total_iters=0) divides by zero.
        if decay_steps > 0:
            # Relative to each group's own peak LR, unlike cosine's absolute eta_min.
            linear_decay_scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=final_lr_fraction,
                total_iters=decay_steps,
            )
            schedulers.append(linear_decay_scheduler)
            milestones.append(warmup_steps + self.constant_steps)

        # A milestone at step 0 never fires, so drop phases that are empty at the start.
        while milestones and milestones[0] == 0:
            schedulers.pop(0)
            milestones.pop(0)

        super(WSDScheduler, self).__init__(
            optimizer, schedulers=schedulers, milestones=milestones
        )

    def state_dict(self):
        # Just the step count: load_state_dict replays to it, whatever the config.
        return {"last_epoch": self.last_epoch}

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
        # Optimizer state was loaded first: replay from our own LR, not the saved one.
        for group, lr in zip(self.optimizer.param_groups, self.get_last_lr()):
            group["lr"] = lr
        while loaded_state["last_epoch"] > self.last_epoch:
            self.step()


class PerGroupCosineAnnealingLR(torch.optim.lr_scheduler.CosineAnnealingLR):
    """CosineAnnealingLR with one eta_min per param group.

    get_lr and _get_closed_form_lr are torch 2.7.1's with eta_min indexed by group,
    so equal eta_mins give bit-identical LRs to the stock scheduler.
    """

    def __init__(self, optimizer, T_max, eta_mins):
        self.eta_mins = eta_mins
        super().__init__(optimizer, T_max=T_max, eta_min=eta_mins[0])

    def get_lr(self):
        groups = self.optimizer.param_groups
        if self.last_epoch == 0:
            return [group["lr"] for group in groups]
        elif self._step_count == 1 and self.last_epoch > 0:
            return [
                eta_min
                + (base_lr - eta_min)
                * (1 + math.cos((self.last_epoch) * math.pi / self.T_max))
                / 2
                for eta_min, base_lr in zip(self.eta_mins, self.base_lrs)
            ]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group["lr"]
                + (base_lr - eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for eta_min, base_lr, group in zip(self.eta_mins, self.base_lrs, groups)
            ]
        return [
            (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max))
            * (group["lr"] - eta_min)
            + eta_min
            for eta_min, group in zip(self.eta_mins, groups)
        ]

    def _get_closed_form_lr(self):
        return [
            eta_min
            + (base_lr - eta_min)
            * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / 2
            for eta_min, base_lr in zip(self.eta_mins, self.base_lrs)
        ]


def get_cosine_scheduler_with_warmup(
    optimizer, warmup_steps: int, n_steps: int, final_lr_fraction: float
):
    # Each group decays to final_lr_fraction of its own LR (simpleP's groups differ);
    # read before the warmup scheduler changes param_groups.
    eta_mins = [final_lr_fraction * group["lr"] for group in optimizer.param_groups]
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    after_warmup_steps = n_steps - warmup_steps - 1
    # n_steps == 0 is legitimate: the conversion and importances configs build a
    # scheduler they never step. Only guard the case that does get stepped, where
    # T_max == 0 would otherwise surface as a bare ZeroDivisionError on step 1.
    assert n_steps == 0 or after_warmup_steps >= 1, (
        f"n_steps={n_steps} must be greater than warmup_steps+1={warmup_steps + 1}: the "
        f"cosine phase would get T_max={after_warmup_steps}, and T_max=0 raises a bare "
        "ZeroDivisionError on the first scheduler step. Easy to hit when shrinking n_steps "
        "for a smoke test -- lower warmup_steps as well."
    )
    constant_scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=1.0
    )  # TODO this is only because of a bug in llm-random
    cosine_scheduler = PerGroupCosineAnnealingLR(
        optimizer, T_max=after_warmup_steps, eta_mins=eta_mins
    )
    training_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, constant_scheduler, cosine_scheduler],
        milestones=[warmup_steps, warmup_steps + 1],
    )
    return training_scheduler
