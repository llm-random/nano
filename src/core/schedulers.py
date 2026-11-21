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


class CosineAnnealingToFraction(torch.optim.lr_scheduler.CosineAnnealingLR):
    """CosineAnnealingLR with one floor per param group: `final_lr_fraction` of
    that group's own base lr, where torch's class takes one `eta_min` for all.

    Groups at different learning rates need it -- the AME banks run at
    trainer.bank_lr_ratio x the backbone's (src.core.model.parameter_groups).
    With one absolute floor a group at 4x would anneal down to the backbone's
    floor and lose its ratio over the run; with a floor per group every group
    keeps its ratio at every step (exactly, for a power-of-two ratio).

    The formulas are torch 2.7.1's `CosineAnnealingLR.get_lr` and
    `_get_closed_form_lr`, copied term for term with `eta_min` indexed per
    group. With one learning rate across groups each floor is the old scalar,
    computed by the same multiplication, so the schedule is bit-identical to
    the torch class -- tests/test_bank_lr_ratio.py pins that, and will catch a
    torch upgrade that changes the upstream formulas.
    """

    def __init__(self, optimizer, T_max, final_lr_fraction, last_epoch=-1):
        # Before super().__init__: its initial step already calls get_lr.
        self.final_lr_fraction = final_lr_fraction
        super().__init__(optimizer, T_max=T_max, eta_min=0.0, last_epoch=last_epoch)

    def _floors(self):
        return [self.final_lr_fraction * base_lr for base_lr in self.base_lrs]

    def get_lr(self):
        groups = self.optimizer.param_groups
        floors = self._floors()
        if self.last_epoch == 0:
            return [group["lr"] for group in groups]
        elif self._step_count == 1 and self.last_epoch > 0:
            return [
                eta_min
                + (base_lr - eta_min)
                * (1 + math.cos((self.last_epoch) * math.pi / self.T_max))
                / 2
                for base_lr, eta_min in zip(self.base_lrs, floors)
            ]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group["lr"]
                + (base_lr - eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, eta_min, group in zip(self.base_lrs, floors, groups)
            ]
        return [
            (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max))
            * (group["lr"] - eta_min)
            + eta_min
            for group, eta_min in zip(groups, floors)
        ]

    def _get_closed_form_lr(self):
        return [
            eta_min
            + (base_lr - eta_min)
            * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / 2
            for base_lr, eta_min in zip(self.base_lrs, self._floors())
        ]


def get_cosine_scheduler_with_warmup(
    optimizer, warmup_steps: int, n_steps: int, final_lr_fraction: float
):
    # Param groups may run at different learning rates (AME's banks at
    # trainer.bank_lr_ratio x the rest). Warmup and the constant step are
    # multiplicative, so they keep every ratio as it is; the cosine floors each
    # group at final_lr_fraction of its OWN lr (CosineAnnealingToFraction), so
    # it keeps them too. With one lr across groups this is the schedule it
    # always was, bit for bit.
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
    cosine_scheduler = CosineAnnealingToFraction(
        optimizer,
        T_max=after_warmup_steps,
        final_lr_fraction=final_lr_fraction,
    )
    training_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, constant_scheduler, cosine_scheduler],
        milestones=[warmup_steps, warmup_steps + 1],
    )
    return training_scheduler
