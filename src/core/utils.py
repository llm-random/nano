import math


def create_batch_fingerprint(batch):
    def prefix_suffix_only(array, prefix=3, suffix=3):
        prefix_part = array[:prefix]
        suffix_part = array[-suffix:]
        result = prefix_part + suffix_part
        return result

    first_row = prefix_suffix_only(batch[0]).numpy().tolist()
    middle_row = prefix_suffix_only(batch[len(batch) // 2]).numpy().tolist()
    last_row = prefix_suffix_only(batch[-1]).numpy().tolist()

    return first_row + middle_row + last_row


def solve_config_lr(
    config_lr: float,
) -> tuple[
    float, float
]:  # TODO temporary place - move to devinitions eval+ when created
    ret_lr, ret_exp_lr = None, None
    if config_lr < 1.0:
        ret_lr = config_lr
        ret_exp_lr = -1 * math.log(config_lr) / math.log(2)
    if config_lr >= 1.0:
        # By convention a config learning_rate of n means 2**-n, so n *is* the exponent.
        # This used to read `ret_exp_lr = ret_exp_lr`, leaving it None on this branch --
        # which is the branch every config using the convention takes, so wandb recorded
        # exp_lr: None for those runs.
        ret_lr = 2**-config_lr
        ret_exp_lr = config_lr
    return ret_lr, ret_exp_lr
