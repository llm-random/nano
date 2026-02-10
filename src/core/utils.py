import math
import os
import torch
from torch.distributed.tensor import DTensor


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


def cast_state_dict_to_tensors(state_dict, device="cpu"):
    """
    Convert all DTensors in a state dict to regular torch.Tensors.
    By default, gathers them to CPU.
    """
    full_state = {}
    for k, v in state_dict.items():
        if isinstance(v, DTensor):
            full_state[k] = v.full_tensor().float().to(device)
            if os.environ["RANK"] != "0":
                full_state[k] = None
        elif isinstance(v, torch.Tensor):
            full_state[k] = v.to(device)
            if os.environ["RANK"] != "0":
                full_state[k] = None
        else:
            full_state[k] = v
    return full_state


def finish_exp(model):
    return False


def solve_config_lr(
    config_lr: float,
) -> tuple[
    float, float
]:  # TODO temporary place - move to definitions eval+ when created
    ret_lr, ret_exp_lr = None, None
    if config_lr < 1.0:
        ret_lr = config_lr
        ret_exp_lr = -1 * math.log(config_lr) / math.log(2)
    if config_lr >= 1.0:
        ret_lr = 2**-config_lr
        ret_exp_lr = ret_exp_lr
    return ret_lr, ret_exp_lr
