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

def print_state_dict_info(state_dict): # used for debugging + log info
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            print(f"{name:60s} {type(param)}, shape={tuple(param.shape)} "
            # print(f"{name:60s} {param}, shape={tuple(param.shape)} "
                f"norm={param.norm().item():.4f}")
        else:
            # Sometimes buffers / metadata can be non-tensors
            print(f"{name:60s} NON-TENSOR {type(param)}")
