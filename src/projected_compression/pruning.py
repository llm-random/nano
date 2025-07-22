import torch

def generate_structured_prune_mask(weights: torch.Tensor, prune_n: int, prune_m: int) -> torch.Tensor:
    """
    Generates a binary, structured pruning mask for weight matrix W
    Args:
        W (torch.Tensor): Original weight tensor (2D).
        prune_n (int): Number of least important weights to prune per block.
        prune_m (int): Width of block.
    Returns:
        torch.BoolTensor: Mask with `True` at pruned positions.
    """
    assert weights.ndim == 2, "Expected matching 2D tensors"
    output_dim, input_dim = weights.shape
    assert prune_n > 0
    assert prune_m > 0
    assert input_dim % prune_m == 0, "input_dim must be divisible by prune_m"

    abs_weights = torch.abs(weights)

    num_blocks = input_dim // prune_m
    blocks = abs_weights.view(output_dim, num_blocks, prune_m)

    # Get top-k indices per block (least important `prune_n`)
    _, idx = torch.topk(blocks, prune_n, dim=2, largest=False)

    block_mask = torch.zeros_like(blocks, dtype=torch.bool)
    block_mask.scatter_(2, idx, True)

    W_mask = block_mask.view(output_dim, -1)

    return W_mask


def generate_unstructured_prune_mask(weights: torch.Tensor, sparsity_ratio: float) -> torch.Tensor:
    assert 0 < sparsity_ratio < 1, "sparsity_ratio must be between 0 and 1"

    device = weights.device
    abs_weights = torch.abs(weights)
    flat_sorted = torch.sort(abs_weights.flatten().to(device))[0]
    threshold_index = int(abs_weights.numel() * sparsity_ratio)
    threshold = flat_sorted[threshold_index].cpu()
    W_mask = (abs_weights <= threshold)

    return W_mask