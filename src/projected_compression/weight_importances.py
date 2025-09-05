import torch
import torch.nn as nn
import logging
import os

from main import get_device
from src.core.checkpointing import get_full_checkpoint_path

device = get_device()
logger = logging.getLogger(__name__)


def _calculate_activations_dimension_importances(model: nn.Module, calibration_data, dmodel, dff, n_blocks, device="cuda"):
    """
    Calculate importance of each neuron (dmodel and dff) using forward hooks.
    """
    dmodel_importances = torch.zeros(dmodel, device=device)
    dff_importances = torch.zeros(n_blocks, dff, device=device)

    handles = []

    # --- Hook functions ---
    def hook_dmodel_pre_attn(layer, inp, out):
        nonlocal dmodel_importances
        # inp[0] has shape [batch, seq, dmodel]
        dmodel_importances += torch.sum(torch.abs(out.detach()), dim=[0, 1])

    def hook_dmodel_pre_ff(layer, inp, out):
        nonlocal dmodel_importances
        dmodel_importances += torch.sum(torch.abs(out.detach()), dim=[0, 1])

    def hook_ff_pre_act(layer, inp, out, block_idx=None):
        nonlocal dff_importances
        dff_importances[block_idx] += torch.sum(torch.abs(out.detach()), dim=[0, 1])

    # --- Register hooks ---
    for block_idx, block in enumerate(model.encoder.blocks):
        # normalized pre-attention
        handles.append(block.attention_layer.norm.register_forward_hook(hook_dmodel_pre_attn))

        # normalized pre-ff
        handles.append(block.ff_layer.norm.register_forward_hook(hook_dmodel_pre_ff))

        # ff_pre_act with block index captured
        handles.append(
            block.ff_layer.layer.ff_pre_act.register_forward_hook(
                lambda layer, inp, out, idx=block_idx: hook_ff_pre_act(layer, inp, out, idx)
            )
        )

    # --- Run calibration data ---
    with torch.no_grad():
        for i, batch in enumerate(calibration_data):
            logger.info(f"Beginning batch {i}")
            _ = model(batch.to(device))

    # cleanup
    for h in handles:
        h.remove()

    return dmodel_importances, dff_importances


def _calculate_dummy_dimension_importances(dmodel, dff, n_blocks):
    dmodel_importances = torch.range(0, dmodel-1)
    dff_importances = torch.zeros(n_blocks, dff)
    dff_importances[:,] = torch.range(0, dff-1)

    return dmodel_importances, dff_importances


def minitron_importances(model: nn.Module, dataloader, dmodel, dff, calibration_dataset_size, seq_len, total_batch_size, n_blocks, checkpoint_save_path):
    logger.info(f"Calculating minitron style weight importances calculation.")

    calibration_data = torch.zeros(calibration_dataset_size // total_batch_size, total_batch_size, seq_len, dtype=torch.long, device=device)
    for i, batch in enumerate(dataloader):
        if i * total_batch_size >= calibration_dataset_size:
            break
        calibration_data[i] = batch[:, :seq_len]

    dmodel_importances, dff_importances = _calculate_activations_dimension_importances(
        model, calibration_data, dmodel, dff, n_blocks
    )
    logger.info(f"Calculated dimensions importances")

    dict_to_save = {"dmodel_importances": dmodel_importances, "dff_importances": dff_importances}
    path = get_full_checkpoint_path(checkpoint_save_path) + "/minitron_dimensions_importances.pt"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict_to_save, path)
    logger.info(f"Saved importances to {path}.")

    return dict_to_save


def dummy_importances(model: nn.Module, dmodel, dff, n_blocks, checkpoint_save_path):
    
    logger.info(f"Calculating minitron style weight importances calculation.")

    dmodel_importances, dff_importances = _calculate_dummy_dimension_importances(
        dmodel, dff, n_blocks
    )
    logger.info(f"Calculated dimensions importances")

    dict_to_save = {"dmodel_importances": dmodel_importances, "dff_importances": dff_importances}
    path = get_full_checkpoint_path(checkpoint_save_path) + "/random_dimensions_importances.pt"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict_to_save, path)

    logger.info(f"Saved importances to {path}.")

    return dict_to_save
