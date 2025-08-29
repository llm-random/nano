import torch
import torch.nn as nn
import json
import logging
import os

from src.core.checkpointing import get_full_checkpoint_path

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_dimension_importances(model: nn.Module, calibration_data, dmodel, dff, n_blocks):
    """
    Calculate the importance of each neuron in the model based on the calibration data.
    Returns a list of importance scores for dmodel and dff dimensions.
    """
    dmodel_importance = torch.zeros(dmodel, device=device)
    dff_importance = torch.zeros(n_blocks, dff, device=device)

    # Forward pass through the model with calibration data
    with torch.no_grad():
        for batch in calibration_data:
            x = model.embedding(batch)
            for layer_number, layer in enumerate(model.encoder.blocks):
                y = layer.attention_layer.norm(x) # normalized_pre_attn
                dmodel_importance += torch.sum(torch.abs(y), dim=[0, 1])  # Sum across batch and sequence dimensions

                y = layer.attention_layer.layer(y) # attention_output
                x = x + y  # Residual connection


                y = layer.ff_layer.norm(x) # normalized_pre_ff
                dmodel_importance += torch.sum(torch.abs(y), dim=[0, 1])  # Sum across batch and sequence dimensions

                ff_layer = layer.ff_layer.layer
                ff_gated = ff_layer.silu(ff_layer.gate(y))

                y = ff_layer.ff_pre_act(y) # ff_pre_act
                dff_importance[layer_number] += torch.sum(torch.abs(y), dim=[0, 1])

                y = ff_layer.ff_post_act(y * ff_gated) # ff_output

                x = x + y  # Residual connection
            x = model.head(x)

            assert torch.allclose(x, model(batch)), "Model output does not match expected output"

    return dmodel_importance, dff_importance

def prune(model: nn.Module, dmodel_indices, dff_indices, target_dmodel):
    """
    Prune the model by selecting the top k neurons based on their importance scores.
    """
    # Embedding
    embedding_weight = model.embedding.embedding.weight.data
    model.embedding.embedding.weight.data = embedding_weight[:, dmodel_indices]

    # Head
    head_weight = model.head.linear.weight.data
    model.head.linear.weight.data = head_weight[:, dmodel_indices]
    model.head.norm.weight.data = model.head.norm.weight[dmodel_indices]
    model.head.norm.normalized_shape = tuple([target_dmodel])

    for layer, dff_indices_per_layer in zip(model.encoder.blocks, dff_indices):
        layer.attention_layer.norm.weight.data = layer.attention_layer.norm.weight[dmodel_indices]
        layer.attention_layer.norm.normalized_shape = tuple([target_dmodel])
        layer.attention_layer.layer.q_proj.weight.data = layer.attention_layer.layer.q_proj.weight[:, dmodel_indices]
        layer.attention_layer.layer.k_proj.weight.data = layer.attention_layer.layer.k_proj.weight[:, dmodel_indices]
        layer.attention_layer.layer.v_proj.weight.data = layer.attention_layer.layer.v_proj.weight[:, dmodel_indices]
        layer.attention_layer.layer.o_proj.weight.data = layer.attention_layer.layer.o_proj.weight[dmodel_indices, :]

        layer.ff_layer.norm.weight.data = layer.ff_layer.norm.weight[dmodel_indices]
        layer.ff_layer.norm.normalized_shape = tuple([target_dmodel])
        layer.ff_layer.layer.ff_pre_act.weight.data = layer.ff_layer.layer.ff_pre_act.weight[:, dmodel_indices][dff_indices_per_layer, :]
        layer.ff_layer.layer.gate.weight.data = layer.ff_layer.layer.gate.weight[:, dmodel_indices][dff_indices_per_layer, :]
        layer.ff_layer.layer.ff_post_act.weight.data = layer.ff_layer.layer.ff_post_act.weight[dmodel_indices, :][:, dff_indices_per_layer]

    return model

def minitron_prune(model: nn.Module, dataloader, dmodel, target_dmodel, dff, target_dff, calibration_dataset_size, seq_len, total_batch_size, n_blocks, checkpoint_save_path):

    calibration_data = torch.zeros(calibration_dataset_size // total_batch_size, total_batch_size, seq_len, dtype=torch.long, device=device)
    for i, batch in enumerate(dataloader):
        if i * total_batch_size >= calibration_dataset_size:
            break
        calibration_data[i] = batch[:, :seq_len]

    dmodel_importance, dff_importance = calculate_dimension_importances(
        model, calibration_data, dmodel, dff, n_blocks
    )

    logger.debug("Importance dimensions calculated.")

    dmodel_top_indices = torch.topk(dmodel_importance, dim=0, largest=True, k=target_dmodel).indices.tolist()

    dff_top_indices = []
    for i in range(n_blocks):
        dff_top_indices_current = torch.topk(dff_importance[i], dim=0, largest=True, k=target_dff).indices.tolist()
        dff_top_indices.append(dff_top_indices_current)

    # save to file indices as dict
    dict_to_save = {"dmodel_top_indices": dmodel_top_indices, "dff_top_indices": dff_top_indices}
    path = get_full_checkpoint_path(checkpoint_save_path) + "/top_indices.json"

    # check if path exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(dict_to_save, f)

    model = prune(model, dmodel_top_indices, dff_top_indices, target_dmodel)

    logger.info("Model pruned.")

    return model

