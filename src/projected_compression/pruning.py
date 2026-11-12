import torch
import torch.nn as nn
import logging

from src.projected_compression.channel_selection import describe, load_selection
from src.projected_compression.channel_selectors import model_dims

logger = logging.getLogger(__name__)


def prune_with_indices(model: nn.Module, dmodel_indices, dff_indices):
    """Keep only the given channels: `dmodel_indices` [target_dmodel] everywhere the
    residual stream is indexed, `dff_indices[l]` [target_dff] in block l's FFN."""
    # Follow the model rather than main.get_device(): apply_functions runs after main.py
    # has already placed the model (see compression.py:init_compression).
    device = next(model.parameters()).device
    dmodel_indices = torch.as_tensor(dmodel_indices, device=device)
    dff_indices = [torch.as_tensor(idx, device=device) for idx in dff_indices]
    assert len(dff_indices) == len(model.encoder.blocks)
    target_dmodel = len(dmodel_indices)

    # Embedding
    embedding_weight = model.embedding.weight.data
    model.embedding.weight.data = embedding_weight[:, dmodel_indices]

    # Head
    head_weight = model.head.linear.weight.data
    model.head.linear.weight.data = head_weight[:, dmodel_indices]
    model.head.norm.weight.data = model.head.norm.weight[dmodel_indices]
    model.head.norm.normalized_shape = tuple([target_dmodel])

    for layer, dff_indices_per_layer in zip(model.encoder.blocks, dff_indices):
        layer.attention_layer.norm.weight.data = layer.attention_layer.norm.weight[
            dmodel_indices
        ]
        layer.attention_layer.norm.normalized_shape = tuple([target_dmodel])
        layer.attention_layer.layer.q_proj.weight.data = (
            layer.attention_layer.layer.q_proj.weight[:, dmodel_indices]
        )
        layer.attention_layer.layer.k_proj.weight.data = (
            layer.attention_layer.layer.k_proj.weight[:, dmodel_indices]
        )
        layer.attention_layer.layer.v_proj.weight.data = (
            layer.attention_layer.layer.v_proj.weight[:, dmodel_indices]
        )
        layer.attention_layer.layer.o_proj.weight.data = (
            layer.attention_layer.layer.o_proj.weight[dmodel_indices, :]
        )

        layer.ff_layer.norm.weight.data = layer.ff_layer.norm.weight[dmodel_indices]
        layer.ff_layer.norm.normalized_shape = tuple([target_dmodel])
        layer.ff_layer.layer.ff_pre_act.weight.data = (
            layer.ff_layer.layer.ff_pre_act.weight[:, dmodel_indices][
                dff_indices_per_layer, :
            ]
        )
        if hasattr(layer.ff_layer.layer, "gate"):
            layer.ff_layer.layer.gate.weight.data = layer.ff_layer.layer.gate.weight[
                :, dmodel_indices
            ][dff_indices_per_layer, :]
        layer.ff_layer.layer.ff_post_act.weight.data = (
            layer.ff_layer.layer.ff_post_act.weight[dmodel_indices, :][
                :, dff_indices_per_layer
            ]
        )

    _sync_size_attributes(model)
    logger.info("Model pruned.")
    return model


def _sync_size_attributes(model: nn.Module):
    """Make the size attributes of Linear / Embedding modules match their (pruned)
    weights; slicing `weight.data` alone leaves them at the original sizes."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.out_features, module.in_features = module.weight.shape
        elif isinstance(module, nn.Embedding):
            module.num_embeddings, module.embedding_dim = module.weight.shape


def prune(model: nn.Module, dimensions_importances_path, target_dmodel, target_dff):
    """Keep the top-`target` channels of an importances file (hard pruning, HP)."""
    dimensions_importances = torch.load(dimensions_importances_path)
    dmodel_importances = dimensions_importances["dmodel_importances"]
    dff_importances = dimensions_importances["dff_importances"]

    dmodel_indices = torch.topk(
        dmodel_importances, dim=0, largest=True, k=target_dmodel
    ).indices
    dff_indices = [
        torch.topk(dff_importances[i], dim=0, largest=True, k=target_dff).indices
        for i in range(len(dff_importances))
    ]
    return prune_with_indices(model, dmodel_indices, dff_indices)


def prune_from_selection(model: nn.Module, selection_path):
    """Prune to the channels recorded in a ChannelSelection JSON file
    (see src/projected_compression/channel_selection.py)."""
    selection = load_selection(selection_path)
    expected = {
        "dmodel": selection.dmodel,
        "dff": selection.dff,
        "n_blocks": selection.n_blocks,
    }
    actual = model_dims(model)
    if actual != expected:
        raise ValueError(
            f"channel selection {selection_path} is for {expected} "
            f"({selection.source_model}), but the model has {actual}"
        )
    logger.info(
        f"Pruning with channel selection {selection_path}:\n{describe(selection)}"
    )

    dmodel_indices = (
        list(range(selection.dmodel))
        if selection.dmodel_keep is None
        else selection.dmodel_keep
    )
    dff_indices = (
        [list(range(selection.dff))] * selection.n_blocks
        if selection.dff_keep is None
        else selection.dff_keep
    )
    return prune_with_indices(model, dmodel_indices, dff_indices)
