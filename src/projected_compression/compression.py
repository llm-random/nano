import torch
import torch.nn as nn


def get_nested_attr(module, attr_path):
    attrs = attr_path.split(".")
    for attr in attrs:
        module = getattr(module, attr)
    return module


def determine_dmodel_magnitudes(block_state_dict):
    magnitudes = []

    leftside_projections = [
        "attention_layer.layer.q_proj.weight",
        "attention_layer.layer.k_proj.weight",
        "attention_layer.layer.v_proj.weight",
        "ff_layer.layer.ff_pre_act.weight",
    ]

    # For models with SiLU (e.g. LLama) 
    if "ff_layer.layer.gate.weight" in block_state_dict.keys(): 
        leftside_projections.append("ff_layer.layer.gate.weight")

    rightside_projections = [
        "attention_layer.layer.o_proj.weight",
        "ff_layer.layer.ff_post_act.weight",
    ]

    for layer_name in leftside_projections:
        weight = block_state_dict[layer_name]
        magnitudes.append(torch.norm(weight, dim=0))

    for layer_name in rightside_projections:
        weight = block_state_dict[layer_name]
        magnitudes.append(torch.norm(weight, dim=1))

    return magnitudes


def determine_dff_magnitudes(block_state_dict):
    weight = block_state_dict["ff_layer.layer.ff_post_act.weight"]
    dff_magnitude = torch.norm(weight, dim=0)

    rightside_projections = ["ff_layer.layer.ff_pre_act.weight"]
    # For models with SiLU (e.g. LLama) 
    if "ff_layer.layer.gate.weight" in block_state_dict.keys():
        rightside_projections.append("ff_layer.layer.gate.weight")

    for layer_name in rightside_projections:
        weight = block_state_dict[layer_name]
        dff_magnitude += torch.norm(weight, dim=1)

    return dff_magnitude


def calculate_dimension_importances(model: nn.Module, topk_dmodel, topk_dff):
    dmodel_magnitudes = []
    dff_magnitudes = []

    # Embedding
    embedding_weight = model.embedding.embedding.weight.data
    dmodel_magnitudes.append(torch.norm(embedding_weight, dim=0))

    # Head
    head_weight = model.head.linear.weight.data
    dmodel_magnitudes.append(torch.norm(head_weight, dim=0))

    for block in model.encoder.blocks:
        block_state_dict = block.state_dict()
        dmodel_magnitudes.extend(determine_dmodel_magnitudes(block_state_dict))
        dff_magnitudes.append(determine_dff_magnitudes(block_state_dict))

    mean_dmodel_magnitudes = torch.stack(dmodel_magnitudes, dim=1).mean(dim=1)
    dmodel_top_indices = torch.topk(mean_dmodel_magnitudes, topk_dmodel).indices
    dmodel_top_indices = dmodel_top_indices.sort().values

    dff_magnitudes = torch.stack(dff_magnitudes)
    dff_top_indices = torch.topk(dff_magnitudes, dim=1, k=topk_dff).indices

    # sort indices
    dmodel_top_indices = dmodel_top_indices.sort().values
    for indices in dff_top_indices:
        indices.copy_(indices.sort().values)

    return dmodel_top_indices, dff_top_indices


def initialize_projection_weights(
    model: nn.Module, dmodel_top_indices, dff_top_indices
):
    model.head.linear.init_projections(dmodel_top_indices, None)
    model.embedding.init_projection(dmodel_top_indices)

    cloned_data = model.head.norm.weight.data.clone()
    model.head.norm.weight = torch.nn.Parameter(cloned_data[dmodel_top_indices])
    model.head.norm.normalized_shape = tuple(model.head.norm.weight.shape)

    for i, block in enumerate(model.encoder.blocks):
        layers_to_init_projections = [
            ("attention_layer.layer.q_proj", dmodel_top_indices, None),
            ("attention_layer.layer.k_proj", dmodel_top_indices, None),
            ("attention_layer.layer.v_proj", dmodel_top_indices, None),
            ("ff_layer.layer.ff_pre_act", dmodel_top_indices, dff_top_indices[i]),
            ("attention_layer.layer.o_proj", None, dmodel_top_indices),
            ("ff_layer.layer.ff_post_act", dff_top_indices[i], dmodel_top_indices)
        ]
        # For models with SiLU (e.g. LLama) 
        if "ff_layer.layer.gate.weight" in block.state_dict().keys():
            layers_to_init_projections.append(("ff_layer.layer.gate", dmodel_top_indices, dff_top_indices[i]))
        
        for layer_name, in_topk_indices, out_topk_indices in layers_to_init_projections:
            get_nested_attr(block, layer_name).init_projections(
                in_topk_indices, out_topk_indices
            )

        cloned_data = block.attention_layer.norm.weight.data.clone()
        block.attention_layer.norm.weight = torch.nn.Parameter(
            cloned_data[dmodel_top_indices]
        )
        block.attention_layer.norm.normalized_shape = tuple(
            block.attention_layer.norm.weight.shape
        )

        cloned_data = block.ff_layer.norm.weight.data.clone()
        block.ff_layer.norm.weight = torch.nn.Parameter(cloned_data[dmodel_top_indices])
        block.ff_layer.norm.normalized_shape = tuple(block.ff_layer.norm.weight.shape)


def init_compression(model: nn.Module, dmodel, dff):
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    dmodel_top_indices, dff_top_indices = calculate_dimension_importances(
        model, dmodel, dff
    )
    initialize_projection_weights(model, dmodel_top_indices, dff_top_indices)


