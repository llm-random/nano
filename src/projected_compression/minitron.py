import torch
import torch.nn as nn
import json

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
            i = 0
            for layer in model.encoder.blocks:
                y = layer.attention_layer.norm(x) # normalized_pre_attn
                dmodel_importance += torch.sum(torch.abs(y), dim=[0, 1])  # Sum across batch and sequence dimensions

                y = layer.attention_layer.layer(y) # attention_output
                x = x + y  # Residual connection


                y = layer.ff_layer.norm(x) # normalized_pre_ff
                dmodel_importance += torch.sum(torch.abs(y), dim=[0, 1])  # Sum across batch and sequence dimensions

                ff_layer = layer.ff_layer.layer
                ff_gated = ff_layer.silu(ff_layer.gate(y))

                y = ff_layer.ff_pre_act(y) # ff_pre_act
                dff_importance[i] += torch.sum(torch.abs(y), dim=[0, 1])

                y = ff_layer.ff_post_act(y * ff_gated) # ff_output

                x = x + y  # Residual connection

                i += 1
            x = model.head(x)

            assert torch.allclose(x, model(batch)), "Model output does not match expected output"

    return dmodel_importance, dff_importance

def prune(model: nn.Module, dmodel_indices, dff_indices):
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
    model.head.norm.normalized_shape = tuple([3072])

    i = 0
    for block in model.encoder.blocks:
        block.attention_layer.norm.weight.data = block.attention_layer.norm.weight[dmodel_indices]
        block.attention_layer.norm.normalized_shape = tuple([3072])
        block.attention_layer.layer.q_proj.weight.data = block.attention_layer.layer.q_proj.weight[:, dmodel_indices]
        block.attention_layer.layer.k_proj.weight.data = block.attention_layer.layer.k_proj.weight[:, dmodel_indices]
        block.attention_layer.layer.v_proj.weight.data = block.attention_layer.layer.v_proj.weight[:, dmodel_indices]
        block.attention_layer.layer.o_proj.weight.data = block.attention_layer.layer.o_proj.weight[dmodel_indices, :]

        block.ff_layer.norm.weight.data = block.ff_layer.norm.weight[dmodel_indices]
        block.ff_layer.norm.normalized_shape = tuple([3072])
        block.ff_layer.layer.ff_pre_act.weight.data = block.ff_layer.layer.ff_pre_act.weight[:, dmodel_indices][dff_indices[i], :]
        block.ff_layer.layer.gate.weight.data = block.ff_layer.layer.gate.weight[:, dmodel_indices][dff_indices[i], :]
        block.ff_layer.layer.ff_post_act.weight.data = block.ff_layer.layer.ff_post_act.weight[dmodel_indices, :][:, dff_indices[i]]

        i += 1

    return model

def minitron_prune(model: nn.Module, dataloader, dmodel, dff, calibration_dataset_size, seq_len, total_batch_size, n_blocks):

    calibration_data = torch.zeros(calibration_dataset_size // total_batch_size, total_batch_size, seq_len, dtype=torch.long, device=device)
    for i, batch in enumerate(dataloader):
        if i * total_batch_size >= calibration_dataset_size:
            break
        calibration_data[i] = batch[:, :seq_len]

    dmodel_importance, dff_importance = calculate_dimension_importances(
        model, calibration_data, dmodel, dff, n_blocks
    )

    print("Importance dimensions calculated.")

    # select top k neurons
    topk_dmodel = 3072
    topk_dff = 9216

    dmodel_top_indices = torch.topk(dmodel_importance, dim=0, largest=True, k=topk_dmodel).indices.tolist()

    dff_top_indices = []
    for i in range(n_blocks):
        dff_top_indices_current = torch.topk(dff_importance[i], dim=0, largest=True, k=topk_dff).indices.tolist()
        dff_top_indices.append(dff_top_indices_current)

    # save to file indices as dict
    dict_to_save = {"dmodel_top_indices": dmodel_top_indices, "dff_top_indices": dff_top_indices}
    path = "./top_indices.json"

    with open(path, "w") as f:
        json.dump(dict_to_save, f)

    model = prune(model, dmodel_top_indices, dff_top_indices)

    print("Model pruned.")

    return model

