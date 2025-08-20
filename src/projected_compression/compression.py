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
    for layer_name in rightside_projections:
        weight = block_state_dict[layer_name]
        dff_magnitude += torch.norm(weight, dim=1)

    return dff_magnitude


def calculate_dimension_importances(model: nn.Module, topk_dmodel, topk_dff):
    dmodel_magnitudes = []
    dff_magnitudes = []

    # Embedding
    embedding_weight = model.embedding.embedding.embedding.weight.data
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
            ("ff_layer.layer.ff_post_act", dff_top_indices[i], dmodel_top_indices),
        ]

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


from collections import defaultdict
import os


class ModelTracer:
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        first_layer: str = "embedding",
        input_dir: str = None,
    ):
        self.model = model
        self.model_name = model_name
        self.first_layer = first_layer
        self.input_dir = input_dir
        self.registry = defaultdict(dict)
        self.step_counter = 0
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Module) and len(list(module.children())) == 0:
                module.register_forward_hook(self._forward_hook(name))
                module.register_full_backward_hook(self._backward_hook(name))
            # if name == self.first_layer and self.input_dir:
            #     module.register_forward_pre_hook(self._input_override_hook())

    def get_batch_for_step(self, step):
        input_path = os.path.join(self.input_dir, f"step_{step:03d}_input.pt")
        if not os.path.exists(input_path):
            raise FileNotFoundError(
                f"[{self.model_name}] Input file not found: {input_path}"
            )
        override_input = torch.load(input_path)

        output_path = os.path.join(self.input_dir, f"step_{step:03d}_target.pt")
        if not os.path.exists(output_path):
            raise FileNotFoundError(
                f"[{self.model_name}] Input file not found: {output_path}"
            )
        override_output = torch.load(output_path)

        return (override_input, override_output)

    def _input_override_hook(self):
        def hook(module, input):
            input_path = os.path.join(
                self.input_dir, f"step_{self.step_counter:03d}_input.pt"
            )
            if not os.path.exists(input_path):
                raise FileNotFoundError(
                    f"[{self.model_name}] Input file not found: {input_path}"
                )

            override_input = torch.load(input_path)
            print(f"[{self.model_name}] Using input from: {input_path}")
            self.step_counter += 1
            return (override_input,)

        return hook

    def _forward_hook(self, layer_name):
        def hook(module, input, output):
            self.registry[layer_name]["activation"] = output.detach().cpu()

        return hook

    def _backward_hook(self, layer_name):
        def hook(module, grad_input, grad_output):
            self.registry[layer_name]["grad"] = grad_output[0].detach().cpu()

        return hook

    def export(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(dict(self.registry), save_path)

    def export_weights(self, save_path: str):
        weights = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        torch.save(weights, save_path)
