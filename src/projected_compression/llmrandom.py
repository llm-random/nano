from collections import OrderedDict
import re
import torch


def remap_statedict_llmrandom_to_nano(llmrandom_dict):
    remapped = {}
    for key, value in llmrandom_dict.items():
        new_key = key

        # Embedding
        new_key = new_key.replace(
            "embedding_layer.layers.0.weight", "embedding.embedding.embedding.weight"
        )

        # Final norm and lm head
        new_key = new_key.replace(
            "head.unembedding.head_norm.weight", "head.norm.weight"
        )
        new_key = new_key.replace("head.unembedding.head.weight", "head.linear.weight")

        # Layers
        layer_match = re.match(r"encoder\.blocks\.block_(\d+)\.block\.(.*)", new_key)
        if layer_match:
            layer_num = layer_match.group(1)
            sub_key = layer_match.group(2)

            if "residual_attention.layer.attention.rope" in sub_key:
                continue

            # Attention projections
            sub_key = sub_key.replace(
                "residual_attention.layer.attention.input_projection_q.weight",
                f"attention_layer.layer.q_proj.weight",
            )
            sub_key = sub_key.replace(
                "residual_attention.layer.attention.input_projection_k.weight",
                f"attention_layer.layer.k_proj.weight",
            )
            sub_key = sub_key.replace(
                "residual_attention.layer.attention.input_projection_v.weight",
                f"attention_layer.layer.v_proj.weight",
            )
            sub_key = sub_key.replace(
                "residual_attention.layer.attention.output_projection.weight",
                f"attention_layer.layer.o_proj.weight",
            )

            # Attention norms
            sub_key = sub_key.replace(
                "residual_attention.layer.pre_norm.weight",
                "attention_layer.norm.weight",
            )
            sub_key = sub_key.replace(
                "residual_feedforward.layer.pre_norm.weight", "ff_layer.norm.weight"
            )

            # MLP
            sub_key = sub_key.replace(
                "residual_feedforward.layer.feedforward.logging_ff_pre_relu.weight",
                "ff_layer.layer.ff_pre_act.weight",
            )
            sub_key = sub_key.replace(
                "residual_feedforward.layer.feedforward.logging_ff_post_relu.weight",
                "ff_layer.layer.ff_post_act.weight",
            )

            new_key = f"encoder.blocks.{layer_num}.{sub_key}"

        remapped[new_key] = value

    return OrderedDict(remapped)


def load_llmrandom_checkpoint(load_config, model):
    checkpoint = torch.load(load_config.path)

    remapped_state_dict = remap_statedict_llmrandom_to_nano(checkpoint["model"])

    dhead = model.encoder.blocks[0].attention_layer.layer.dhead
    heads = model.encoder.blocks[0].attention_layer.layer.q_heads
    dmodel = model.encoder.blocks[0].attention_layer.layer.dmodel
    for n_layer in range(len(model.encoder.blocks)):
        q = remapped_state_dict[
            f"encoder.blocks.{n_layer}.attention_layer.layer.q_proj.weight"
        ]
        k = remapped_state_dict[
            f"encoder.blocks.{n_layer}.attention_layer.layer.k_proj.weight"
        ]
        v = remapped_state_dict[
            f"encoder.blocks.{n_layer}.attention_layer.layer.v_proj.weight"
        ]

        con = torch.cat((q, k, v))
        con = con.view(heads, 3 * dhead, dmodel)
        q, k, v = con.chunk(3, dim=1)

        remapped_state_dict[
            f"encoder.blocks.{n_layer}.attention_layer.layer.q_proj.weight"
        ] = q.reshape(heads * dhead, -1)
        remapped_state_dict[
            f"encoder.blocks.{n_layer}.attention_layer.layer.k_proj.weight"
        ] = k.reshape(heads * dhead, -1)
        remapped_state_dict[
            f"encoder.blocks.{n_layer}.attention_layer.layer.v_proj.weight"
        ] = v.reshape(heads * dhead, -1)

    model.load_state_dict(remapped_state_dict)
