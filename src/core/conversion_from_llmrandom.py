from collections import OrderedDict
import re
import torch


def remap_llmrandom_state_dict_to_nano(llmrandom_dict):

    replacement_mappings = [
        # Embedding
        (r"embedding_layer\.layers\.0\.weight", "embedding.embedding.weight"),
        # Attention projections
        (
            r"encoder\.blocks\.block_(\d+)\.block\.residual_attention\.layer\.pre_norm\.weight",
            r"encoder.blocks.\1.attention_layer.norm.weight",
        ),
        (
            r"encoder\.blocks\.block_(\d+)\.block\.residual_attention\.layer\.attention\.input_projection_q\.weight",
            r"encoder.blocks.\1.attention_layer.layer.q_proj.weight",
        ),
        (
            r"encoder\.blocks\.block_(\d+)\.block\.residual_attention\.layer\.attention\.input_projection_k\.weight",
            r"encoder.blocks.\1.attention_layer.layer.k_proj.weight",
        ),
        (
            r"encoder\.blocks\.block_(\d+)\.block\.residual_attention\.layer\.attention\.input_projection_v\.weight",
            r"encoder.blocks.\1.attention_layer.layer.v_proj.weight",
        ),
        (
            r"encoder\.blocks\.block_(\d+)\.block\.residual_attention\.layer\.attention\.output_projection\.weight",
            r"encoder.blocks.\1.attention_layer.layer.o_proj.weight",
        ),
        # Feed-forward
        (
            r"encoder\.blocks\.block_(\d+)\.block\.residual_feedforward\.layer\.pre_norm\.weight",
            r"encoder.blocks.\1.ff_layer.norm.weight",
        ),
        (
            r"encoder\.blocks\.block_(\d+)\.block\.residual_feedforward\.layer\.feedforward\.logging_ff_pre_relu\.weight",
            r"encoder.blocks.\1.ff_layer.layer.ff_pre_act.weight",
        ),
        (
            r"encoder\.blocks\.block_(\d+)\.block\.residual_feedforward\.layer\.feedforward\.logging_ff_post_relu\.weight",
            r"encoder.blocks.\1.ff_layer.layer.ff_post_act.weight",
        ),
        # Head
        (r"head\.unembedding\.head_norm\.weight", "head.norm.weight"),
        (r"head\.unembedding\.head\.weight", "head.linear.weight"),
    ]

    remapped = {}
    for key, value in llmrandom_dict.items():
        new_key = key

        if "residual_attention.layer.attention.rope" in new_key:
            continue

        for pattern, replacement in replacement_mappings:
            new_key = re.sub(pattern, replacement, new_key)

        remapped[new_key] = value

    return OrderedDict(remapped)


def fix_qkv_from_llmrandom(model, remapped_state_dict):
    """
    This function fixes mixing-up of weights in llm-random.
    The code in an old repo was similar to this:
        ```
        projected = torch.concat((q,k,v), dim=-1)

        batch, seq_len = x.shape[:-1]
        projected = projected.view(
            batch, seq_len, self.heads, 3 * self.dhead
        ).transpose(1, 2)
        ```

        Istead of first splitting into 3 parts (q,k,v) and then into heads, we were splitting into heads and then into 3 groups ending up mixing weights.
    """
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


def load_llmrandom_checkpoint(load_config, model):
    checkpoint = torch.load(load_config.path)
    remapped_state_dict = remap_llmrandom_state_dict_to_nano(checkpoint["model"])
    fix_qkv_from_llmrandom(model, remapped_state_dict)
    model.load_state_dict(remapped_state_dict)
