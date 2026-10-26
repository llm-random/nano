# Maps SliceGPT's model/layer adapter interface (TransformerCompression/src/slicegpt/model_adapter.py)
# onto the nano Llama model (src/projected_compression/model.py).
from torch import nn


def get_blocks(model: nn.Module) -> list[nn.Module]:
    return list(model.encoder.blocks)


def get_embedding(model: nn.Module) -> nn.Embedding:
    return model.embedding


def get_lm_head(model: nn.Module) -> nn.Linear:
    return model.head.linear


def get_pre_head_norm(model: nn.Module) -> nn.Module:
    return model.head.norm


def get_attention_inputs(block: nn.Module) -> list[nn.Linear]:
    attention = block.attention_layer.layer
    return [attention.q_proj, attention.k_proj, attention.v_proj]


def get_attention_output(block: nn.Module) -> nn.Linear:
    return block.attention_layer.layer.o_proj


def get_mlp_inputs(block: nn.Module) -> list[nn.Linear]:
    ff = block.ff_layer.layer
    return [ff.ff_pre_act, ff.gate]


def get_mlp_output(block: nn.Module) -> nn.Linear:
    return block.ff_layer.layer.ff_post_act


def get_first_norm(block: nn.Module) -> nn.Module:
    return block.attention_layer.norm


def get_second_norm(block: nn.Module) -> nn.Module:
    return block.ff_layer.norm
