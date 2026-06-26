from transformers import AutoConfig, LlamaForCausalLM, Qwen3ForCausalLM, AutoTokenizer
from src.core.metric_loggers import get_metric_logger
from src.projected_compression.initialization import create_model
import torch.distributed.checkpoint as dcp
import os
import torch
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
)
from hydra.utils import instantiate


def load_pc_state_dict_to_llama(state_dict, original_llama):
    conf = AutoConfig.from_pretrained(original_llama)

    # conf.num_hidden_layers = # stays the same
    # conf.num_key_value_heads = # stays the same
    # conf.num_attention_heads = # stays the same

    conf.hidden_size = state_dict["encoder.blocks.0.attention_layer.norm.weight"].shape[
        0
    ]
    conf.intermediate_size = state_dict[
        "encoder.blocks.0.ff_layer.layer.gate.weight"
    ].shape[0]
    conf.torch_dtype = (
        None  # prevent saved config.json from causing dtype mismatch on load
    )
    # LLaMA 3.2-1B has tie_word_embeddings=True, but the compressed model always has
    # independent embedding and lm_head (different projections). With tied weights,
    # load_state_dict silently overwrites embed_tokens with lm_head values (lm_head is
    # processed last in named_parameters). This corrupts the embedding matrix entirely.
    conf.tie_word_embeddings = False

    llama = LlamaForCausalLM(conf)

    new_state_dict = {
        "model.embed_tokens.weight": state_dict["embedding"],
        "model.norm.weight": state_dict["head.norm.weight"],
        "lm_head.weight": state_dict["head.linear.weight"],
    }

    # embed_tokens
    for layer in range(conf.num_hidden_layers):
        prefix_src = f"encoder.blocks.{layer}."
        prefix_tgt = f"model.layers.{layer}."

        # attention
        new_state_dict[f"{prefix_tgt}input_layernorm.weight"] = state_dict[
            f"{prefix_src}attention_layer.norm.weight"
        ]
        new_state_dict[f"{prefix_tgt}self_attn.q_proj.weight"] = state_dict[
            f"{prefix_src}attention_layer.layer.q_proj.weight"
        ]
        new_state_dict[f"{prefix_tgt}self_attn.k_proj.weight"] = state_dict[
            f"{prefix_src}attention_layer.layer.k_proj.weight"
        ]
        new_state_dict[f"{prefix_tgt}self_attn.v_proj.weight"] = state_dict[
            f"{prefix_src}attention_layer.layer.v_proj.weight"
        ]
        new_state_dict[f"{prefix_tgt}self_attn.o_proj.weight"] = state_dict[
            f"{prefix_src}attention_layer.layer.o_proj.weight"
        ]
        # mlp
        new_state_dict[f"{prefix_tgt}mlp.gate_proj.weight"] = state_dict[
            f"{prefix_src}ff_layer.layer.gate.weight"
        ]
        new_state_dict[f"{prefix_tgt}mlp.up_proj.weight"] = state_dict[
            f"{prefix_src}ff_layer.layer.ff_pre_act.weight"
        ]
        new_state_dict[f"{prefix_tgt}mlp.down_proj.weight"] = state_dict[
            f"{prefix_src}ff_layer.layer.ff_post_act.weight"
        ]
        new_state_dict[f"{prefix_tgt}post_attention_layernorm.weight"] = state_dict[
            f"{prefix_src}ff_layer.norm.weight"
        ]

    llama.load_state_dict(new_state_dict, strict=True)
    return llama


def load_pc_state_dict_to_qwen(state_dict, original_qwen):
    conf = AutoConfig.from_pretrained(original_qwen)

    # num_hidden_layers / num_attention_heads / num_key_value_heads / head_dim unchanged;
    # only the residual stream (hidden_size) and ff (intermediate_size) are compressed.
    conf.hidden_size = state_dict["encoder.blocks.0.attention_layer.norm.weight"].shape[
        0
    ]
    conf.intermediate_size = state_dict[
        "encoder.blocks.0.ff_layer.layer.gate.weight"
    ].shape[0]
    conf.torch_dtype = None
    # compressed model has independent embedding and lm_head (different projections)
    conf.tie_word_embeddings = False

    qwen = Qwen3ForCausalLM(conf)

    new_state_dict = {
        "model.embed_tokens.weight": state_dict["embedding"],
        "model.norm.weight": state_dict["head.norm.weight"],
        "lm_head.weight": state_dict["head.linear.weight"],
    }

    for layer in range(conf.num_hidden_layers):
        prefix_src = f"encoder.blocks.{layer}."
        prefix_tgt = f"model.layers.{layer}."

        # attention
        new_state_dict[f"{prefix_tgt}input_layernorm.weight"] = state_dict[
            f"{prefix_src}attention_layer.norm.weight"
        ]
        new_state_dict[f"{prefix_tgt}self_attn.q_proj.weight"] = state_dict[
            f"{prefix_src}attention_layer.layer.q_proj.weight"
        ]
        new_state_dict[f"{prefix_tgt}self_attn.k_proj.weight"] = state_dict[
            f"{prefix_src}attention_layer.layer.k_proj.weight"
        ]
        new_state_dict[f"{prefix_tgt}self_attn.v_proj.weight"] = state_dict[
            f"{prefix_src}attention_layer.layer.v_proj.weight"
        ]
        new_state_dict[f"{prefix_tgt}self_attn.o_proj.weight"] = state_dict[
            f"{prefix_src}attention_layer.layer.o_proj.weight"
        ]
        # Qwen3 QK-norm (dhead-sized, not compressed)
        new_state_dict[f"{prefix_tgt}self_attn.q_norm.weight"] = state_dict[
            f"{prefix_src}attention_layer.layer.q_norm.weight"
        ]
        new_state_dict[f"{prefix_tgt}self_attn.k_norm.weight"] = state_dict[
            f"{prefix_src}attention_layer.layer.k_norm.weight"
        ]
        # mlp
        new_state_dict[f"{prefix_tgt}mlp.gate_proj.weight"] = state_dict[
            f"{prefix_src}ff_layer.layer.gate.weight"
        ]
        new_state_dict[f"{prefix_tgt}mlp.up_proj.weight"] = state_dict[
            f"{prefix_src}ff_layer.layer.ff_pre_act.weight"
        ]
        new_state_dict[f"{prefix_tgt}mlp.down_proj.weight"] = state_dict[
            f"{prefix_src}ff_layer.layer.ff_post_act.weight"
        ]
        new_state_dict[f"{prefix_tgt}post_attention_layernorm.weight"] = state_dict[
            f"{prefix_src}ff_layer.norm.weight"
        ]

    qwen.load_state_dict(new_state_dict, strict=True)
    return qwen


def _load_pc_state_dict_to_hf(state_dict, original_path):
    """Dispatch to the Qwen or Llama exporter based on the presence of QK-norm."""
    is_qwen = any("attention_layer.layer.q_norm.weight" in k for k in state_dict)
    if is_qwen:
        return load_pc_state_dict_to_qwen(state_dict, original_path)
    return load_pc_state_dict_to_llama(state_dict, original_path)


def convert_pc_to_hf(cfg, metric_logger):
    if metric_logger is None:
        metric_logger = get_metric_logger(
            metric_logger_config=instantiate(
                cfg.infrastructure.metric_logger, _convert_="all"
            ),
            neptune_run_id=None,
        )

    model = create_model(
        cfg.model, cfg.projected_compression, source_model_for_distillation=False
    )

    dcp.load(model.state_dict(), checkpoint_id=cfg.trainer.checkpoint.load.path)

    model.prepare_compressed_weights()
    model.target_model.embedding = torch.nn.Parameter(
        model.source_model.embedding.weight @ model.projections.embedding.T
        + model.projections.auxiliary_embedding_weights.weight
    )

    target_model_state_dict = get_model_state_dict(
        model=model.target_model,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        ),
    )

    llama_model = None
    if os.environ.get("RANK", "0") == "0":
        llama_model = _load_pc_state_dict_to_hf(
            target_model_state_dict,
            cfg.projected_compression.original_llama_path,
        )
        llama_model.save_pretrained(cfg.trainer.checkpoint.save.path)
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.projected_compression.original_llama_path
        )
        tokenizer.save_pretrained(cfg.trainer.checkpoint.save.path)

    return None, None, None, None, None
