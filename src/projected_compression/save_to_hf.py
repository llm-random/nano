from transformers import AutoConfig, LlamaForCausalLM, AutoTokenizer
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


def save_pc_to_hf(cfg, metric_logger):
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
        llama_model = load_pc_state_dict_to_llama(
            target_model_state_dict,
            original_llama=cfg.projected_compression.original_llama_path,
        )
        llama_model.save_pretrained(cfg.trainer.checkpoint.save.path)
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.projected_compression.original_llama_path
        )
        tokenizer.save_pretrained(cfg.trainer.checkpoint.save.path)

    training_start_config = {"next_step": 0, "run_id": None, "processed_tokens": 0}
    return llama_model, None, None, training_start_config, metric_logger
