from collections import OrderedDict
import re
import torch
from transformers import AutoConfig, AutoModelForCausalLM
import re
from torch.distributed.tensor import DTensor


def print_state_dict_info(state_dict): #dev
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            print(f"{name:60s} {type(param)}, shape={tuple(param.shape)} "
            # print(f"{name:60s} {param}, shape={tuple(param.shape)} "
                f"norm={param.norm().item():.4f}")
        else:
            # Sometimes buffers / metadata can be non-tensors
            print(f"{name:60s} NON-TENSOR {type(param)}")

def cast_state_dict_to_tensors(state_dict, device="cpu"):
    """
    Convert all DTensors in a state dict to regular torch.Tensors.
    By default, gathers them to CPU.
    """
    full_state = {}
    for k, v in state_dict.items():
        if isinstance(v, DTensor):
            full_state[k] = v.full_tensor().float().to(device)
            # full_state[k] = v.to_local().to(device)
        elif isinstance(v, torch.Tensor):
            full_state[k] = v.to(device)
        else:
            full_state[k] = v
    return full_state

def remap_nano_to_llama31_hf(nano_dict):
    """
    Convert Nano state_dict -> Hugging Face LLaMA-3.1 style keys
    """

    replacement_mappings = [
        # Embedding
        (r"^embedding\.embedding\.weight$", "model.embed_tokens.weight"),

        # Attention LayerNorm
        (r"^encoder\.blocks\.(\d+)\.attention_layer\.norm\.weight$",
         r"model.layers.\1.input_layernorm.weight"),

        # Attention projections
        (r"^encoder\.blocks\.(\d+)\.attention_layer\.layer\.q_proj\.weight$",
         r"model.layers.\1.self_attn.q_proj.weight"),
        (r"^encoder\.blocks\.(\d+)\.attention_layer\.layer\.k_proj\.weight$",
         r"model.layers.\1.self_attn.k_proj.weight"),
        (r"^encoder\.blocks\.(\d+)\.attention_layer\.layer\.v_proj\.weight$",
         r"model.layers.\1.self_attn.v_proj.weight"),
        (r"^encoder\.blocks\.(\d+)\.attention_layer\.layer\.o_proj\.weight$",
         r"model.layers.\1.self_attn.o_proj.weight"),

        # Feed-forward norm
        (r"^encoder\.blocks\.(\d+)\.ff_layer\.norm\.weight$",
         r"model.layers.\1.post_attention_layernorm.weight"),

        # # Feed-forward projections
        # (r"^encoder\.blocks\.(\d+)\.ff_layer\.layer\.ff_pre_act\.weight$",
        #  r"model.layers.\1.mlp.gate_proj.weight"),
        # (r"^encoder\.blocks\.(\d+)\.ff_layer\.layer\.ff_post_act\.weight$",
        #  r"model.layers.\1.mlp.down_proj.weight"),
        # (r"^encoder\.blocks\.(\d+)\.ff_layer\.layer\.gate\.weight$",
        #  r"model.layers.\1.mlp.up_proj.weight"),

         # Feed-forward projections
        (r"^encoder\.blocks\.(\d+)\.ff_layer\.layer\.ff_pre_act\.weight$",
        r"model.layers.\1.mlp.up_proj.weight"),
        (r"^encoder\.blocks\.(\d+)\.ff_layer\.layer\.ff_post_act\.weight$",
        r"model.layers.\1.mlp.down_proj.weight"),
        (r"^encoder\.blocks\.(\d+)\.ff_layer\.layer\.gate\.weight$",
        r"model.layers.\1.mlp.gate_proj.weight"),

        # Final norm + lm head
        (r"^head\.norm\.weight$", "model.norm.weight"),
        (r"^head\.linear\.weight$", "lm_head.weight"),
    ]

    hf_dict = {}
    for k, v in nano_dict.items():
        new_k = None
        for patt, repl in replacement_mappings:
            if re.match(patt, k):
                new_k = re.sub(patt, repl, k)
                break
        if new_k is not None:
            hf_dict[new_k] = v

    return hf_dict

# dev Old Nano checkpoint naming
# def remap_nano_to_llama31_hf(nano_dict):
#     """
#     Convert nano state_dict -> Hugging Face LLaMA-3.1 style keys
#     """
#     replacement_mappings = [
#         # Embedding
#         (r"^embedding_layer\.weight$", "model.embed_tokens.weight"),

#         # Attention LayerNorm
#         (r"^encoder\.blocks\.block_(\d+)\.block\.residual_attention\.layer\.pre_norm\.weight$",
#          r"model.layers.\1.input_layernorm.weight"),

#         # Attention projections
#         (r"^encoder\.blocks\.block_(\d+)\.block\.residual_attention\.layer\.attention\.q_proj\.weight$",
#          r"model.layers.\1.self_attn.q_proj.weight"),
#         (r"^encoder\.blocks\.block_(\d+)\.block\.residual_attention\.layer\.attention\.k_proj\.weight$",
#          r"model.layers.\1.self_attn.k_proj.weight"),
#         (r"^encoder\.blocks\.block_(\d+)\.block\.residual_attention\.layer\.attention\.v_proj\.weight$",
#          r"model.layers.\1.self_attn.v_proj.weight"),
#         (r"^encoder\.blocks\.block_(\d+)\.block\.residual_attention\.layer\.attention\.o_proj\.weight$",
#          r"model.layers.\1.self_attn.o_proj.weight"),

#         # RoPE params (HF does not store them as tensors in state_dict → skip)
#         (r"^encoder\.blocks\.block_(\d+)\.block\.residual_attention\.layer\.attention\.rope\..*$", None),

#         # Post-attention LayerNorm
#         (r"^encoder\.blocks\.block_(\d+)\.block\.residual_feedforward\.layer\.pre_norm\.weight$",
#          r"model.layers.\1.post_attention_layernorm.weight"),

#         # Feed-forward projections
#         (r"^encoder\.blocks\.block_(\d+)\.block\.residual_feedforward\.layer\.feedforward\.ff_pre_act\.weight$",
#          r"model.layers.\1.mlp.gate_proj.weight"),
#         (r"^encoder\.blocks\.block_(\d+)\.block\.residual_feedforward\.layer\.feedforward\.ff_post_act\.weight$",
#          r"model.layers.\1.mlp.down_proj.weight"),
#         (r"^encoder\.blocks\.block_(\d+)\.block\.residual_feedforward\.layer\.feedforward\.gate\.weight$",
#          r"model.layers.\1.mlp.up_proj.weight"),

#         # Final norm + lm head
#         (r"^head\.unembedding\.head_norm\.weight$", "model.norm.weight"),
#         (r"^head\.unembedding\.head\.weight$", "lm_head.weight"),
#     ]

#     hf_dict = {}
#     for k, v in nano_dict.items():
#         new_k = k
#         for patt, repl in replacement_mappings:
#             if re.match(patt, k):
#                 if repl is None:
#                     new_k = None  # skip rope buffers
#                 else:
#                     new_k = re.sub(patt, repl, k)
#                 break
#         if new_k is not None:
#             hf_dict[new_k] = v

#     return hf_dict



def save_to_llama_3_hf(nano_model_state_dict, save_dir:str, dmodel:int, dff:int, n_att_heads:int, n_kvatt_heads:int, head_dim:int, nlayers:int):
    config = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B")

    config.hidden_size = dmodel
    config.intermediate_size = dff
    config.num_attention_heads = n_att_heads
    config.num_key_value_heads = n_kvatt_heads
    config.head_dim = head_dim
    config.num_hidden_layers = nlayers

    hf_model = AutoModelForCausalLM.from_config(config)
    print(f"HF -----------------")
    print_state_dict_info(hf_model.state_dict())
    # print(f"nano_model_state_dict -----------------")
    # print_state_dict_info(nano_model_state_dict)
    hf_state_dict = remap_nano_to_llama31_hf(nano_model_state_dict)
    print(f"hf_state_dict from nano -----------------")
    print_state_dict_info(hf_state_dict)
    hf_model.load_state_dict(hf_state_dict, strict=True)

    print(f"Saving HF model with the following config {config}") #dev

    hf_model.save_pretrained(save_dir) 