from collections import OrderedDict
import re

import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from transformers import AutoModelForCausalLM


def remap_olmo2hf_state_dict_to_nano(olmo_state_dict):
    """Remap an OLMo2 HF state dict onto the nano naming.

    OLMo2 uses reordered (post) normalization, so its two per-layer norms map to the
    residual norms differently than Llama:
      post_attention_layernorm   -> attention residual norm (applied after attention)
      post_feedforward_layernorm -> ff residual norm (applied after the MLP)
    It also carries full-dim QK-norm (q_norm / k_norm) and no input_layernorm.
    """
    remapped = {}
    for key, value in olmo_state_dict.items():
        new_key = key

        new_key = new_key.replace("model.embed_tokens.weight", "embedding.weight")
        new_key = new_key.replace("model.norm.weight", "head.norm.weight")
        new_key = new_key.replace("lm_head.weight", "head.linear.weight")

        layer_match = re.match(r"model\.layers\.(\d+)\.(.*)", new_key)
        if layer_match:
            layer_num = layer_match.group(1)
            sub_key = layer_match.group(2)

            # Attention projections
            sub_key = sub_key.replace(
                "self_attn.q_proj.weight", "attention_layer.layer.q_proj.weight"
            )
            sub_key = sub_key.replace(
                "self_attn.k_proj.weight", "attention_layer.layer.k_proj.weight"
            )
            sub_key = sub_key.replace(
                "self_attn.v_proj.weight", "attention_layer.layer.v_proj.weight"
            )
            sub_key = sub_key.replace(
                "self_attn.o_proj.weight", "attention_layer.layer.o_proj.weight"
            )

            # Full-dim QK-norm
            sub_key = sub_key.replace(
                "self_attn.q_norm.weight", "attention_layer.layer.q_norm.weight"
            )
            sub_key = sub_key.replace(
                "self_attn.k_norm.weight", "attention_layer.layer.k_norm.weight"
            )

            # Reordered norms (post-attention / post-feedforward)
            sub_key = sub_key.replace(
                "post_attention_layernorm.weight", "attention_layer.norm.weight"
            )
            sub_key = sub_key.replace(
                "post_feedforward_layernorm.weight", "ff_layer.norm.weight"
            )

            # MLP
            sub_key = sub_key.replace(
                "mlp.up_proj.weight", "ff_layer.layer.ff_pre_act.weight"
            )
            sub_key = sub_key.replace(
                "mlp.gate_proj.weight", "ff_layer.layer.gate.weight"
            )
            sub_key = sub_key.replace(
                "mlp.down_proj.weight", "ff_layer.layer.ff_post_act.weight"
            )

            new_key = f"encoder.blocks.{layer_num}.{sub_key}"

        remapped[new_key] = value

    return OrderedDict(remapped)


def copy_olmo_model_weights_from_HF(model, path):
    hf_model = AutoModelForCausalLM.from_pretrained(path)
    remapped_state_dict = remap_olmo2hf_state_dict_to_nano(hf_model.state_dict())
    model.load_state_dict(remapped_state_dict)


def save_pretrained_olmo_as_nano(cfg: OmegaConf, metric_logger=None):

    hf_model = AutoModelForCausalLM.from_pretrained(cfg.trainer.checkpoint.load.path)
    nano_sd = remap_olmo2hf_state_dict_to_nano(hf_model.state_dict())
    # PC mem-eff projections run in fp32 (cast_bfloat16=false); cast here so the source
    # checkpoint matches the Llama pipeline's fp32 checkpoint regardless of HF dtype
    nano_sd = OrderedDict((k, v.float()) for k, v in nano_sd.items())

    model = instantiate(cfg.model, _convert_="all")
    weights = {k for k in model.state_dict() if not k.endswith((".sin", ".cos"))}
    missing = weights - set(nano_sd)
    if missing:
        raise RuntimeError(f"OLMo2->nano remap left weights unfilled: {sorted(missing)}")

    model.load_state_dict(nano_sd, strict=False, assign=True)

    torch.save(model.state_dict(), cfg.trainer.checkpoint.save.path)

    if cfg.get("apply_functions", None):
        for fn in instantiate(cfg.apply_functions):
            fn(model)

    return None, None, None, None, None


def verify_olmo_against_hf(model, hf_path, vocab_size, seq_len=256, min_agreement=0.99):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_model = AutoModelForCausalLM.from_pretrained(hf_path).to(device).eval().float()
    model = model.to(device).eval().float()

    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)

    with torch.no_grad():
        nano_logits = model(input_ids)
        hf_logits = hf_model(input_ids).logits

    max_abs_diff = (nano_logits - hf_logits).abs().max().item()
    argmax_agreement = (
        (nano_logits.argmax(-1) == hf_logits.argmax(-1)).float().mean().item()
    )
    print(
        f"[verify] max_abs_logit_diff={max_abs_diff:.4g} "
        f"argmax_agreement={argmax_agreement:.4f} over {seq_len} positions"
    )
    if argmax_agreement < min_agreement:
        raise RuntimeError(
            f"OLMo2->nano verification failed: argmax_agreement={argmax_agreement:.4f} "
            f"< {min_agreement} (max_abs_logit_diff={max_abs_diff:.4g})"
        )
