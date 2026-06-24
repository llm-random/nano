from collections import OrderedDict
import re

import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from transformers import AutoModelForCausalLM

from .llama import remap_llamahf_state_dict_to_nano


def remap_qwen3hf_state_dict_to_nano(qwen_state_dict):
    remapped = remap_llamahf_state_dict_to_nano(qwen_state_dict)
    # Qwen3 adds per-head QK-norm that Llama lacks
    renamed = {}
    for key, value in remapped.items():
        key = re.sub(
            r"(encoder\.blocks\.\d+\.)self_attn\.q_norm\.weight",
            r"\1attention_layer.layer.q_norm.weight",
            key,
        )
        key = re.sub(
            r"(encoder\.blocks\.\d+\.)self_attn\.k_norm\.weight",
            r"\1attention_layer.layer.k_norm.weight",
            key,
        )
        renamed[key] = value
    return OrderedDict(renamed)


def copy_qwen_model_weights_from_HF(model, path):
    hf_model = AutoModelForCausalLM.from_pretrained(path)
    remapped_state_dict = remap_qwen3hf_state_dict_to_nano(hf_model.state_dict())
    model.load_state_dict(remapped_state_dict)


def save_pretrained_qwen_as_nano(cfg: OmegaConf, metric_logger=None):

    hf_model = AutoModelForCausalLM.from_pretrained(cfg.trainer.checkpoint.load.path)
    nano_sd = remap_qwen3hf_state_dict_to_nano(hf_model.state_dict())
    # PC mem-eff projections run in fp32 (cast_bfloat16=false); transformers>=5 loads
    # Qwen as bf16 by default, so cast here to match the Llama pipeline's fp32 checkpoint
    nano_sd = OrderedDict((k, v.float()) for k, v in nano_sd.items())

    model = instantiate(cfg.model, _convert_="all")
    weights = {k for k in model.state_dict() if not k.endswith((".sin", ".cos"))}
    missing = weights - set(nano_sd)
    if missing:
        raise RuntimeError(f"Qwen->nano remap left weights unfilled: {sorted(missing)}")

    model.load_state_dict(nano_sd, strict=False, assign=True)

    torch.save(model.state_dict(), cfg.trainer.checkpoint.save.path)

    if cfg.get("apply_functions", None):
        for fn in instantiate(cfg.apply_functions):
            fn(model)

    return None, None, None, None, None


def verify_qwen_against_hf(model, hf_path, vocab_size, seq_len=256, min_agreement=0.99):
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
            f"Qwen->nano verification failed: argmax_agreement={argmax_agreement:.4f} "
            f"< {min_agreement} (max_abs_logit_diff={max_abs_diff:.4g})"
        )
