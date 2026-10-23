"""
Regression test for the tie_word_embeddings corruption bug in load_pc_state_dict_to_llama.

LLaMA 3.2-1B has tie_word_embeddings=True. Without the fix, load_state_dict silently
overwrites embed_tokens.weight with lm_head.weight (lm_head is processed last in
named_parameters because they're the same tensor object). This corrupts the embedding.
"""

import torch
import pytest
from src.projected_compression.convert_memeff_to_hf import load_pc_state_dict_to_llama


def _make_state_dict(dmodel, dff, n_layers, q_heads, kv_heads, head_dim, vocab_size):
    sd = {
        "embedding": torch.ones(vocab_size, dmodel),
        "head.norm.weight": torch.ones(dmodel),
        "head.linear.weight": torch.full(
            (vocab_size, dmodel), 2.0
        ),  # deliberately different
    }
    for layer in range(n_layers):
        p = f"encoder.blocks.{layer}."
        sd[f"{p}attention_layer.norm.weight"] = torch.ones(dmodel)
        sd[f"{p}attention_layer.layer.q_proj.weight"] = torch.ones(
            q_heads * head_dim, dmodel
        )
        sd[f"{p}attention_layer.layer.k_proj.weight"] = torch.ones(
            kv_heads * head_dim, dmodel
        )
        sd[f"{p}attention_layer.layer.v_proj.weight"] = torch.ones(
            kv_heads * head_dim, dmodel
        )
        sd[f"{p}attention_layer.layer.o_proj.weight"] = torch.ones(
            dmodel, q_heads * head_dim
        )
        sd[f"{p}ff_layer.norm.weight"] = torch.ones(dmodel)
        sd[f"{p}ff_layer.layer.gate.weight"] = torch.ones(dff, dmodel)
        sd[f"{p}ff_layer.layer.ff_pre_act.weight"] = torch.ones(dff, dmodel)
        sd[f"{p}ff_layer.layer.ff_post_act.weight"] = torch.ones(dmodel, dff)
    return sd


def test_embedding_not_corrupted_for_1b():
    """embed_tokens must keep the embedding values, not be overwritten by lm_head."""
    sd = _make_state_dict(
        dmodel=16,
        dff=32,
        n_layers=16,
        q_heads=32,
        kv_heads=8,
        head_dim=64,
        vocab_size=128256,
    )
    model = load_pc_state_dict_to_llama(sd, "meta-llama/Llama-3.2-1B")

    emb = model.model.embed_tokens.weight.detach()
    lm_head = model.lm_head.weight.detach()

    assert torch.all(
        emb == 1.0
    ), "embed_tokens.weight was corrupted (expected all-ones)"
    assert torch.all(
        lm_head == 2.0
    ), "lm_head.weight was not loaded correctly (expected all-twos)"
    assert not torch.equal(
        emb, lm_head
    ), "embed_tokens and lm_head should be independent"


def test_embedding_not_corrupted_for_8b():
    """Same check for 8B — should already work, confirms no regression."""
    sd = _make_state_dict(
        dmodel=16,
        dff=32,
        n_layers=32,
        q_heads=32,
        kv_heads=8,
        head_dim=128,
        vocab_size=128256,
    )
    model = load_pc_state_dict_to_llama(sd, "meta-llama/Llama-3.1-8B")

    emb = model.model.embed_tokens.weight.detach()
    lm_head = model.lm_head.weight.detach()

    assert torch.all(emb == 1.0), "embed_tokens.weight corrupted for 8B"
    assert torch.all(lm_head == 2.0), "lm_head.weight not loaded correctly for 8B"
    assert not torch.equal(
        emb, lm_head
    ), "embed_tokens and lm_head should be independent"
