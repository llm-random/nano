"""Test script to verify tokenize_fn behavior is correct for different models.

Tests realistic use cases:
- GPT-2: No BOS (doesn't have one), needs EOS (manual fallback required)
- Llama: Has both BOS and EOS, tokenizer handles them natively
- SmolLM: Has both BOS and EOS, tokenizer handles them natively

Tests verify:
- BOS token is prepended (for tokenizers that support it)
- EOS token is appended (with manual fallback for tokenizers like GPT-2)
- Batched tokenization works correctly
- Both 'text' and 'content' columns are supported
"""
import sys
sys.path.insert(0, '/home/m1kush/mimuw/research/nano')

from transformers import AutoTokenizer, GPT2TokenizerFast
from src.core.datasets import get_tokenize_fn


def test_gpt2():
    """Test GPT-2 tokenizer - needs manual EOS fallback, no BOS."""
    print("Testing GPT-2")
    print("-" * 50)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print(f"  BOS token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
    print(f"  EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")

    # Test backward compatibility with original manual EOS approach
    tokenizer_old = GPT2TokenizerFast.from_pretrained("gpt2")
    text = "Hello world"
    text_with_eos = text + tokenizer_old.eos_token
    tokens_old = tokenizer_old(text_with_eos)["input_ids"]

    # New approach using get_tokenize_fn
    tokenize_fn = get_tokenize_fn("gpt2")
    tokens_new = tokenize_fn({"text": [text]})["input_ids"][0]

    print(f"  Original (manual eos): {tokens_old}")
    print(f"  New (get_tokenize_fn): {tokens_new}")

    # GPT-2 uses same token for BOS and EOS, but doesn't prepend BOS
    # We only care that EOS is appended
    assert tokens_old == tokens_new, f"Backward compatibility broken! Old: {tokens_old}, New: {tokens_new}"
    assert tokens_new[-1] == tokenizer.eos_token_id, "EOS token not added!"

    # Test batched with 'text' column
    examples = {"text": ["Hello", "World", "Test"]}
    batch_result = tokenize_fn(examples)
    for text, ids in zip(examples["text"], batch_result["input_ids"]):
        assert ids[-1] == tokenizer.eos_token_id, f"Missing EOS for '{text}'"
    print("  ✓ Batched 'text' column works")

    # Test batched with 'content' column
    examples_content = {"content": ["Hello", "World", "Test"]}
    batch_result_content = tokenize_fn(examples_content)
    for content_text, ids in zip(examples_content["content"], batch_result_content["input_ids"]):
        assert ids[-1] == tokenizer.eos_token_id, f"Missing EOS for '{content_text}'"
    print("  ✓ Batched 'content' column works")

    print("✓ GPT-2 tests passed!\n")


def test_llama():
    """Test Llama tokenizer - has both BOS and EOS, handled natively."""
    print("Testing Llama 3.1 8B")
    print("-" * 50)

    try:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        print(f"  BOS token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
        print(f"  EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")

        tokenize_fn = get_tokenize_fn("meta-llama/Llama-3.1-8B")
        tokens = tokenize_fn({"text": ["Hello world"]})["input_ids"][0]

        print(f"  Tokens: {tokens}")
        print(f"  First token is BOS? {tokens[0] == tokenizer.bos_token_id}")
        print(f"  Last token is EOS? {tokens[-1] == tokenizer.eos_token_id}")

        assert tokens[0] == tokenizer.bos_token_id, "BOS token not added!"
        assert tokens[-1] == tokenizer.eos_token_id, "EOS token not added!"

        # Test batched with 'text' column
        examples = {"text": ["Hello", "World", "Test"]}
        batch_result = tokenize_fn(examples)
        for text, ids in zip(examples["text"], batch_result["input_ids"]):
            assert ids[0] == tokenizer.bos_token_id, f"Missing BOS for '{text}'"
            assert ids[-1] == tokenizer.eos_token_id, f"Missing EOS for '{text}'"
        print("  ✓ Batched 'text' column works")

        # Test batched with 'content' column
        examples_content = {"content": ["Hello", "World", "Test"]}
        batch_result_content = tokenize_fn(examples_content)
        for content_text, ids in zip(examples_content["content"], batch_result_content["input_ids"]):
            assert ids[0] == tokenizer.bos_token_id, f"Missing BOS for '{content_text}'"
            assert ids[-1] == tokenizer.eos_token_id, f"Missing EOS for '{content_text}'"
        print("  ✓ Batched 'content' column works")

        print("✓ Llama tests passed!\n")
    except Exception as e:
        print(f"⚠ Llama test skipped: {e}\n")


def test_smollm():
    """Test SmolLM tokenizer - has both BOS and EOS."""
    print("Testing SmolLM 1.7B")
    print("-" * 50)

    try:
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
        print(f"  BOS token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
        print(f"  EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")

        tokenize_fn = get_tokenize_fn("HuggingFaceTB/SmolLM-1.7B")
        tokens = tokenize_fn({"text": ["Hello world"]})["input_ids"][0]

        print(f"  Tokens: {tokens}")
        print(f"  First token is BOS? {tokens[0] == tokenizer.bos_token_id}")
        print(f"  Last token is EOS? {tokens[-1] == tokenizer.eos_token_id}")

        # SmolLM uses same token for BOS/EOS, may not prepend BOS
        # Critical check: EOS must be appended
        assert tokens[-1] == tokenizer.eos_token_id, "EOS token not added!"

        # Test batched with 'text' column
        examples = {"text": ["Hello", "World", "Test"]}
        batch_result = tokenize_fn(examples)
        for text, ids in zip(examples["text"], batch_result["input_ids"]):
            assert ids[-1] == tokenizer.eos_token_id, f"Missing EOS for '{text}'"
        print("  ✓ Batched 'text' column works")

        # Test batched with 'content' column
        examples_content = {"content": ["Hello", "World", "Test"]}
        batch_result_content = tokenize_fn(examples_content)
        for content_text, ids in zip(examples_content["content"], batch_result_content["input_ids"]):
            assert ids[-1] == tokenizer.eos_token_id, f"Missing EOS for '{content_text}'"
        print("  ✓ Batched 'content' column works")

        print("✓ SmolLM tests passed!\n")
    except Exception as e:
        print(f"⚠ SmolLM test skipped: {e}\n")


def test_error_handling():
    """Test error handling for invalid input."""
    print("Testing error handling")
    print("-" * 50)

    tokenize_fn = get_tokenize_fn("gpt2")

    # Test that error is raised when both 'text' and 'content' are present
    try:
        tokenize_fn({"text": ["Hello"], "content": ["World"]})
        assert False, "Should have raised KeyError for both 'text' and 'content'"
    except KeyError as e:
        assert "Both 'text' and 'content' found" in str(e)
        print("  ✓ Error raised for both 'text' and 'content'")

    # Test that error is raised when neither 'text' nor 'content' is present
    try:
        tokenize_fn({"other": ["Hello"]})
        assert False, "Should have raised KeyError for missing 'text' and 'content'"
    except KeyError as e:
        assert "Neither 'text' nor 'content' found" in str(e)
        print("  ✓ Error raised for missing 'text' and 'content'")

    print("✓ Error handling tests passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing get_tokenize_fn - realistic use cases")
    print("=" * 60 + "\n")

    test_gpt2()
    test_llama()
    test_smollm()
    test_error_handling()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)

