from transformers import GPT2TokenizerFast


def gpt2_mask_tokenize_fn():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Add additional special tokens so that the vocab size is a multiple of 64
    # https://x.com/karpathy/status/1621578354024677377
    current_size = tokenizer.vocab_size
    diff_multiple_64 = (((current_size // 64) + 1) * 64 - current_size) % 64
    tokens_to_add = diff_multiple_64 - 1  # -1 for the mask token
    additional_special_tokens = [f"<|extra_token_{i}|>" for i in range(tokens_to_add)]
    tokenizer.add_special_tokens(
        {
            "mask_token": "<|mask|>",
            "additional_special_tokens": additional_special_tokens,
        }
    )

    def tokenize_function(examples):
        texts = [text + tokenizer.eos_token for text in examples["text"]]
        batch_encodings = tokenizer(
            texts,
            truncation=False,
            max_length=int(1e10),
        )
        return batch_encodings

    return tokenize_function
