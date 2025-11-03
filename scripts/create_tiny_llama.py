from transformers import AutoConfig, LlamaForCausalLM
import argparse

PROJECT_ORG = "pc-project"
MODEL_NAME = "tiny-llama-16M"


def main():
    parser = argparse.ArgumentParser(description="Save or push model")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("local", help="Save model locally")
    subparsers.add_parser("push", help="Push model to Hugging Face Hub")
    args = parser.parse_args()

    conf = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B")

    conf.num_hidden_layers = 2
    conf.num_key_value_heads = 2
    conf.num_attention_heads = 4
    conf._name_or_path = f"{PROJECT_ORG}/{MODEL_NAME}"
    conf.hidden_size = 128
    conf.intermediate_size = 512
    conf.head_dim = 32

    model = LlamaForCausalLM(conf)

    if args.command == "local":
        print("Saving model locally...")
        model.save_pretrained(MODEL_NAME)

    elif args.command == "push":
        print("Pushing model to hub...")
        model.push_to_hub(f"{PROJECT_ORG}/{MODEL_NAME}", private=True)


if __name__ == "__main__":
    main()
