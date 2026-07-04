import argparse

import torch

import cs336_basics.tokenizer as Tokenizer
from cs336_basics.asgn1_model.Inference import decode
from cs336_basics.asgn1_model.TransformerLM import TransformerLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint")

    # Checkpoint + tokenizer
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/July1Run3/checkpoint_final.pt")
    parser.add_argument("--vocab", type=str,
                        default="cs336_basics/tinystories_train/vocab_dict.pkl")
    parser.add_argument("--merges", type=str,
                        default="cs336_basics/tinystories_train/merges_bytes.pkl")

    # Sampling
    parser.add_argument("--prompt", type=str, default="Akbar was a king")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)

    # Model config (matches the July1Run3 checkpoint / train.py defaults)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # Build the model and load the trained weights.
    model = TransformerLM(
        args.vocab_size, args.context_length, args.d_model,
        args.num_layers, args.num_heads, args.d_ff, args.rope_theta,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    # decode() reads model.device directly (point 2 left unfixed), so expose it.
    model.device = device
    model.eval()

    tokenizer = Tokenizer.Tokenizer.from_files(
        args.vocab, args.merges, special_tokens=["<|endoftext|>"],
    )

    prompt_ids = tokenizer.encode(args.prompt)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    text = decode(
        model, x,
        max_tokens=args.max_tokens,
        lambda_=args.temperature,
        vocab=tokenizer.vocab,
        threshold=args.top_p,
        tokenizer=tokenizer,
    )

    print(text)


if __name__ == "__main__":
    main()
