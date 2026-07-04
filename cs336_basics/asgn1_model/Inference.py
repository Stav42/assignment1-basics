import torch
import cs336_basics.asgn1_model.TransformerLM as TransformerLM
from cs336_basics.asgn1_model.softmax import softmax
import cs336_basics.tokenizer as Tokenizer

def decode(
    model: TransformerLM,
    x: torch.tensor,
    max_tokens: int,
    lambda_: float,
    vocab: dict[int, bytes],
    threshold: float | None = None,
    tokenizer: Tokenizer.Tokenizer | None = None,
) -> str:

    valid = True
    token_len = 0
    model.eval()

    # Resolve the <|endoftext|> id once so we can stop when the model emits it.
    eos_bytes = b"<|endoftext|>"
    eos_id = next((tid for tid, b in vocab.items() if b == eos_bytes), None)

    with torch.no_grad():
        x = x.to(model.device)

        while valid:
            # The model was trained with a fixed context window; RoPE and the
            # causal mask are sized to context_length, so only the most recent
            # context_length tokens can be fed back in as the sequence grows.
            v = model(x[:, -model.context_length:])
            next_dist = v[:, -1, :]

            dist = softmax(next_dist/lambda_, dim=-1)

            dist_pair = [[i, x] for i, x in enumerate(dist[0].tolist())]
            
            if threshold is not None:
                dist_pair = sorted(dist_pair, key=lambda x: x[1], reverse=True)

                cumulative = 0.0
                for i, (_, probability) in enumerate(dist_pair):
                    cumulative += probability
                    if cumulative > threshold:
                        dist_pair[i+1:] = [[j, 0] for j, _ in dist_pair[i+1:]]
                        break

            dist_pair = sorted(dist_pair, key=lambda x: x[0])
            val_index = [x[1] for x in dist_pair]

            next_token = torch.multinomial(torch.tensor(val_index), 1).item()

            # if vocab is not None:
            #     next_token = vocab[next_token]

            x = torch.cat((x, x.new_tensor([[next_token]])), dim=1)

            token_len += 1
            if token_len >= max_tokens:
                valid = False

            if next_token == eos_id:
                valid = False
        

    if tokenizer is None:
        raise ValueError("decode() requires a tokenizer instance for detokenization")

    decoded_tokens = tokenizer.decode(x[0].tolist())

    return decoded_tokens

