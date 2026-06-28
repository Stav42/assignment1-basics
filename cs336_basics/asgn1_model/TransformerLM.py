import torch
import torch.nn as nn

from cs336_basics.asgn1_model.Embedding import Embedding
from cs336_basics.asgn1_model.Linear import Linear
from cs336_basics.asgn1_model.RMSNorm import RMSNorm
from cs336_basics.asgn1_model.TransformerBlock import TransformerBlock


class TransformerLM(nn.Module):

    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int,
                 num_heads: int, d_ff: int, rope_theta: float, device=None, dtype=None):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta,
                             device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        # in_indices: (batch, seq_len) -> logits: (batch, seq_len, vocab_size)
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)
