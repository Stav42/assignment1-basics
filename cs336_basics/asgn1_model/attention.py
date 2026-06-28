import math

import torch
import torch.nn as nn
from einops import einsum, rearrange

from cs336_basics.asgn1_model.softmax import softmax
from cs336_basics.asgn1_model.Linear import Linear


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # Q: (..., queries, d_k), K: (..., keys, d_k), V: (..., keys, d_v)
    # mask: (..., queries, keys) where True = attend, False = do not attend.
    d_k = Q.shape[-1]

    scores = einsum(Q, K, "... q d_k, ... k d_k -> ... q k") / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = softmax(scores, dim=-1)

    return einsum(weights, V, "... q k, ... k d_v -> ... q d_v")


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, rope=None, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # = d_v
        self.rope = rope

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # x: (..., seq_len, d_model)
        seq_len = x.shape[-2]

        # Project, then split the d_model dim into (num_heads, d_k), moving heads
        # to a batch-like position so attention runs independently per head.
        Q = rearrange(self.q_proj(x), "... seq (h d) -> ... h seq d", h=self.num_heads)
        K = rearrange(self.k_proj(x), "... seq (h d) -> ... h seq d", h=self.num_heads)
        V = rearrange(self.v_proj(x), "... seq (h d) -> ... h seq d", h=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            # Add a head axis so the same rotation broadcasts across all heads.
            tp = rearrange(token_positions, "... seq -> ... 1 seq")
            Q = self.rope(Q, tp)
            K = self.rope(K, tp)

        # Causal mask: query i may attend to key j only when j <= i.
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        attn = scaled_dot_product_attention(Q, K, V, mask)  # (..., h, seq, d_v)

        # Concatenate heads back into d_model and apply the output projection.
        attn = rearrange(attn, "... h seq d -> ... seq (h d)")
        return self.output_proj(attn)
