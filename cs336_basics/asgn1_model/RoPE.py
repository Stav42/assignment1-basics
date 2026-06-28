import torch
import torch.nn as nn
from einops import einsum, rearrange


class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Angles theta_{i,k} = i / theta^((2k)/d_k) for position i and pair k.
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        k = torch.arange(d_k // 2, device=device, dtype=torch.float32)
        freqs = 1.0 / (theta ** (2 * k / d_k))  # (d_k/2,)
        angles = einsum(positions, freqs, "i, k -> i k")  # (max_seq_len, d_k/2)

        # Fixed (non-learnable) sin/cos tables.
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k), token_positions: (..., seq_len)
        cos = self.cos[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin[token_positions]  # (..., seq_len, d_k/2)

        # Split the last dim into consecutive (even, odd) pairs.
        x_pairs = rearrange(x, "... (half two) -> ... half two", two=2)
        x1 = x_pairs[..., 0]
        x2 = x_pairs[..., 1]

        # Rotate each pair by its angle.
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        out = torch.stack((out1, out2), dim=-1)  # (..., seq_len, d_k/2, 2)
        return rearrange(out, "... half two -> ... (half two)")
