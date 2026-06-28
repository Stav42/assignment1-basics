import torch
import torch.nn as nn

from cs336_basics.asgn1_model.RMSNorm import RMSNorm
from cs336_basics.asgn1_model.SwiGLU import SwiGLU
from cs336_basics.asgn1_model.RoPE import RotaryPositionalEmbedding
from cs336_basics.asgn1_model.attention import MultiHeadSelfAttention


class TransformerBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float,
                 device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device=device)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-norm: normalize before each sublayer, add the residual after.
        y = x + self.attn(self.ln1(x), token_positions)
        z = y + self.ffn(self.ln2(y))
        return z
