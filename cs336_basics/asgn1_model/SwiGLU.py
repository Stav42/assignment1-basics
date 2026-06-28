import torch.nn as nn
import torch

from cs336_basics.asgn1_model.Linear import Linear


def silu(x: torch.Tensor) -> torch.Tensor:
    # SiLU(x) = x * sigmoid(x)
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        # FFN(x) = W2( SiLU(W1 x) ⊙ W3 x )
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        gate = silu(self.w1(x))
        up = self.w3(x)
        return self.w2(gate * up)
