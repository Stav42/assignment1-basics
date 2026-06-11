import torch.nn as nn
import torch
from einops import reduce, rearrange
import math

class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        g_params = torch.ones(d_model, device=self.device, dtype=self.dtype)
        self.g = nn.Parameter(g_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x: (batch_size, sequence_length, d_model) - Input shape
        # a: is R^d vector. 
        # x normalization should be done on the innermost vector

        in_type = x.dtype
        x = x.to(torch.float32)
        den = torch.sqrt(( 1/self.d_model)*(reduce(x**2, "b s d_model -> b s", "sum")) + self.eps )
        den = rearrange(den, "b s -> b s 1")
        result = self.g * x/den

        return result.to(in_type)
    
