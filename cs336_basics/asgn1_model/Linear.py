import torch.nn as nn
import torch
from einops import einsum
import math

class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device: torch.device | None = device
        self.dtype: torch.dtype | None = dtype

        W = torch.empty(out_features, in_features, device=self.device, dtype=self.dtype)
        sigma_2 = 2/(in_features+out_features)
        sigma = math.sqrt(sigma_2)
        nn.init.trunc_normal_(W, 0, sigma, -3*sigma, 3*sigma)
        ## Initialize the weight and bias parameters
        self.W = nn.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ## Apply the linear transformation
        
        return  einsum(self.W, x,
         "out_features in_features, ... in_features -> ... out_features")
    


