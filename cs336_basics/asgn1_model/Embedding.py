import torch.nn as nn
import torch
from einops import einsum

class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        weight = torch.empty(num_embeddings, embedding_dim, device=self.device, dtype=self.dtype)

        ## Initialize
        nn.init.trunc_normal_(weight, 0, 1, -3, 3)
        self.weight = nn.Parameter(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Draw out corresponding indexes to get the embedding matrix
        out = self.weight[token_ids.long()]
        return out





