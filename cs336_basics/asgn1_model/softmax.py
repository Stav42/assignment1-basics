import torch
from einops import reduce


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # Apply softmax over `dim` with the subtract-the-max trick for numerical stability.
    # einops reduces over named axes, so move the target dim to the last position,
    # reduce there, then move it back.
    x = torch.movedim(x, dim, -1)

    max_v = reduce(x, "... d -> ... 1", "max")
    exp = torch.exp(x - max_v)
    denom = reduce(exp, "... d -> ... 1", "sum")
    out = exp / denom

    return torch.movedim(out, -1, dim)
