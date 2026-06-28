import math

import torch

def lr_schedule(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> float:
    """
    Compute the learning rate at time step t using a cosine annealing schedule.

    Args:
        t (int): The current time step.
        alpha_max (float): The maximum learning rate.
        alpha_min (float): The minimum learning rate.
        T_w (int): The number of time steps for warmup.
        T_c (int): The number of time steps for cosine annealing.

    Returns:
        float: The learning rate at time step t.
    """
    lr = 0.0

    if t < T_w:
        lr = t / T_w * alpha_max
    elif T_w <= t <= T_c:
        lr = alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + math.cos((t - T_w) / (T_c - T_w) * 3.141592653589793))
    else:
        lr = alpha_min

    return lr


def gradient_clip(parameters: torch.Tensor, M: float, eps: float = 1e-6) -> torch.Tensor:
    """
    Clip the gradients to a specified range.

    Args:
        gradients (torch.Tensor): The gradients to be clipped.
        M (float): The maximum l2 norm.
        eps (float): A small value to avoid division by zero.

    Returns:
        torch.Tensor: The clipped gradients.
    """

    gradients = torch.cat([p.grad.flatten() for p in parameters if p.grad is not None])

    norm = torch.norm(gradients)

    if norm > M:
        scaling_factor = M / (norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.data *= scaling_factor

    return gradients
