import torch

def calculate_ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:


    # Find the vocab index corresponding to the target token
    # Is this one target token or many?
    # I believe the targets has to be teh index otherwise this becomes a really expensive 
    numerator = logits[targets]
    denominator = torch.sum(torch.exp(logits))
    loss = -torch.log(numerator / denominator)
    return loss
