import torch
import os
import typing

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str|os.PathLike|typing.BinaryIO|typing.IO[bytes]):

    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()

    checkpoint = {
        'model_state': model_state,
        'optimizer_state': optimizer_state,
        'iteration': iteration
    }

    # Save the checkpoint to the specified output
    torch.save(checkpoint, out)

def load_checkpoint(src: str|os.PathLike|typing.BinaryIO|typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    checkpoint = torch.load(src, map_location=model.device if hasattr(model, 'device') else None)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint['iteration']