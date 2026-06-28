
import numpy as np
import torch

def sample_data(x: np.ndarray, batch_size: int, context_length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of data from the input array x.

    Args:
        x (np.ndarray): Input data array.
        batch_size (int): Number of samples in the batch.
        context_length (int): Length of the context window.
        device (torch.device): Device to which the tensors will be moved.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the input and target tensors.
    """
    # Randomly select starting indices for the batch
    start_indices = np.random.randint(0, len(x) - context_length, size=batch_size)
    
    # Prepare input and target arrays
    inputs = np.zeros((batch_size, context_length), dtype=np.int64)
    targets = np.zeros((batch_size, context_length), dtype=np.int64)

    for i, start_idx in enumerate(start_indices):
        inputs[i] = x[start_idx:start_idx + context_length]
        targets[i] = x[start_idx + 1:start_idx + 1 + context_length]

    # Convert to PyTorch tensors and move to the specified device
    inputs_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
    targets_tensor = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs_tensor, targets_tensor