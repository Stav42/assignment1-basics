import torch
from collections.abc import Iterable, Callable
from typing import Optional
import math



class SGD(torch.optim.Optimizer):

    def __init__(self, params: dict, lr=1e-3, device: torch.device | None = None, dtype: torch.dtype | None = None):
        self.params = params
        self.device = device
        self.dtype = dtype
        self.lr = lr
        if self.lr<0:
            raise ValueError("Learning rate must be non-negative")
        
        defaults = {
            'lr': lr,
            'device': device,
            'dtype': dtype,
        }

        super().__init__(params, defaults)


    def step(self, closure: Optional[Callable]=None) -> None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get('t', 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state['t'] = t + 1

        return loss 