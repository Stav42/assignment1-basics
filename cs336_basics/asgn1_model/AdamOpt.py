import torch
from collections.abc import Iterable, Callable
from typing import Optional
import math



class AdamW(torch.optim.Optimizer):

    def __init__(self, params: dict, lr=1e-3, betas = (0.9, 0.999), weight_decay=0.01, eps=1e-8, device: torch.device | None = None, dtype: torch.dtype | None = None):
        self.params = params
        self.device = device
        self.dtype = dtype
        self.lr = lr
        self.beta_1 = betas[0]
        self.beta_2 = betas[1]
        self.lambda_ = weight_decay
        self.eps = eps
        if self.lr<0:
            raise ValueError("Learning rate must be non-negative")
        
        defaults = {
            'lr': lr,
            'beta_1': self.beta_1,
            'beta_2': self.beta_2,
            'lambda_': self.lambda_,
            'eps': self.eps,
            'device': self.device,
            'dtype': self.dtype,
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
                grad = p.grad.data
                m = state.get('m', torch.zeros_like(p.data))
                v = state.get('v', torch.zeros_like(p.data))
                t = state.get('t', 1)

                lr_t = lr * math.sqrt(1-self.beta_2**t) / (1-self.beta_1**t)
                
                p.data -= lr * self.lambda_ * p.data

                m = self.beta_1 * m + (1 - self.beta_1) * p.grad.data
                v = self.beta_2 * v + (1 - self.beta_2) * (p.grad.data ** 2)

                p.data -= lr_t * m / (torch.sqrt(v) + self.eps)
                state['m'] = m
                state['v'] = v
                state['t'] = t + 1

        return loss 