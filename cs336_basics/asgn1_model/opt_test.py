import torch

from cs336_basics.asgn1_model.Optimizer import SGD

if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1.8)

    for t in range(10):
        opt.zero_grad()
        loss = (weights ** 2).sum()
        print(loss.cpu().item())
        loss.backward()
        opt.step()