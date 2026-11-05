# Why skip connections help: the gradient reaching the FIRST layer at initialization,
# for plain and residual stacks of increasing depth (PyTorch's default initialization,
# no normalization). Residual branches are scaled by 1/sqrt(depth), a common practice
# that keeps the sum of many branches from growing.
import torch
from torch import nn


def first_layer_grad(depth, residual):
    torch.manual_seed(0)
    layers = [nn.Linear(64, 64) for _ in range(depth)]
    h = torch.randn(128, 64)
    for lin in layers:
        out = torch.relu(lin(h))
        h = h + out / depth ** 0.5 if residual else out      # the skip connection
    h.pow(2).mean().backward()
    return layers[0].weight.grad.norm().item(), h.std().item()


print(f"{'depth':>6s}{'plain: grad':>14s}{'signal':>10s}{'residual: grad':>17s}{'signal':>10s}")
for depth in [5, 20, 50, 100]:
    (gp, sp), (gr, sr) = first_layer_grad(depth, False), first_layer_grad(depth, True)
    print(f"{depth:>6d}{gp:>14.1e}{sp:>10.2f}{gr:>17.1e}{sr:>10.2f}")
