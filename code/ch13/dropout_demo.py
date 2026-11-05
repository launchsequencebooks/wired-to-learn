# Dropout in training mode vs evaluation mode.
import torch
from torch import nn

torch.manual_seed(0)
drop = nn.Dropout(p=0.5)
x = torch.ones(2, 10)
drop.train()
print("training mode:\n", drop(x))            # about half zeroed, survivors doubled
big = drop(torch.ones(1, 100_000))
print("mean in training mode:", round(big.mean().item(), 3))   # still about 1
drop.eval()
print("evaluation mode:\n", drop(x))          # identity
