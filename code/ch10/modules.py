# Building networks with nn.Module and nn.Sequential.
import torch
from torch import nn


class DigitMLP(nn.Module):
    """The 784-256-128-10 network of Chapter 9, the PyTorch way."""

    def __init__(self, n_in=784, hidden=(256, 128), n_out=10):
        super().__init__()
        self.fc1 = nn.Linear(n_in, hidden[0])        # TinyNet: Dense
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.out = nn.Linear(hidden[1], n_out)

    def forward(self, x):                            # no backward() needed!
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)                           # raw logits


if __name__ == "__main__":
    model = DigitMLP()
    print(model)
    n_params = sum(p.numel() for p in model.parameters())
    print("parameters:", f"{n_params:,}")

    # The same architecture as a one-liner
    seq = nn.Sequential(nn.Linear(784, 256), nn.ReLU(),
                        nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 10))
    print("Sequential parameters:", f"{sum(p.numel() for p in seq.parameters()):,}")

    logits = model(torch.randn(64, 784))                 # call the module, not .forward
    print("output shape:", tuple(logits.shape))
    for name, p in model.named_parameters():
        print(f"  {name:10s} {str(tuple(p.shape)):12s} requires_grad={p.requires_grad}")
