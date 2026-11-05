# Gradient checkpointing: trade compute for memory.
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path += [str(root), str(root / "ch16")]
import torch
from torch.utils.checkpoint import checkpoint
from resnet import ResNet
from wtlkit import benchmark


class CheckpointedResNet(ResNet):
    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:                          # keep only each block's input;
            x = checkpoint(block, x, use_reentrant=False)  # recompute its insides during backward
        return self.head(x)


def saved_megabytes(model, X):
    """Total size of the tensors autograd keeps for the backward pass."""
    total = 0

    def pack(t):
        nonlocal total
        total += t.numel() * t.element_size()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = model(X).sum()
    loss.backward()
    return total / 1e6


torch.manual_seed(0)
X = torch.randn(128, 1, 28, 28)
for name, model in [("normal", ResNet(n=3)), ("checkpointed", CheckpointedResNet(n=3))]:
    mb = saved_megabytes(model, X)

    def step():
        model.zero_grad()
        model(X).sum().backward()
    print(f"{name:13s} activations kept for backward {mb:6.1f} MB   step {benchmark(step, 2, 5):4.0f} ms")
