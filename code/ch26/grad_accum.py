# Gradient accumulation: a big batch's gradient from several small ones.
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path += [str(root), str(root / "ch16")]
import torch
from torch import nn
from resnet import ResNet

torch.manual_seed(0)
X, y = torch.randn(128, 1, 28, 28), torch.randint(0, 10, (128,))
loss_fn = nn.CrossEntropyLoss()


def grads(model):
    return torch.cat([p.grad.flatten() for p in model.parameters()])


def compare(make_model):
    torch.manual_seed(1); model = make_model()
    loss_fn(model(X), y).backward()                       # one batch of 128
    g_big = grads(model).clone()
    model.zero_grad()
    for Xm, ym in zip(X.split(32), y.split(32)):           # four micro-batches of 32
        (loss_fn(model(Xm), ym) / 4).backward()            # divide: we want the mean, not the sum
    g_acc = grads(model)
    return (g_big - g_acc).abs().max().item() / g_big.abs().max().item()


mlp = lambda: nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
print(f"MLP:       largest relative difference {compare(mlp):.1e}")
print(f"ResNet-20: largest relative difference {compare(lambda: ResNet(n=3)):.1e}")
