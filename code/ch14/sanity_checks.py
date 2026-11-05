# Two checks to run before any long training job.
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from wtlkit import make_loaders, set_seed

set_seed(0)
train_loader, _, _ = make_loaders(batch_size=64)
model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
loss_fn = nn.CrossEntropyLoss()
X, y = next(iter(train_loader))

# Check 1: the initial loss should be close to -ln(1/10) for 10 balanced classes
with torch.no_grad():
    print(f"initial loss {loss_fn(model(X), y).item():.3f}  (expected about {math.log(10):.3f})")

# Check 2: the network must be able to memorize one small batch perfectly
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for step in range(1, 201):
    opt.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward(); opt.step()
    if step in (1, 50, 200):
        acc = (model(X).argmax(1) == y).float().mean().item()
        print(f"step {step:3d}: loss {loss.item():.4f}  batch accuracy {acc:.0%}")
