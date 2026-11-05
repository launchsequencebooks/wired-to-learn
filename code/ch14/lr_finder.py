# The learning-rate range test: raise the learning rate exponentially during one
# short run and watch where the loss falls fastest and where it blows up.
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from wtlkit import make_loaders, set_seed


def lr_range_test(model, loader, make_opt, lr_min=1e-6, lr_max=10.0, steps=300, smooth=0.95):
    opt = make_opt(model.parameters(), lr_min)
    factor = (lr_max / lr_min) ** (1 / steps)
    loss_fn, lrs, losses, avg, best = nn.CrossEntropyLoss(), [], [], 0.0, float("inf")
    it = iter(loader)
    for step in range(steps):
        try:
            X, y = next(it)
        except StopIteration:
            it = iter(loader); X, y = next(it)
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward(); opt.step()
        avg = smooth * avg + (1 - smooth) * loss.item()          # smoothed loss
        smoothed = avg / (1 - smooth ** (step + 1))
        lrs.append(opt.param_groups[0]["lr"]); losses.append(smoothed)
        best = min(best, smoothed)
        if smoothed > 4 * best:                                  # diverged: stop
            break
        for g in opt.param_groups:
            g["lr"] *= factor
    return lrs, losses


set_seed(0)
train_loader, _, _ = make_loaders(batch_size=64)
model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))
lrs, losses = lr_range_test(model, train_loader,
                            lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9))
i_min = min(range(len(losses)), key=losses.__getitem__)
slopes = [(losses[i + 10] - losses[i]) / (math.log10(lrs[i + 10]) - math.log10(lrs[i]))
          for i in range(len(losses) - 10)]
i_steep = min(range(len(slopes)), key=slopes.__getitem__)
print(f"tested {len(lrs)} learning rates from {lrs[0]:.0e} to {lrs[-1]:.2f}")
print(f"lowest loss at lr = {lrs[i_min]:.3f}; steepest descent around lr = {lrs[i_steep + 5]:.3f}")
print(f"suggested learning rate (lowest-loss lr / 10): {lrs[i_min] / 10:.3f}")
json.dump({"lr": lrs, "loss": losses, "min": lrs[i_min], "steep": lrs[i_steep + 5]},
          open("lr_finder.json", "w"))
