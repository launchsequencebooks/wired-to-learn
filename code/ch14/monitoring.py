# Watching the inside of training: gradient norms per layer, the ratio of
# update size to weight size, and the fraction of dead ReLU units.
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from wtlkit import make_loaders, set_seed

def run(lr, steps=300):
    set_seed(0)
    train_loader, _, _ = make_loaders(batch_size=64)
    model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    linears = [m for m in model if isinstance(m, nn.Linear)]
    log = {"loss": [], "grad_norm": [], "update_ratio": [], "dead": []}
    it = iter(train_loader)
    for step in range(steps):
        X, y = next(it)
        before = [l.weight.detach().clone() for l in linears]
        opt.zero_grad()
        h1 = model[1](model[0](X))                      # first hidden activations
        loss = nn.functional.cross_entropy(model[2:](h1), y)
        loss.backward(); opt.step()
        log["loss"].append(loss.item())
        log["grad_norm"].append([l.weight.grad.norm().item() for l in linears])
        log["update_ratio"].append([((l.weight - b).norm() / b.norm()).item()
                                    for l, b in zip(linears, before)])
        log["dead"].append((h1.max(0).values == 0).float().mean().item())
    return log

logs = {}
for lr in [0.01, 1.0]:
    log = run(lr); logs[str(lr)] = log
    print(f"lr {lr:<4}: final loss {log['loss'][-1]:.3f}  "
          f"grad norm (layer 1) {log['grad_norm'][-1][0]:.1e}  "
          f"update/weight ratio (layer 1) "
          f"{sum(r[0] for r in log['update_ratio'][-50:]) / 50:.1e}  "
          f"dead units in layer 1 {log['dead'][-1]:.0%}")
json.dump(logs, open("monitoring.json", "w"))
