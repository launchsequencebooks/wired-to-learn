# Inductive bias in action: shuffle the pixels of every image with one fixed permutation.
# The information is unchanged - but the spatial structure a CNN relies on is gone.
import sys, json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from wtlkit import mnist_tensors, set_seed, fit, evaluate

(X_tr, y_tr), (X_va, y_va), (X_te, y_te) = mnist_tensors(n_train=20000)
perm = torch.randperm(784, generator=torch.Generator().manual_seed(0))


def loaders(permute):
    f = (lambda X: X[:, perm]) if permute else (lambda X: X)
    g = torch.Generator().manual_seed(0)
    return (DataLoader(TensorDataset(f(X_tr), y_tr), batch_size=128, shuffle=True, generator=g),
            DataLoader(TensorDataset(f(X_va), y_va), batch_size=1024),
            DataLoader(TensorDataset(f(X_te), y_te), batch_size=1024))


def mlp():
    return nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))


def cnn():
    return nn.Sequential(nn.Unflatten(1, (1, 28, 28)),
                         nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(64 * 49, 10))


results = {}
print(f"{'model':6s}{'pixels':>12s}{'params':>9s}{'test acc':>10s}")
for name, make in [("MLP", mlp), ("CNN", cnn)]:
    for permute in [False, True]:
        set_seed(0)
        tr, va, te = loaders(permute)
        model = make()
        fit(model, tr, va, torch.optim.Adam(model.parameters(), lr=1e-3), epochs=4, verbose=0)
        acc = evaluate(model, te)[1]
        key = f"{name}, {'shuffled' if permute else 'normal'}"
        results[key] = acc
        print(f"{name:6s}{'shuffled' if permute else 'normal':>12s}"
              f"{sum(p.numel() for p in model.parameters()):>9,}{acc:>10.2%}")
json.dump({"results": results, "perm": perm.tolist()}, open("inductive_bias.json", "w"))
