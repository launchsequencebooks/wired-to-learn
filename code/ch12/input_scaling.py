# Why we standardize inputs: the same network on raw and standardized features.
import torch
from torch import nn
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)          # 569 tumors, 30 measurements
print("feature ranges:", f"{X.min(0).min():.4f} to {X.max(0).max():.1f}")
rng = torch.Generator().manual_seed(0)
idx = torch.randperm(len(X), generator=rng)
tr, va = idx[:450], idx[450:]
X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y)

def run(X_tr, X_va, label, lr=0.01, epochs=100):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(30, 32), nn.ReLU(), nn.Linear(32, 2))
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    first = None
    for epoch in range(1, epochs + 1):
        for i in range(0, len(X_tr), 32):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(X_tr[i:i + 32]), y[tr][i:i + 32])
            loss.backward(); opt.step()
        acc = (model(X_va).argmax(1) == y[va]).float().mean().item()
        if first is None and acc >= 0.95:
            first = epoch
    print(f"{label:14s} final val acc {acc:.1%}  loss {loss.item():8.4f}  "
          f"first epoch >= 95%: {first}")

run(X[tr], X[va], "raw features")
mean, std = X[tr].mean(0), X[tr].std(0)              # statistics from training data only
run((X[tr] - mean) / std, (X[va] - mean) / std, "standardized")
