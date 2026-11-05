# The most powerful regularizer: more data. Same network and training budget,
# different amounts of training data.
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from wtlkit import make_loaders, set_seed, train_one_epoch, evaluate

STEPS = 4000                                    # same number of updates for every size
sizes, out = [500, 1000, 2000, 5000, 10000, 20000, 50000], []
print(f"{'train images':>12s}{'train acc':>11s}{'val acc':>9s}{'gap':>7s}")
for n in sizes:
    set_seed(0)
    train_loader, val_loader, _ = make_loaders(batch_size=64, n_train=n)
    model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = max(1, round(STEPS / len(train_loader)))
    for _ in range(epochs):
        train_one_epoch(model, train_loader, opt, nn.CrossEntropyLoss())
    tr, va = evaluate(model, train_loader)[1], evaluate(model, val_loader)[1]
    out.append({"n": n, "train": tr, "val": va})
    print(f"{n:>12,}{tr:>11.2%}{va:>9.2%}{tr - va:>7.2%}")
json.dump(out, open("data_size.json", "w"))
