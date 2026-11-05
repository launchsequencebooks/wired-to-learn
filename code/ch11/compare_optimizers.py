# Optimizer shoot-out on MNIST: the same network and data for each, repeated
# over three seeds so that the spread (the noise floor) is printed next to the mean.
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from wtlkit import make_loaders, set_seed, fit, evaluate


def mlp():
    return nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128),
                         nn.ReLU(), nn.Linear(128, 10))


OPTIMIZERS = {
    "SGD (lr 0.05)":              lambda p: torch.optim.SGD(p, lr=0.05),
    "SGD + momentum":             lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9),
    "Nesterov":                   lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9,
                                                            nesterov=True),
    "AdaGrad (lr 0.01)":          lambda p: torch.optim.Adagrad(p, lr=0.01),
    "RMSProp (lr 0.001)":         lambda p: torch.optim.RMSprop(p, lr=1e-3),
    "Adam (lr 0.001)":            lambda p: torch.optim.Adam(p, lr=1e-3),
    "AdamW (lr 0.001, wd 0.01)":  lambda p: torch.optim.AdamW(p, lr=1e-3,
                                                              weight_decay=0.01),
}
SEEDS = (0, 1, 2)
EPOCHS = 8

results = {}          # seed-0 histories, plotted in Figure 11.3
print(f"{'optimizer':28s}{'val acc (3 seeds)':>19s}"
      f"{'val loss':>10s}{'best epoch':>12s}")
for name, make_opt in OPTIMIZERS.items():
    accs = []
    for seed in SEEDS:
        set_seed(seed)
        train_loader, val_loader, _ = make_loaders(batch_size=64, seed=seed)
        model = mlp()
        hist = fit(model, train_loader, val_loader, make_opt(model.parameters()),
                   epochs=EPOCHS, verbose=0)
        accs.append(hist["val_acc"][-1])
        if seed == 0:                   # loss and best epoch are reported for seed 0
            results[name] = hist
            best = min(range(EPOCHS), key=lambda i: hist["val_loss"][i]) + 1
            val_loss = hist["val_loss"][-1]
    mean, half_range = sum(accs) / len(accs), (max(accs) - min(accs)) / 2
    print(f"{name:28s}{mean:>11.2%} ±{half_range:5.2%}{val_loss:>10.4f}{best:>12d}")
json.dump(results, open("optimizer_histories.json", "w"))
