# A deep (6 hidden layers) MLP on MNIST at a modest and a high learning rate:
# default init without BatchNorm, He init without BatchNorm, and with BatchNorm.
# The He row separates the effect of initialization from the effect of BatchNorm.
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from wtlkit import make_loaders, set_seed, fit


def deep_mlp(batchnorm, depth=6, width=256):
    layers, n = [], 784
    for _ in range(depth):
        layers.append(nn.Linear(n, width))
        if batchnorm:
            layers.append(nn.BatchNorm1d(width))       # normalize, then activate
        layers.append(nn.ReLU())
        n = width
    return nn.Sequential(*layers, nn.Linear(n, 10))


CONFIGS = [(False, "default"), (False, "he"), (True, "default")]   # (BatchNorm, init)

results = {}
print(f"{'configuration':34s}{'epoch 1 acc':>12s}{'epoch 5 acc':>12s}")
for lr in [0.01, 0.1]:
    for bn, init in CONFIGS:
        set_seed(0)
        train_loader, val_loader, _ = make_loaders(batch_size=128)
        model = deep_mlp(bn)
        if init == "he":
            for m in model:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    nn.init.zeros_(m.bias)
        opt = torch.optim.SGD(model.parameters(), lr=lr)          # plain SGD
        hist = fit(model, train_loader, val_loader, opt, epochs=5, verbose=0)
        name = f"lr {lr}, {'BatchNorm' if bn else init + ' init, no BatchNorm'}"
        results[name] = hist
        print(f"{name:34s}{hist['val_acc'][0]:>12.2%}{hist['val_acc'][-1]:>12.2%}")
json.dump(results, open("bn_histories.json", "w"))
