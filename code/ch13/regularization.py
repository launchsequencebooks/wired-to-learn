# Fighting overfitting on a small training set (5,000 MNIST images):
# compare regularization techniques under identical conditions.
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from wtlkit import make_loaders, set_seed, fit, evaluate
from augment import random_affine


def mlp(p_drop=0.0):
    return nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Dropout(p_drop),
                         nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p_drop),
                         nn.Linear(512, 10))


CONFIGS = {
    "baseline":            dict(),
    "weight decay 0.1":    dict(wd=0.1),
    "dropout 0.5":         dict(p_drop=0.5),
    "label smoothing 0.1": dict(smooth=0.1),
    "augmentation":        dict(augment=True),
    "all combined":        dict(wd=0.1, p_drop=0.5, smooth=0.1, augment=True),
}
EPOCHS, results = 40, {}
print(f"{'technique':22s}{'train acc':>10s}{'best val acc':>13s}{'final val acc':>14s}{'best epoch':>11s}")
for name, c in CONFIGS.items():
    set_seed(0)
    train_loader, val_loader, _ = make_loaders(batch_size=64, n_train=5000)
    model = mlp(c.get("p_drop", 0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=c.get("wd", 0.0))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=c.get("smooth", 0.0))
    hist = fit(model, train_loader, val_loader, opt, epochs=EPOCHS, loss_fn=loss_fn,
               augment=random_affine if c.get("augment") else None, verbose=0)
    train_acc = evaluate(model, train_loader)[1]
    best = max(range(EPOCHS), key=lambda i: hist["val_acc"][i])
    results[name] = hist
    print(f"{name:22s}{train_acc:>10.2%}{hist['val_acc'][best]:>13.2%}"
          f"{hist['val_acc'][-1]:>14.2%}{best + 1:>11d}")
json.dump(results, open("regularization_histories.json", "w"))
