# Random search over four hyperparameters, judged on validation accuracy.
import sys, json, math, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from wtlkit import make_loaders, set_seed, fit

rng = random.Random(42)
def sample():
    return {"lr": 10 ** rng.uniform(-4, -1.5),          # log-uniform
            "width": rng.choice([64, 128, 256, 512]),
            "dropout": rng.uniform(0.0, 0.5),
            "weight_decay": 10 ** rng.uniform(-5, -1)}

trials = []
print(f"{'trial':>5s}{'lr':>9s}{'width':>7s}{'dropout':>9s}{'w.decay':>9s}{'val acc':>9s}")
for t in range(1, 13):
    hp = sample()
    set_seed(0)
    train_loader, val_loader, _ = make_loaders(batch_size=128, n_train=20000)
    model = nn.Sequential(nn.Linear(784, hp["width"]), nn.ReLU(), nn.Dropout(hp["dropout"]),
                          nn.Linear(hp["width"], hp["width"]), nn.ReLU(), nn.Dropout(hp["dropout"]),
                          nn.Linear(hp["width"], 10))
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    hist = fit(model, train_loader, val_loader, opt, epochs=4, verbose=0)
    hp["val_acc"] = max(hist["val_acc"]); trials.append(hp)
    print(f"{t:>5d}{hp['lr']:>9.5f}{hp['width']:>7d}{hp['dropout']:>9.2f}"
          f"{hp['weight_decay']:>9.5f}{hp['val_acc']:>9.2%}")
best = max(trials, key=lambda h: h["val_acc"])
print("best:", {k: round(v, 5) if isinstance(v, float) else v for k, v in best.items()})
json.dump(trials, open("random_search.json", "w"))
