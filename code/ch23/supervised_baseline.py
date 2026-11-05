# The fair comparison: the same encoder trained end-to-end WITH labels,
# but only on the few labeled images that the linear probe saw.
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from torch import nn
from wtlkit import mnist_tensors, set_seed
from simclr import Encoder, labeled_subset, augment, summary

(X_tr, y_tr), (X_va, y_va), (X_te, y_te) = mnist_tensors(dataset="fashion", image=True)
results = {}
print(f"{'labels per class':>17s}{'supervised from scratch':>25s}")
for per_class in [10, 100, 1000]:
    accs = []
    for seed in range(5) if per_class < 1000 else [0]:   # 5 labeled subsets when few
        set_seed(seed)
        idx = labeled_subset(y_tr, per_class, seed=seed)
        model = nn.Sequential(Encoder(), nn.Linear(128, 10))
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        steps = 1500 if per_class < 1000 else 5000          # more data, more steps
        for step in range(steps):
            b = idx[torch.randint(len(idx), (min(128, len(idx)),))]
            model.train()
            loss = F.cross_entropy(model(augment(X_tr[b])), y_tr[b])   # same views
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(X_te[i:i + 2000]).argmax(1)
                              for i in range(0, len(X_te), 2000)])
            accs.append((pred == y_te).float().mean().item())
    results[per_class] = accs
    print(f"{per_class:>17d}{summary(accs):>25s}")
json.dump(results, open("supervised_results.json", "w"))
