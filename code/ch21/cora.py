# Classifying scientific papers in the Cora citation graph: does the graph help?
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from wtlkit import load_cora, set_seed
from gcn import GCN, normalized_adjacency

g = load_cora()
X = torch.tensor(g["x"]); X = X / X.sum(1, keepdim=True).clamp(min=1)   # row-normalize words
y = torch.tensor(g["y"]); n = len(y)
A_norm = normalized_adjacency(torch.tensor(g["edge_index"]), n)
tr, va, te = (torch.tensor(g[k]) for k in ("train", "val", "test"))
print(f"{n} papers, {g['edge_index'].shape[1] // 2} citation links, {X.shape[1]} word features, "
      f"{int(y.max()) + 1} topics")
print(f"labeled for training: {len(tr)} | validation: {len(va)} | test: {len(te)}")


class MLP(nn.Module):                                   # the same network, minus the graph
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(0.5), nn.Linear(X.shape[1], 16), nn.ReLU(),
                                 nn.Dropout(0.5), nn.Linear(16, 7))

    def forward(self, x, A_norm=None):
        return self.net(x)


def run(model, epochs=200):
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val, best_test = 0, 0
    for epoch in range(epochs):
        model.train()
        loss = nn.functional.cross_entropy(model(X, A_norm)[tr], y[tr])
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(X, A_norm).argmax(1)
        val, test = ((pred[s] == y[s]).float().mean().item() for s in (va, te))
        if val > best_val:
            best_val, best_test = val, test                  # model selection on validation
    return best_val, best_test


results = {}
for name, make in [("MLP (words only)", MLP), ("GCN (words + citations)", lambda: GCN(X.shape[1], 16, 7))]:
    accs = []
    for seed in range(5):
        set_seed(seed)
        accs.append(run(make())[1])
    t = torch.tensor(accs)
    results[name] = accs
    print(f"{name:26s} test accuracy {t.mean():.1%} +/- {t.std():.1%} over 5 seeds")
json.dump(results, open("cora_results.json", "w"))
