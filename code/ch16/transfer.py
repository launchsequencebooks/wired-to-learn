# Transfer learning: features learned on five kinds of clothing help to recognize five
# *different* kinds from only 20 labeled examples each.
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from wtlkit import mnist_tensors, set_seed, fit, evaluate, FASHION_CLASSES
from resnet import ResNet


class FrozenFeatures(ResNet):
    """A ResNet whose pretrained layers stay in eval mode, so BatchNorm's running
    statistics are not changed by the tiny target batches."""
    def train(self, mode=True):
        super().train(mode)
        self.stem.eval(); self.blocks.eval()
        return self


(X_tr, y_tr), (X_va, y_va), (X_te, y_te) = mnist_tensors(dataset="fashion", image=True)
src, tgt = [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]
print("source classes:", [FASHION_CLASSES[c] for c in src])
print("target classes:", [FASHION_CLASSES[c] for c in tgt])


def subset(X, y, classes, per_class=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = []
    for k, c in enumerate(classes):
        ids = (y == c).nonzero().squeeze(1)
        ids = ids[torch.randperm(len(ids), generator=g)][:per_class]
        idx.append(ids)
    idx = torch.cat(idx)
    remap = torch.full((10,), -1, dtype=torch.long)
    remap[classes] = torch.arange(len(classes))
    return X[idx], remap[y[idx]]


loader = lambda X, y, bs=128, shuffle=False: DataLoader(TensorDataset(X, y), batch_size=bs, shuffle=shuffle)

# 1. Pretrain on the source task (all ~25,000 source images)
set_seed(0)
base = ResNet(n=1, num_classes=5)
opt = torch.optim.SGD(base.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
fit(base, loader(*subset(X_tr, y_tr, src), shuffle=True), loader(*subset(X_va, y_va, src)),
    opt, epochs=3, verbose=0)
print(f"source task accuracy: {evaluate(base, loader(*subset(X_te, y_te, src)))[1]:.2%}")
torch.save(base.state_dict(), "source_model.pt")

# 2. The target task: only 20 labeled images per class
Xs, ys = subset(X_tr, y_tr, tgt, per_class=20)
target_test = loader(*subset(X_te, y_te, tgt))
print(f"target training images: {len(Xs)}")


def run(mode):
    set_seed(1)
    net = FrozenFeatures if mode == "frozen features" else ResNet
    model = net(n=1, num_classes=5)
    if mode != "from scratch":
        model.load_state_dict(torch.load("source_model.pt"))
        model.head[-1] = nn.Linear(64, 5)                  # new classifier for new classes
    if mode == "frozen features":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    lr = 0.01 if mode == "fine-tune all" else 0.05
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    fit(model, loader(Xs, ys, bs=20, shuffle=True), target_test, opt, epochs=30, verbose=0)
    return evaluate(model, target_test)[1]


results = {m: run(m) for m in ["from scratch", "frozen features", "fine-tune all"]}
for m, acc in results.items():
    print(f"{m:16s} target test accuracy {acc:.2%}")
json.dump(results, open("transfer_results.json", "w"))
