# Where distillation shines: few labels, many unlabeled images the teacher can label.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from distill import student, teacher, T_logits, X_tr, y_tr, X_te, y_te, distill_loss
from wtlkit import accuracy, set_seed


def train(X, y, t, use_teacher, seed, epochs):
    set_seed(seed)
    model = student()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loader = DataLoader(TensorDataset(X, y, t), batch_size=128, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    for _ in range(epochs):
        model.train()
        for Xb, yb, tb in loader:
            out = model(Xb)
            # alpha=1: learn only from the teacher's soft predictions, no labels at all
            loss = distill_loss(out, tb, yb, alpha=1.0) if use_teacher else F.cross_entropy(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return accuracy(model, X_te, y_te)


few = torch.cat([(y_tr == c).nonzero().flatten()[:100] for c in range(10)])   # 100 labeled images per class
for seed in range(3):
    a = train(X_tr[few], y_tr[few], T_logits[few], False, seed, epochs=40)
    b = train(X_tr, y_tr, T_logits, True, seed, epochs=10)   # labels in y_tr are ignored when alpha=1
    print(f"seed {seed}: 1,000 labeled images {a:.2%}   teacher-labeled 50,000 images {b:.2%}")
