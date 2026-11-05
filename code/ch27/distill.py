# Knowledge distillation: a small student learns from ResNet-20's soft predictions.
import sys, json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path += [str(root), str(root / "ch16")]
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from resnet import ResNet
from wtlkit import mnist_tensors, accuracy, set_seed, count_params, benchmark

(X_tr, y_tr), _, (X_te, y_te) = mnist_tensors(dataset="fashion", image=True)
teacher = ResNet(n=3)
teacher.load_state_dict(torch.load(root / "ch16" / "fashion_resnet20.pt"))
teacher.eval()
with torch.no_grad():                                      # the teacher's opinions, computed once
    T_logits = torch.cat([teacher(X_tr[i:i + 1000]) for i in range(0, len(X_tr), 1000)])


def student():
    return nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(32 * 7 * 7, 10))


def distill_loss(s_logits, t_logits, y, T=4.0, alpha=0.7):
    soft = F.kl_div(F.log_softmax(s_logits / T, 1), F.softmax(t_logits / T, 1),
                    reduction="batchmean") * T * T          # match the teacher's softened distribution
    hard = F.cross_entropy(s_logits, y)                     # and still respect the true labels
    return alpha * soft + (1 - alpha) * hard


def train_student(use_teacher, seed, epochs=10):
    set_seed(seed)
    model = student()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loader = DataLoader(TensorDataset(X_tr, y_tr, T_logits), batch_size=128, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    for _ in range(epochs):
        model.train()
        for X, y, t in loader:
            out = model(X)
            loss = distill_loss(out, t, y) if use_teacher else F.cross_entropy(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


if __name__ == "__main__":
    # three seeds each: a difference smaller than the seed-to-seed spread would mean nothing
    results = {"labels only": [], "distilled": []}
    for seed in range(3):
        plain, distilled = train_student(False, seed), train_student(True, seed)
        results["labels only"].append(accuracy(plain, X_te, y_te))
        results["distilled"].append(accuracy(distilled, X_te, y_te))
        print(f"seed {seed}: student, labels only {results['labels only'][-1]:.2%}   "
              f"student, distilled {results['distilled'][-1]:.2%}")
    for name, accs in results.items():
        t = torch.tensor(accs)
        print(f"{name:12s} mean {t.mean():.2%}  (min {t.min():.2%}, max {t.max():.2%})")

    X1 = X_te[:256]
    for name, m in [("teacher (ResNet-20)", teacher), ("student", distilled)]:
        m.eval()
        with torch.no_grad():
            ms = benchmark(lambda: m(X1))
        print(f"{name:20s} {count_params(m):7,} params   256 images in {ms:5.1f} ms")
    json.dump(results, open("distill.json", "w"), indent=1)
