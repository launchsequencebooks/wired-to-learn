# Are the models' probabilities honest? Measure calibration, and fix it with one number.
import os, sys, json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path += [str(root), str(root / "ch31")]    # ch31: the Triage module lives there
import torch
import torch.nn.functional as F
from wtlkit.models import ResNet
from wtlkit import mnist_tensors


def ece(logits, y, bins=10):
    """Expected calibration error: the gap between confidence and accuracy, averaged over confidence bins."""
    conf, pred = logits.softmax(1).max(1)
    gap, table = 0.0, []
    for lo, hi in zip(torch.linspace(0, 1, bins + 1)[:-1], torch.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc, c = (pred[m] == y[m]).float().mean().item(), conf[m].mean().item()
            gap += m.float().mean().item() * abs(acc - c)
            table.append((c, acc, m.sum().item()))
    return gap, table


def fit_temperature(logits, y):
    """Temperature scaling: one number T divides all logits; choose it to minimize validation loss."""
    log_T = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=100)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_T.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return log_T.exp().item()


def clothing_resnet():                                    # Chapter 16
    _, (X_va, y_va), (X_te, y_te) = mnist_tensors(dataset="fashion", image=True)
    m = ResNet(n=3); m.load_state_dict(torch.load(root / "ch16" / "fashion_resnet20.pt")); m.eval()
    with torch.no_grad():
        f = lambda X: torch.cat([m(X[i:i + 1000]) for i in range(0, len(X), 1000)])
        return f(X_va), y_va, f(X_te), y_te


def ticket_ensemble():                                    # Chapter 31 (its files live in ch31/)
    os.chdir(root / "ch31")
    from triage import Triage
    from triage_data import split
    t = Triage()
    (t_va, y_va), (t_te, y_te) = split("val"), split("test")
    logits = lambda texts: torch.tensor(t.intent_probs(texts)).clamp(min=1e-9).log()   # probabilities -> logits
    return logits(t_va), y_va, logits(t_te), y_te


tables = {}
for name, get in [("clothing ResNet-20", clothing_resnet), ("ticket ensemble", ticket_ensemble)]:
    z_va, y_va, z_te, y_te = get()
    T = fit_temperature(z_va, y_va)
    (e0, t0), (e1, t1) = ece(z_te, y_te), ece(z_te / T, y_te)
    acc = (z_te.argmax(1) == y_te).float().mean()
    conf = z_te.softmax(1).max(1).values.mean()
    print(f"{name:18s} accuracy {acc:.1%}, mean confidence {conf:.1%}:  ECE {e0:.1%} -> {e1:.1%} with T = {T:.2f}")
    tables[name] = {"before": t0, "after": t1, "T": T}
json.dump(tables, open(root / "ch36" / "calibration.json", "w"))
