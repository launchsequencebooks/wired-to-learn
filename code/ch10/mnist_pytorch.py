# Chapter 9's digit reader, rebuilt in PyTorch with the standard training loop.
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ch09"))
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from mnist_loader import load_mnist
from modules import DigitMLP

torch.manual_seed(0)                                  # reproducible runs
(X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_mnist()
to_ds = lambda X, y: TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
train_loader = DataLoader(to_ds(X_tr, y_tr), batch_size=64, shuffle=True)
val_loader = DataLoader(to_ds(X_va, y_va), batch_size=1024)
test_loader = DataLoader(to_ds(X_te, y_te), batch_size=1024)

model = DigitMLP()
loss_fn = nn.CrossEntropyLoss()                       # softmax + CE, takes logits
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)


def evaluate(loader):
    model.eval()                                      # evaluation mode
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():                             # no graph, less memory
        for X, y in loader:
            logits = model(X)
            loss_sum += loss_fn(logits, y).item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
    return loss_sum / total, correct / total


for epoch in range(1, 16):
    model.train()                                     # training mode
    t0 = time.perf_counter()
    for X, y in train_loader:
        optimizer.zero_grad()                         # 1. clear old gradients
        loss = loss_fn(model(X), y)                   # 2. forward + loss
        loss.backward()                               # 3. backward (autograd)
        optimizer.step()                              # 4. update
    if epoch % 3 == 0:
        vl, va = evaluate(val_loader)
        print(f"epoch {epoch:2d}  val_loss {vl:.4f}  val_acc {va:.2%}  "
              f"({time.perf_counter() - t0:.1f}s)")

print(f"test accuracy: {evaluate(test_loader)[1]:.2%}")
torch.save(model.state_dict(), "digit_mlp_torch.pt")  # weights only
