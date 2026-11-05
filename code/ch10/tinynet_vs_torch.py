# Proof that TinyNet and PyTorch compute the same thing: load Chapter 9's
# trained TinyNet weights into a PyTorch model and compare outputs and gradients.
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root)); sys.path.insert(0, str(root / "ch09"))
import numpy as np
import torch
from torch import nn
import tinynet as tn
from mnist_loader import load_mnist
from predict_digit import build_model

weights = root / "ch09" / "digit_mlp.npz"
if not weights.exists():
    sys.exit(f"{weights} not found: "
             "run code/ch09/mlp_digits.py first (about 3 minutes).")
tiny = build_model().load(weights)
torch_model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128),
                            nn.ReLU(), nn.Linear(128, 10)).double()
dense = [l for l in tiny.layers if isinstance(l, tn.Dense)]
linear = [m for m in torch_model if isinstance(m, nn.Linear)]
with torch.no_grad():
    for d, lin in zip(dense, linear):
        lin.weight.copy_(torch.from_numpy(d.W.T))    # PyTorch stores (out, in)
        lin.bias.copy_(torch.from_numpy(d.b))

_, _, (X_te, y_te) = load_mnist()
X = X_te[:256].astype(np.float64); y = y_te[:256]
diff = np.abs(tiny.forward(X, training=False) - torch_model(torch.from_numpy(X)).detach().numpy())
print(f"largest difference in logits: {diff.max():.1e}")

# gradients: TinyNet's hand-written backward vs. PyTorch autograd
loss_tn = tn.SoftmaxCrossEntropy()
loss_tn.forward(tiny.forward(X), y); tiny.backward(loss_tn.backward())
loss_pt = nn.functional.cross_entropy(torch_model(torch.from_numpy(X)), torch.from_numpy(y))
loss_pt.backward()
for k, (d, lin) in enumerate(zip(dense, linear), 1):
    g = np.abs(d.dW - lin.weight.grad.numpy().T).max()
    print(f"layer {k}: largest weight-gradient difference {g:.1e}")
