# An autoencoder squeezes each digit through a 16-number bottleneck and rebuilds it.
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from wtlkit import mnist_tensors, set_seed


class AutoEncoder(nn.Module):
    def __init__(self, code=16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(),
                                     nn.Linear(64, code))
        self.decoder = nn.Sequential(nn.Linear(code, 64), nn.ReLU(), nn.Linear(64, 256), nn.ReLU(),
                                     nn.Linear(256, 784), nn.Sigmoid())   # pixels in [0, 1]

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(model, X, epochs=10, noise=0.0, lr=1e-3, verbose=True):
    """Unsupervised: the target is the input itself (or the clean input, for denoising)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(X))
        total = 0.0
        for i in range(0, len(X), 128):
            clean = X[perm[i:i + 128]]
            noisy = (clean + noise * torch.randn_like(clean)).clamp(0, 1) if noise else clean
            loss = nn.functional.mse_loss(model(noisy), clean)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(clean)
        if verbose and epoch % 5 == 0:
            print(f"epoch {epoch:2d}  reconstruction MSE {total / len(X):.4f}")
    return model


if __name__ == "__main__":
    set_seed(0)
    (X_tr, y_tr), (X_va, _), (X_te, y_te) = mnist_tensors()
    model = train_autoencoder(AutoEncoder(16), X_tr, epochs=15)
    model.eval()
    with torch.no_grad():
        test_mse = nn.functional.mse_loss(model(X_te), X_te).item()
        codes = model.encoder(X_te)
    print(f"test reconstruction MSE {test_mse:.4f}  (784 pixels -> 16 numbers: {784 // 16}x compression)")
    print("code of the first test digit:", codes[0].numpy().round(1))
    torch.save(model.state_dict(), "autoencoder.pt")
