# A variational autoencoder with a 2-D latent space: a map of all handwritten digits.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from wtlkit import mnist_tensors, set_seed


class VAE(nn.Module):
    def __init__(self, latent=2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.mu = nn.Linear(256, latent)                  # centre of the code's distribution
        self.logvar = nn.Linear(256, latent)              # log of its spread
        self.dec = nn.Sequential(nn.Linear(latent, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(),
                                 nn.Linear(512, 784))     # logits of the pixels

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)   # reparameterization trick
        return self.dec(z), mu, logvar


def vae_loss(logits, x, mu, logvar):
    recon = nn.functional.binary_cross_entropy_with_logits(logits, x, reduction="sum") / len(x)
    kl = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp()) / len(x)   # distance to N(0, I)
    return recon + kl, recon, kl


if __name__ == "__main__":
    set_seed(0)
    (X_tr, _), _, (X_te, y_te) = mnist_tensors()
    model = VAE()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, 21):
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), 128):
            x = X_tr[perm[i:i + 128]]
            logits, mu, logvar = model(x)
            loss, recon, kl = vae_loss(logits, x, mu, logvar)
            opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                logits, mu, logvar = model(X_te)
                l, r, k = vae_loss(logits, X_te, mu, logvar)
            model.train()
            print(f"epoch {epoch:2d}  test loss {l:.1f} = reconstruction {r:.1f} + KL {k:.2f}  (per image)")
    model.eval()
    with torch.no_grad():
        mu = model(X_te)[1]
        samples = torch.sigmoid(model.dec(torch.randn(8, 2)))   # brand-new digits from noise
    for i in range(3):
        print(f"test digit {y_te[i].item()}: latent mean ({mu[i, 0]:+.2f}, {mu[i, 1]:+.2f})")
    print("generated", tuple(samples.shape), "new images from random latent points")
    torch.save(model.state_dict(), "vae.pt")
