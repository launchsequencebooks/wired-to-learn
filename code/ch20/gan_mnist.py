# A generative adversarial network that learns to draw handwritten digits.
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from wtlkit import mnist_tensors, set_seed

Z = 64                                                       # size of the random input


def generator():
    return nn.Sequential(nn.Linear(Z, 256), nn.LeakyReLU(0.2),
                         nn.Linear(256, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
                         nn.Linear(512, 784), nn.Tanh())     # pixels in [-1, 1]


def discriminator():
    return nn.Sequential(nn.Linear(784, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
                         nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
                         nn.Linear(256, 1))                  # a logit: real or fake?


if __name__ == "__main__":
    set_seed(0)
    (X, _), _, _ = mnist_tensors()
    X = X * 2 - 1                                            # rescale to [-1, 1] like the generator
    G, D = generator(), discriminator()
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    fixed_z = torch.randn(32, Z)                             # watch the same 32 "seeds" improve
    snapshots, log = {}, []
    for epoch in range(1, 31):
        perm = torch.randperm(len(X))
        for i in range(0, len(X) - 127, 128):
            real = X[perm[i:i + 128]]
            fake = G(torch.randn(128, Z))
            # 1. discriminator: label real images 1 (smoothed to 0.9) and fakes 0
            d_loss = bce(D(real), torch.full((128, 1), 0.9)) + bce(D(fake.detach()), torch.zeros(128, 1))
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()
            # 2. generator: try to make the discriminator call the fakes real
            g_loss = bce(D(fake), torch.ones(128, 1))
            opt_g.zero_grad(); g_loss.backward(); opt_g.step()
        with torch.no_grad():
            D.eval()
            p_real = torch.sigmoid(D(X[:2000])).mean().item()
            p_fake = torch.sigmoid(D(G(torch.randn(2000, Z)))).mean().item()
            D.train()
        log.append((epoch, d_loss.item(), g_loss.item(), p_real, p_fake))
        if epoch in (1, 5, 30):
            G.eval()
            with torch.no_grad():
                snapshots[epoch] = ((G(fixed_z) + 1) / 2).tolist()
            G.train()
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:2d}  D loss {d_loss.item():.3f}  G loss {g_loss.item():.3f}  "
                  f"D(real) {p_real:.2f}  D(fake) {p_fake:.2f}")
        if epoch == 30:                  # are the samples copies of training images?
            with torch.no_grad():
                nn_dist = torch.cdist(G(fixed_z), X).min(1).values
                real_dist = torch.cdist(X[:32], X[32:]).min(1).values
            print(f"nearest training image, mean L2 distance {nn_dist.mean():.2f} "
                  f"(real-to-real: {real_dist.mean():.2f})")
    torch.save(G.state_dict(), "generator.pt")
    json.dump({"log": log, "snapshots": snapshots}, open("gan_run.json", "w"))
