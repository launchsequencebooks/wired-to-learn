# SimCLR on Fashion-MNIST: learn image features from 50,000 images WITHOUT labels,
# then train a linear classifier on only a handful of labeled images.
import sys, json, math, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
import torch.nn.functional as F
from wtlkit import mnist_tensors, set_seed
from ntxent import nt_xent


def augment(x):
    """Two random views of each image: crop-and-resize, flip, shift, contrast, erase, noise."""
    N = len(x)
    s = torch.empty(N).uniform_(0.65, 1.0)                  # zoom in to 65-100% of the image
    t = (torch.rand(N, 2) * 2 - 1) * (1 - s)[:, None]       # random crop position
    theta = torch.zeros(N, 2, 3)
    theta[:, 0, 0] = s * torch.where(torch.rand(N) < 0.5, -1.0, 1.0)   # horizontal flip
    theta[:, 1, 1] = s
    theta[:, :, 2] = t
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False)
    x = x * torch.empty(N, 1, 1, 1).uniform_(0.6, 1.4)      # contrast jitter
    i, j = torch.randint(0, 20, (2, N, 1, 1, 1))     # one 8x8 patch position per image
    rows = torch.arange(28).view(1, 1, 28, 1); cols = torch.arange(28).view(1, 1, 1, 28)
    patch = (rows >= i) & (rows < i + 8) & (cols >= j) & (cols < j + 8)
    erase = patch & (torch.rand(N, 1, 1, 1) < 0.5)   # erase it in half of the images
    x = x.masked_fill(erase, 0.0)
    return (x + 0.05 * torch.randn_like(x)).clamp(0, 1)


class Encoder(nn.Module):
    """A small CNN that maps a 28x28 image to a 128-number feature vector."""

    def __init__(self):
        super().__init__()
        def block(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU())
        self.net = nn.Sequential(block(1, 32), nn.MaxPool2d(2), block(32, 64), nn.MaxPool2d(2),
                                 block(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def features(encoder, X, bs=2000):
    encoder.eval()
    return torch.cat([encoder(X[i:i + bs]) for i in range(0, len(X), bs)])


def linear_probe(f_tr, y_tr, f_te, y_te, epochs=300):
    """Logistic regression on frozen features."""
    torch.manual_seed(0)
    mean, std = f_tr.mean(0), f_tr.std(0) + 1e-6
    f_tr, f_te = (f_tr - mean) / std, (f_te - mean) / std
    clf = nn.Linear(f_tr.shape[1], 10)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(epochs):
        opt.zero_grad(); F.cross_entropy(clf(f_tr), y_tr).backward(); opt.step()
    return (clf(f_te).argmax(1) == y_te).float().mean().item()


def summary(accs):
    """Mean accuracy, with the min-max range when there is more than one run."""
    m = sum(accs) / len(accs)
    if len(accs) == 1:
        return f"{m:.2%}"
    return f"{m:.2%} ({min(accs):.1%}-{max(accs):.1%})"


def labeled_subset(y, per_class, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.cat([(y == c).nonzero().squeeze(1)[torch.randperm(int((y == c).sum()), generator=g)[:per_class]]
                      for c in range(10)])


if __name__ == "__main__":
    set_seed(0)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = mnist_tensors(dataset="fashion", image=True)
    encoder = Encoder()
    head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64))   # projection head
    params = list(encoder.parameters()) + list(head.parameters())
    EPOCHS, BS = 8, 256
    steps = EPOCHS * (len(X_tr) // BS)
    opt = torch.optim.AdamW(params, lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    random_encoder = Encoder()                                     # a frozen, untrained baseline
    t0 = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):                             # NOTE: labels are never used here
        encoder.train(); head.train()
        perm, total = torch.randperm(len(X_tr)), 0.0
        for k in range(len(X_tr) // BS):
            x = X_tr[perm[k * BS:(k + 1) * BS]]
            loss = nt_xent(head(encoder(augment(x))), head(encoder(augment(x))), temperature=0.5)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            total += loss.item()
        if epoch % 2 == 0:
            print(f"epoch {epoch:2d}  contrastive loss {total / (len(X_tr) // BS):.3f}  "
                  f"({time.perf_counter() - t0:.0f}s)")
    torch.save(encoder.state_dict(), "simclr_encoder.pt")

    f_tr_ssl, f_te_ssl = features(encoder, X_tr), features(encoder, X_te)
    f_tr_rnd, f_te_rnd = features(random_encoder, X_tr), features(random_encoder, X_te)
    results = {}
    print(f"\n{'labels per class':>17s}{'random features':>24s}"
          f"{'SimCLR features':>24s}")
    for per_class in [10, 100, 1000]:
        rs, ss = [], []
        for seed in range(5) if per_class < 1000 else [0]:   # 5 labeled subsets if few
            idx = labeled_subset(y_tr, per_class, seed=seed)
            rs.append(linear_probe(f_tr_rnd[idx], y_tr[idx], f_te_rnd, y_te))
            ss.append(linear_probe(f_tr_ssl[idx], y_tr[idx], f_te_ssl, y_te))
        results[per_class] = {"random": rs, "simclr": ss}
        print(f"{per_class:>17d}{summary(rs):>24s}{summary(ss):>24s}")
    json.dump(results, open("simclr_results.json", "w"))
    torch.save({"f": f_te_ssl[:2000], "y": y_te[:2000]}, "simclr_test_features.pt")
