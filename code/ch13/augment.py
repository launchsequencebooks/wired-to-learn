# Data augmentation for digits: small random rotations, zooms and shifts,
# applied on the fly to each batch (so the network never sees the exact same image twice).
import math
import torch
import torch.nn.functional as F


def random_affine(X, max_deg=12, max_shift=0.12, scale=(0.9, 1.1)):
    """X: (N, 784) batch of flattened 28x28 images in [0, 1]."""
    N = X.shape[0]
    imgs = X.view(N, 1, 28, 28)
    ang = (torch.rand(N) * 2 - 1) * math.radians(max_deg)
    s = torch.empty(N).uniform_(*scale)
    tx, ty = [(torch.rand(N) * 2 - 1) * max_shift * 2 for _ in range(2)]  # grid units span 2
    cos, sin = torch.cos(ang) / s, torch.sin(ang) / s
    theta = torch.stack([torch.stack([cos, -sin, tx], 1),
                         torch.stack([sin, cos, ty], 1)], 1)            # (N, 2, 3)
    grid = F.affine_grid(theta, imgs.shape, align_corners=False)
    return F.grid_sample(imgs, grid, align_corners=False, padding_mode="zeros").view(N, 784)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.zeros(1, 784); x.view(28, 28)[6:22, 13:15] = 1.0        # a vertical bar "1"
    out = random_affine(x.repeat(4, 1))
    print("input ink:", x.sum().item(), "| augmented ink:", out.sum(1).round().tolist())
    print("ink center (row, col) per copy:")
    ys, xs = torch.meshgrid(torch.arange(28.), torch.arange(28.), indexing="ij")
    for img in out.view(-1, 28, 28):
        m = img.sum()
        print(f"  ({(ys * img).sum() / m:.1f}, {(xs * img).sum() / m:.1f})")
