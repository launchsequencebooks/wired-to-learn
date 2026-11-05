# Convolution from scratch in NumPy, checked against PyTorch.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ch09"))
import numpy as np
import torch
import torch.nn.functional as F
from mnist_loader import load_mnist


def conv2d(image, kernel, stride=1, padding=0):
    """2-D convolution (strictly, cross-correlation, as in every DL library)."""
    if padding:
        image = np.pad(image, padding)
    kh, kw = kernel.shape
    out_h = (image.shape[0] - kh) // stride + 1
    out_w = (image.shape[1] - kw) // stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = image[i * stride:i * stride + kh, j * stride:j * stride + kw]
            out[i, j] = np.sum(patch * kernel)          # dot product of patch and kernel
    return out


KERNELS = {
    "vertical edges":   np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),   # Sobel
    "horizontal edges": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float),
    "blur":             np.ones((3, 3)) / 9,
}

if __name__ == "__main__":
    (X, y), _, _ = load_mnist()
    digit = X[0].reshape(28, 28).astype(float)           # the first training image
    print("digit label:", y[0], "| image shape:", digit.shape)
    for name, k in KERNELS.items():
        mine = conv2d(digit, k, padding=1)
        ref = F.conv2d(torch.tensor(digit)[None, None], torch.tensor(k)[None, None], padding=1)[0, 0]
        print(f"{name:16s} output {mine.shape}  range [{mine.min():+.2f}, {mine.max():+.2f}]  "
              f"max diff vs PyTorch {np.abs(mine - ref.numpy()).max():.1e}")
    print("stride 2, no padding:", conv2d(digit, KERNELS["blur"], stride=2).shape)
