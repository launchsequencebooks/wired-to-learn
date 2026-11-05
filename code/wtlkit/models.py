"""The networks that later chapters reuse, in one place.

Each class was built and explained in an earlier chapter; the applications of
Part VII import them from here instead of from that chapter's script, so that
importing a network never runs a chapter's training or data download.

    ResNet, BasicBlock            Chapter 16  (ch16/resnet.py)
    Block, CausalSelfAttention    Chapter 18  (ch18/mini_gpt.py)
    AutoEncoder, train_autoencoder  Chapter 19  (ch19/autoencoder.py)
    Encoder                       Chapter 23  (ch23/simclr.py)

Attribute names match the chapter scripts, so the state dicts those chapters
save (fashion_resnet20.pt, autoencoder.pt, simclr_encoder.pt, mini_gpt.pt)
load into these classes unchanged.
"""
import math
import torch
from torch import nn


# ---------------------------------------------------------------- Chapter 16
class BasicBlock(nn.Module):
    """Two 3x3 convolutions plus a shortcut: output = ReLU(F(x) + x)."""

    def __init__(self, c_in, c_out, stride=1, residual=True):
        super().__init__()
        self.residual = residual
        self.f = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out))
        self.shortcut = nn.Identity()
        if stride != 1 or c_in != c_out:                 # match shapes with a 1x1 conv
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        out = self.f(x)
        if self.residual:
            out = out + self.shortcut(x)                 # the skip connection
        return torch.relu(out)


class ResNet(nn.Module):
    """depth = 6n + 2 layers: a stem, three stages of n blocks, and a classifier."""

    def __init__(self, n=3, widths=(16, 32, 64), num_classes=10, residual=True):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, widths[0], 3, padding=1, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU())
        blocks, c_in = [], widths[0]
        for stage, c_out in enumerate(widths):
            for i in range(n):
                stride = 2 if (stage > 0 and i == 0) else 1   # 28 -> 14 -> 7
                blocks.append(BasicBlock(c_in, c_out, stride, residual))
                c_in = c_out
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(c_in, num_classes))

    def features(self, x):                               # everything except the classifier
        return self.blocks(self.stem(x)).mean(dim=(2, 3))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


# ---------------------------------------------------------------- Chapter 18
class CausalSelfAttention(nn.Module):
    """Multi-head attention with a causal mask: each position sees only earlier ones."""

    def __init__(self, d_model, n_heads, block_size=100):
        super().__init__()
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)         # all three projections at once
        self.proj = nn.Linear(d_model, d_model)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).bool())
        self.last_weights = None

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, heads, T, dk): each head attends independently
        q, k, v = (t.view(B, T, self.h, self.dk).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))   # no peeking ahead
        att = torch.softmax(att, dim=-1)
        self.last_weights = att.detach()
        y = (att @ v).transpose(1, 2).reshape(B, T, C)     # concatenate the heads
        return self.proj(y)


class Block(nn.Module):
    """Pre-norm Transformer block: x + Attention(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, d_model, n_heads, dropout=0.1, block_size=100):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, block_size)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))                     # tokens exchange information
        x = x + self.mlp(self.ln2(x))                      # each token thinks on its own
        return x


# ---------------------------------------------------------------- Chapter 19
class AutoEncoder(nn.Module):
    """784 pixels -> a small code -> 784 pixels; trained to rebuild its input."""

    def __init__(self, code=16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 64),
                                     nn.ReLU(), nn.Linear(64, code))
        self.decoder = nn.Sequential(nn.Linear(code, 64), nn.ReLU(), nn.Linear(64, 256),
                                     nn.ReLU(), nn.Linear(256, 784), nn.Sigmoid())

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


# ---------------------------------------------------------------- Chapter 23
class Encoder(nn.Module):
    """A small CNN that maps a 28x28 image to a 128-number feature vector (SimCLR)."""

    def __init__(self):
        super().__init__()
        def block(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(b), nn.ReLU())
        self.net = nn.Sequential(block(1, 32), nn.MaxPool2d(2), block(32, 64), nn.MaxPool2d(2),
                                 block(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)
