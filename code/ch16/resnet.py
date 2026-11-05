# A small ResNet for 28x28 grayscale images (the CIFAR-style ResNet of He et al., 2016).
import torch
from torch import nn


class BasicBlock(nn.Module):
    """Two 3x3 convolutions plus a shortcut: output = ReLU(F(x) + x)."""

    def __init__(self, c_in, c_out, stride=1, residual=True):
        super().__init__()
        self.residual = residual
        self.f = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out))
        self.shortcut = nn.Identity()
        if stride != 1 or c_in != c_out:                 # match shapes with a 1x1 conv
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                                          nn.BatchNorm2d(c_out))

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
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c_in, num_classes))

    def features(self, x):                               # everything except the classifier
        return self.blocks(self.stem(x)).mean(dim=(2, 3))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


if __name__ == "__main__":
    for n in [1, 3]:
        for residual in [False, True]:
            m = ResNet(n, residual=residual)
            name = f"{'ResNet' if residual else 'Plain'}-{6 * n + 2}"
            print(f"{name:10s} parameters {sum(p.numel() for p in m.parameters()):>8,}  "
                  f"output {tuple(m(torch.randn(2, 1, 28, 28)).shape)}")
    x = torch.randn(4, 16, 28, 28)
    print("block keeps shape:", tuple(BasicBlock(16, 16)(x).shape),
          "| downsampling block:", tuple(BasicBlock(16, 32, stride=2)(x).shape))
