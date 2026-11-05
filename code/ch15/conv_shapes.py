# Output sizes and parameter counts of convolutional layers.
import torch
from torch import nn

x = torch.randn(1, 1, 28, 28)            # (batch, channels, height, width)
layers = {
    "Conv2d(1, 32, 3)":             nn.Conv2d(1, 32, 3),
    "Conv2d(1, 32, 3, padding=1)":  nn.Conv2d(1, 32, 3, padding=1),
    "Conv2d(1, 32, 5, stride=2)":   nn.Conv2d(1, 32, 5, stride=2),
    "MaxPool2d(2)":                 nn.MaxPool2d(2),
}
for name, layer in layers.items():
    n = sum(p.numel() for p in layer.parameters())
    print(f"{name:30s} -> output {tuple(layer(x).shape)}  parameters {n}")

# Weight sharing: a conv layer vs a dense layer producing the same-sized output
conv = nn.Conv2d(1, 32, 3, padding=1)                     # 28x28 -> 32 x 28x28
dense = nn.Linear(28 * 28, 32 * 28 * 28)                  # same output size, all-to-all
count = lambda m: sum(p.numel() for p in m.parameters())
print(f"conv layer : {count(conv):>12,} parameters")
print(f"dense layer: {count(dense):>12,} parameters ({count(dense) / count(conv):,.0f}x more)")
