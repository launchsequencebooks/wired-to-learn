# Translation equivariance: shift the input, and a conv layer's output shifts too.
import torch
from torch import nn

torch.manual_seed(0)
conv = nn.Conv2d(1, 4, 3, padding=1, bias=False)
x = torch.zeros(1, 1, 28, 28)
x[0, 0, 8:14, 6:10] = 1.0                          # a small blob
x_shift = torch.roll(x, shifts=(5, 7), dims=(2, 3))  # move it 5 down, 7 right

with torch.no_grad():
    out, out_shift = conv(x), conv(x_shift)
diff = (torch.roll(out, shifts=(5, 7), dims=(2, 3)) - out_shift).abs().max().item()
print(f"conv(shift(x)) vs shift(conv(x)): max difference {diff:.1e}")

pool = nn.MaxPool2d(2)                             # pooling adds a little invariance
same = (pool(x) == pool(torch.roll(x, 1, dims=3))).float().mean().item()
print(f"after a 1-pixel shift, {same:.0%} of max-pooled outputs are unchanged")
