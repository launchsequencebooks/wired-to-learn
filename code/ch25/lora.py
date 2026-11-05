# LoRA (low-rank adaptation) from scratch: freeze a big weight matrix W and learn a small
# low-rank update B @ A next to it.
import math
import torch
from torch import nn


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear: y = x W^T + b + (alpha / r) * x A^T B^T."""

    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False                                   # the pretrained weights stay fixed
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))      # zero: start exactly at the base model
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T

    @torch.no_grad()
    def merge(self):
        """Fold the update into the base weights: no extra cost at inference time."""
        self.base.weight += self.scale * self.B @ self.A
        return self.base


def add_lora(module, r=8, alpha=16, targets=("qkv", "proj"), _top=True):
    """Replace every nn.Linear whose attribute name is in `targets` with a LoRA version."""
    n = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name in targets:
            setattr(module, name, LoRALinear(child, r, alpha)); n += 1
        else:
            n += add_lora(child, r, alpha, targets, _top=False)
    if _top and n == 0:                               # fail loudly, not silently
        raise ValueError(f"no nn.Linear named {targets} found; "
                         "check the model's layer names (print the model)")
    return n if not _top else module


if __name__ == "__main__":
    torch.manual_seed(0)
    base = nn.Linear(512, 512)
    layer = LoRALinear(base, r=8)
    x = torch.randn(4, 512)
    print("starts identical to the base layer:", torch.allclose(layer(x), base(x)))
    full = sum(p.numel() for p in base.parameters())
    lora = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    print(f"trainable parameters: full fine-tuning {full:,} | LoRA r=8 {lora:,} ({lora / full:.1%})")
    with torch.no_grad():
        layer.B.normal_()                                              # pretend we trained it
        before = layer(x)
    merged = layer.merge()
    print("merged layer gives the same output:", torch.allclose(merged(x), before, atol=1e-4))
