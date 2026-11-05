# How many bits does ResNet-20 really need? Quantize its weights and measure.
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path += [str(root), str(root / "ch16")]
import json
import torch
from resnet import ResNet
from wtlkit import mnist_tensors, accuracy
from quantize import fake_quantize

_, _, (X_te, y_te) = mnist_tensors(dataset="fashion", image=True)
model = ResNet(n=3)
model.load_state_dict(torch.load(root / "ch16" / "fashion_resnet20.pt"))
original = {k: v.clone() for k, v in model.state_dict().items()}
n_weights = sum(p.numel() for n, p in model.named_parameters() if p.dim() > 1)
print(f"32-bit floats          test accuracy {accuracy(model, X_te, y_te):.2%}   weights {n_weights * 4 / 1024:.0f} KB")

results = {}
for per_channel in [False, True]:
    for bits in [8, 6, 4, 3, 2]:
        state = {k: (fake_quantize(v, bits, per_channel) if v.dim() > 1 else v)   # conv and linear weights
                 for k, v in original.items()}
        model.load_state_dict(state)
        acc = accuracy(model, X_te, y_te)
        results[f"{bits}-{'channel' if per_channel else 'tensor'}"] = acc
        print(f"{bits}-bit, per-{'channel' if per_channel else 'tensor '}   test accuracy {acc:.2%}   "
              f"weights {n_weights * bits / 8 / 1024:.0f} KB")
json.dump(results, open("weight_quant.json", "w"), indent=1)
