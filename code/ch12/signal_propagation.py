# How initialization decides whether signals survive a deep network.
# A 30-layer, 256-wide network; we watch the spread (standard deviation)
# of activations going forward and of gradients coming back.
import json
import math
import torch

torch.manual_seed(0)
DEPTH, WIDTH = 30, 256
SCHEMES = {
    "small N(0, 0.01), ReLU": (lambda n: 0.01, torch.relu),
    "N(0, 1), ReLU":          (lambda n: 1.0, torch.relu),
    "Xavier, tanh":           (lambda n: math.sqrt(1 / n), torch.tanh),
    "He, ReLU":               (lambda n: math.sqrt(2 / n), torch.relu),
}
results = {}
for name, (std_fn, act) in SCHEMES.items():
    Ws = [(torch.randn(WIDTH, WIDTH) * std_fn(WIDTH)).requires_grad_() for _ in range(DEPTH)]
    h = torch.randn(512, WIDTH)                        # a batch of random inputs
    acts = []
    for W in Ws:
        h = act(h @ W)
        h.retain_grad(); acts.append(h)
    (h * torch.randn_like(h)).sum().backward()         # send random gradients back
    a_std = [a.std().item() for a in acts]
    g_std = [a.grad.std().item() for a in acts]
    results[name] = {"act": a_std, "grad": g_std}
    print(f"{name:23s} act: L1 {a_std[0]:.1e} L30 {a_std[-1]:.1e} | "
          f"grad: L30 {g_std[-1]:.1e} L1 {g_std[0]:.1e}")
json.dump(results, open("signal_propagation.json", "w"))
