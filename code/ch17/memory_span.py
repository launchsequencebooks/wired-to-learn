# How far back can the gradient reach? The size of d(output at the last step)/d(input at step t)
# for a vanilla RNN and an LSTM at initialization.
import json
import torch
from torch import nn

T, n = 100, 64
results = {}
for name, cls in [("RNN", nn.RNN), ("LSTM", nn.LSTM), ("GRU", nn.GRU)]:
    torch.manual_seed(0)
    net = cls(n, n, batch_first=True)
    with torch.no_grad():                    # common practice: forget-gate bias starts at 1
        if name == "LSTM":
            net.bias_ih_l0[n:2 * n] = 1.0
    x = torch.randn(32, T, n, requires_grad=True)
    out, _ = net(x)
    out[:, -1].pow(2).sum().backward()
    g = x.grad.norm(dim=2).mean(0)                           # gradient size per time step
    g = (g / g[-1]).tolist()
    results[name] = g
    print(f"{name:5s} relative gradient from step 99 back to: step 90 {g[90]:.1e}, "
          f"step 50 {g[50]:.1e}, step 0 {g[0]:.1e}")
json.dump(results, open("memory_span.json", "w"))
