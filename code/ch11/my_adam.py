# Writing Adam as a PyTorch optimizer, then checking it against torch.optim.Adam.
import torch
from torch import nn


class MyAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if group["weight_decay"]:              # AdamW: decoupled decay
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                state = self.state[p]
                if not state:                          # first step: create buffers
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["t"] += 1
                m, v, t = state["m"], state["v"], state["t"]
                m.mul_(b1).add_(g, alpha=1 - b1)            # m = b1 m + (1-b1) g
                v.mul_(b2).addcmul_(g, g, value=1 - b2)     # v = b2 v + (1-b2) g^2
                m_hat = m / (1 - b1 ** t)                    # bias correction
                v_hat = v / (1 - b2 ** t)
                p.addcdiv_(m_hat, v_hat.sqrt() + group["eps"], value=-group["lr"])


def train(opt_class, steps=200, **kw):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1)).double()
    X, y = torch.randn(256, 10, dtype=torch.float64), torch.randn(256, 1, dtype=torch.float64)
    opt = opt_class(model.parameters(), **kw)
    for _ in range(steps):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(X), y)
        loss.backward()
        opt.step()
    return model, loss.item()


for decay, ref in [(0.0, torch.optim.Adam), (0.01, torch.optim.AdamW)]:
    m1, l1 = train(MyAdam, lr=1e-2, weight_decay=decay)
    m2, l2 = train(ref, lr=1e-2, weight_decay=decay)
    diff = max((a - b).abs().max().item() for a, b in zip(m1.parameters(), m2.parameters()))
    print(f"{ref.__name__:6s} final loss: mine {l1:.6f}  PyTorch {l2:.6f}  "
          f"largest parameter difference {diff:.1e}")
