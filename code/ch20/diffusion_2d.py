# A denoising diffusion model (DDPM) from scratch, on 2-D points shaped like a spiral.
import json, math
import numpy as np
import torch
from torch import nn

torch.manual_seed(0)


def spiral(n):
    t = torch.sqrt(torch.rand(n)) * 3 * math.pi
    pts = torch.stack([t * torch.cos(t), t * torch.sin(t)], 1) / 5
    return pts + 0.04 * torch.randn(n, 2)


# ---- the forward (noising) process: q(x_t | x_0) = N(sqrt(abar_t) x_0, (1 - abar_t) I)
T = 200
# the top value is chosen so that almost no signal survives at t = 199
betas = torch.linspace(1e-4, 0.1, T)                  # noise added at each step
alphas = 1 - betas
abar = torch.cumprod(alphas, 0)                         # fraction of signal left after t steps
print(f"signal left at t = {T-1}: {abar[-1].sqrt():.3f}")


def noisy(x0, t):
    eps = torch.randn_like(x0)
    return abar[t].sqrt()[:, None] * x0 + (1 - abar[t]).sqrt()[:, None] * eps, eps


class Denoiser(nn.Module):
    """Predicts the noise eps that was added, given the noisy point and the step t."""

    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 + 16, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2))

    def forward(self, x, t):
        freqs = torch.exp(torch.arange(8) * -math.log(1000) / 8)          # sinusoidal time code
        ang = t[:, None].float() * freqs[None]
        return self.net(torch.cat([x, torch.sin(ang), torch.cos(ang)], 1))


model = Denoiser()
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 4000)
for step in range(1, 4001):
    x0 = spiral(512)
    t = torch.randint(0, T, (512,))
    xt, eps = noisy(x0, t)
    loss = nn.functional.mse_loss(model(xt, t), eps)    # the entire training objective
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 1000 == 0:
        print(f"step {step}  noise-prediction loss {loss.item():.3f}")


@torch.no_grad()
def sample(n, keep=(199, 150, 100, 50, 0)):
    x = torch.randn(n, 2)                               # start from pure noise
    frames = {}
    for t in reversed(range(T)):                        # walk back step by step
        tt = torch.full((n,), t)
        eps = model(x, tt)
        x = (x - betas[t] / (1 - abar[t]).sqrt() * eps) / alphas[t].sqrt()
        if t > 0:
            x = x + betas[t].sqrt() * torch.randn_like(x)
        if t in keep:
            frames[t] = x.clone()
    return x, frames


gen, frames = sample(2000)
data = spiral(2000)
def nearest(a, b):                                       # mean distance to the closest point of b
    return torch.cdist(a, b).min(1).values.mean().item()
print(f"mean distance to the nearest real point: generated {nearest(gen, data):.3f}, "
      f"pure noise {nearest(torch.randn(2000, 2), data):.3f}, "
      f"real data {nearest(spiral(2000), data):.3f}")
print(f"coverage: mean distance from each real point to the nearest generated point "
      f"{nearest(data, gen):.3f}")
fwd = {t: noisy(data[:2000], torch.full((2000,), t))[0].tolist()
       for t in (0, 20, 60, 120, 199)}
json.dump({"forward": fwd, "reverse": {k: v.tolist() for k, v in frames.items()}},
          open("diffusion_frames.json", "w"))
