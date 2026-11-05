# Learning-rate schedules: how the step size changes over training.
import json
import torch

def lrs(make_sched, steps=1000, base_lr=0.1):
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=base_lr)
    sched = make_sched(opt)
    out = []
    for _ in range(steps):
        out.append(opt.param_groups[0]["lr"])
        opt.step(); sched.step()
    return out

warmup = 50
schedules = {
    "constant":       lambda o: torch.optim.lr_scheduler.LambdaLR(o, lambda s: 1.0),
    "step decay":     lambda o: torch.optim.lr_scheduler.StepLR(o, step_size=300, gamma=0.3),
    "cosine":         lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=1000),
    "warmup + cosine": lambda o: torch.optim.lr_scheduler.LambdaLR(
        o, lambda s: (s + 1) / warmup if s < warmup else
        0.5 * (1 + torch.cos(torch.tensor((s - warmup) / (1000 - warmup) * 3.14159265))).item()),
    "one-cycle":      lambda o: torch.optim.lr_scheduler.OneCycleLR(o, max_lr=0.1, total_steps=1000),
}
curves = {name: lrs(make) for name, make in schedules.items()}
for name, c in curves.items():
    print(f"{name:16s} start {c[0]:.4f}  step 100 {c[100]:.4f}  step 500 {c[500]:.4f}  end {c[-1]:.5f}")
json.dump(curves, open("schedules.json", "w"))
