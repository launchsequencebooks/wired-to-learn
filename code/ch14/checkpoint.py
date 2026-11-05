# Checkpointing: save everything needed to resume training exactly.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from wtlkit import make_loaders, set_seed, train_one_epoch, evaluate

def setup():
    set_seed(0)
    train_loader, val_loader, _ = make_loaders(batch_size=64, n_train=10000)
    model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
    return train_loader, val_loader, model, opt, sched

loss_fn = nn.CrossEntropyLoss()

# Run A: four epochs straight through
tl, vl, model, opt, sched = setup()
for epoch in range(4):
    train_one_epoch(model, tl, opt, loss_fn); sched.step()
print(f"uninterrupted : val acc {evaluate(model, vl)[1]:.4%}")

# Run B: two epochs, save a checkpoint, 'crash', then resume from disk
tl, vl, model, opt, sched = setup()
for epoch in range(2):
    train_one_epoch(model, tl, opt, loss_fn); sched.step()
torch.save({"epoch": 2, "model": model.state_dict(), "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(), "torch_rng": torch.get_rng_state(),
            "loader_rng": tl.generator.get_state()}, "checkpoint.pt")
del model, opt, sched                                   # simulate a crash

tl, vl, model, opt, sched = setup()                     # fresh objects ...
ckpt = torch.load("checkpoint.pt")                      # ... restored from disk
model.load_state_dict(ckpt["model"]); opt.load_state_dict(ckpt["optimizer"])
sched.load_state_dict(ckpt["scheduler"]); torch.set_rng_state(ckpt["torch_rng"])
tl.generator.set_state(ckpt["loader_rng"])
for epoch in range(ckpt["epoch"], 4):
    train_one_epoch(model, tl, opt, loss_fn); sched.step()
print(f"with a restart: val acc {evaluate(model, vl)[1]:.4%}")
