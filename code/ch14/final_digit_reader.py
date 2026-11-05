# Everything from Part III in one model: the improved digit reader.
# BatchNorm + dropout + AdamW + one-cycle schedule + augmentation +
# label smoothing + gradient clipping + early stopping + restore of the
# best-validation-loss weights.
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root)); sys.path.insert(0, str(root / "ch13"))
import torch
from torch import nn
from wtlkit import make_loaders, set_seed, fit, evaluate, count_params
from augment import random_affine


def block(n_in, n_out, p):
    return [nn.Linear(n_in, n_out), nn.BatchNorm1d(n_out), nn.ReLU(), nn.Dropout(p)]


set_seed(0)
train_loader, val_loader, test_loader = make_loaders(batch_size=128)
model = nn.Sequential(*block(784, 1024, 0.3), *block(1024, 512, 0.3), nn.Linear(512, 10))
print(f"parameters: {count_params(model):,}")

EPOCHS = 30
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.05)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, epochs=EPOCHS,
                                            steps_per_epoch=len(train_loader))
hist = fit(model, train_loader, val_loader, opt, epochs=EPOCHS,
           loss_fn=nn.CrossEntropyLoss(label_smoothing=0.05), scheduler=sched,
           clip=1.0, augment=random_affine, patience=8, verbose=5)
torch.save(model.state_dict(), "final_digit_reader.pt")
test_loss, test_acc = evaluate(model, test_loader)
print(f"test accuracy: {test_acc:.2%}  (test error {1 - test_acc:.2%})")
import json; json.dump(hist, open("final_history.json", "w"))
