"""Training loop helpers: one epoch, evaluation, and a full fit() with
optional scheduler, gradient clipping, augmentation and early stopping."""
import copy
import random
import time
import numpy as np
import torch
from torch import nn


def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, loader, optimizer, loss_fn, scheduler=None,
                    clip=None, augment=None):
    model.train()
    total, n = 0.0, 0
    for X, y in loader:
        if augment is not None:
            X = augment(X)
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        if clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if scheduler is not None:            # per-batch schedulers (e.g. OneCycle)
            scheduler.step()
        total += loss.detach() * len(y); n += len(y)   # a tensor: no GPU sync per step
    return (total / n).item()                           # one .item() per epoch


@torch.no_grad()
def evaluate(model, loader, loss_fn=nn.CrossEntropyLoss()):
    model.eval()
    total, correct, n = 0.0, 0, 0
    for X, y in loader:
        logits = model(X)
        total += loss_fn(logits, y).item() * len(y)
        correct += (logits.argmax(1) == y).sum().item(); n += len(y)
    return total / n, correct / n


def fit(model, train_loader, val_loader, optimizer, epochs=10, loss_fn=None,
        scheduler=None, clip=None, augment=None, patience=None, verbose=1,
        monitor="val_loss"):
    """Train and return a history dict with keys loss, val_loss, val_acc, lr.
    scheduler  : stepped once per *batch* (OneCycleLR, per-step LambdaLR). For
                 per-epoch schedulers step them yourself outside fit().
    clip       : max gradient norm (nn.utils.clip_grad_norm_).
    augment    : function applied to each training batch X before the forward pass.
    patience   : stop after this many epochs without improvement of the monitored
                 metric; whether or not stopping triggered, restore the best weights.
    monitor    : "val_loss" (default, lower is better) or "val_acc" (higher is better).
    verbose    : print every `verbose` epochs (0 = silent).
    Validation loss is always plain cross-entropy, even if loss_fn smooths labels."""
    if monitor not in ("val_loss", "val_acc"):
        raise ValueError(f"monitor must be 'val_loss' or 'val_acc', got {monitor!r}")
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    hist = {"loss": [], "val_loss": [], "val_acc": [], "lr": []}
    best = float("inf") if monitor == "val_loss" else -float("inf")
    best_state, waited = None, 0
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        hist["lr"].append(optimizer.param_groups[0]["lr"])
        tl = train_one_epoch(model, train_loader, optimizer, loss_fn, scheduler, clip, augment)
        vl, va = evaluate(model, val_loader)
        hist["loss"].append(tl); hist["val_loss"].append(vl); hist["val_acc"].append(va)
        if verbose and (epoch % verbose == 0 or epoch == epochs):
            print(f"epoch {epoch:3d}  loss {tl:.4f}  val_loss {vl:.4f}  "
                  f"val_acc {va:.2%}  ({time.perf_counter() - t0:.1f}s)")
        if patience is not None:
            score = vl if monitor == "val_loss" else va
            improved = score < best if monitor == "val_loss" else score > best
            if improved:
                best, waited = score, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                waited += 1
                if waited >= patience:
                    if verbose:
                        print(f"early stop at epoch {epoch}; best {monitor} {best:.4f}")
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return hist
