"""Figures for Part III (Chapters 10-14)."""
import sys, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from style import CYAN, GOLD, MAG, VIOLET, NAVY, MUTED, GRID, save

CODE = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "code") + "/"  # was /home/claude/book/code/
J = lambda p: json.load(open(CODE + p))
ink = LinearSegmentedColormap.from_list("ink", ["white", NAVY])
SIX = [GOLD, CYAN, NAVY, VIOLET, MAG, "#2E9E6A"]
LS = ["-", "--", "-.", ":", "-", "--"]


def label_end(ax, x, y, text, color, dy=0.0):
    ax.text(x, y + dy, " " + text, color=color, fontsize=7.3, va="center")


# ------------------------------------------------------------------ Ch 11
def ch11_paths():
    sys.path.insert(0, CODE + "ch11")
    from optimizer_paths import run, SETTINGS, loss
    xx, yy = np.meshgrid(np.linspace(-10, 10, 300), np.linspace(-2.5, 2.5, 200))
    LL = 0.05 * xx**2 + 2 * yy**2
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.6))
    for ax, (name, lr), c in zip(axes.ravel(), SETTINGS.items(), SIX):
        ax.contour(xx, yy, LL, levels=np.geomspace(0.02, 25, 10), colors=MUTED, linewidths=0.45)
        p = run(name, lr, steps=50)
        ax.plot(p[:, 0], p[:, 1], "-o", color=c, ms=2.2, lw=1.1)
        ax.plot(0, 0, "*", color=NAVY, ms=9)
        ax.set_title(f"{name} (lr {lr})\nloss after 50 steps: {loss(p[-1]):.4f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_xlim(-10, 4); ax.set_ylim(-2.5, 2.5)
    fig.tight_layout(h_pad=0.6)
    save(fig, "ch11_paths")


def ch11_schedules():
    cur = J("ch11/schedules.json")
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    for (name, c), col, ls in zip(cur.items(), SIX, LS):
        ax.plot(c, color=col, ls=ls, lw=1.8)
    pos = {"constant": (1005, 0.1), "step decay": (1005, 0.0027), "cosine": (430, 0.072),
           "warmup + cosine": (60, 0.108), "one-cycle": (335, 0.1035)}
    for (name, c), col in zip(cur.items(), SIX):
        x, y = pos[name]
        ax.text(x, y, name, color=col, fontsize=7.5, va="center")
    ax.set_xlim(0, 1180); ax.set_ylim(0, 0.115)
    ax.set_xlabel("training step"); ax.set_ylabel("learning rate")
    ax.set_title("Five learning-rate schedules")
    save(fig, "ch11_schedules")


def ch11_mnist():
    h = J("ch11/optimizer_histories.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ep = np.arange(1, 9)
    cols = [GOLD, CYAN, NAVY, "#2E9E6A", VIOLET, MAG, MAG]
    lss = ["-", "-", "--", "-.", "-", ":", ":"]
    mks = ["x", "o", "s", "^", "D", "v", "P"]          # grayscale: one marker per optimizer (P3-BOOK-005)
    for (name, hist), c, ls, mk in zip(h.items(), cols, lss, mks):
        kw = dict(color=c, ls=ls, lw=1.4, marker=mk, ms=3.5, mfc="white")
        axes[0].plot(ep, hist["loss"], **kw)
        axes[1].plot(ep, np.array(hist["val_acc"]) * 100, **kw)
    axes[0].set_yscale("log"); axes[0].set_title("(a) training loss"); axes[0].set_xlabel("epoch")
    axes[1].set_title("(b) validation accuracy (%)"); axes[1].set_xlabel("epoch"); axes[1].set_ylim(93.5, 98.5)
    names = [n.split(" (")[0] for n in h]
    handles = [plt.Line2D([], [], color=c, ls=ls, lw=1.6, marker=mk, ms=4, mfc="white") for c, ls, mk in zip(cols, lss, mks)]
    fig.legend(handles, names, loc="lower center", ncol=4, fontsize=7.3, bbox_to_anchor=(0.5, -0.3))
    for ax in axes: ax.grid(True, which="major")
    save(fig, "ch11_mnist")


# ------------------------------------------------------------------ Ch 12
def ch12_signal():
    r = J("ch12/signal_propagation.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    L = np.arange(1, 31)
    for (name, d), c, ls in zip(r.items(), [GOLD, MAG, VIOLET, CYAN], ["--", ":", "-.", "-"]):
        axes[0].plot(L, d["act"], color=c, ls=ls)
        axes[1].plot(L, d["grad"], color=c, ls=ls)
    for ax, t in zip(axes, ["(a) activations, going forward", "(b) gradients, coming back"]):
        ax.set_yscale("log"); ax.set_xlabel("layer"); ax.set_title(t); ax.set_ylim(1e-30, 1e32)
    axes[0].set_ylabel("standard deviation")
    axes[1].invert_xaxis()
    handles = [plt.Line2D([], [], color=c, ls=ls) for c, ls in zip([GOLD, MAG, VIOLET, CYAN], ["--", ":", "-.", "-"])]
    fig.legend(handles, list(r), loc="lower center", ncol=4, fontsize=7.3, bbox_to_anchor=(0.5, -0.27))
    save(fig, "ch12_signal")


def ch12_bn():
    """Figure 12.2: one curve per configuration of code/ch12/bn_experiment.py. Line style by
    initialization (solid = default init, dashed = He init, dotted = BatchNorm), marker by learning
    rate (circle = 0.01, square = 0.1), so that the figure reads in grayscale (P3-BOOK-005)."""
    h = J("ch12/bn_histories.json")
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    style = {"default": (NAVY, "-", "default init"), "he": (CYAN, "--", "He init"), "BatchNorm": (GOLD, ":", "BatchNorm")}
    n_bn = sum("BatchNorm" in n and "without" not in n and "no BatchNorm" not in n for n in h)
    for name, hist in h.items():
        lr = "0.1" if "lr 0.1" in name else "0.01"
        if "BatchNorm" in name and "without" not in name and "no BatchNorm" not in name:
            kind = "BatchNorm"
        elif "he" in name.lower():
            kind = "he"
        else:
            kind = "default"
        c, ls, lab = style[kind]
        ep = np.arange(1, len(hist["val_acc"]) + 1)
        ax.plot(ep, np.array(hist["val_acc"]) * 100, color=c, ls=ls, lw=1.6, marker="o" if lr == "0.01" else "s",
                ms=4, mfc="white" if lr == "0.01" else c, mew=1.2, label=f"lr {lr}, {lab}")
    ax.set_xlim(0.8, ep[-1] + 0.2); ax.set_xticks(ep); ax.set_ylim(0, 105)
    ax.legend(loc="center right", fontsize=6.8, ncol=1, handlelength=3)
    if len(h) < 6:
        print("ch12_bn: bn_histories.json holds", len(h), "runs; re-run code/ch12/bn_experiment.py for the six-run figure")
    ax.set_xlabel("epoch"); ax.set_ylabel("validation accuracy (%)")
    ax.set_title("A 6-hidden-layer network, plain SGD")
    save(fig, "ch12_bn")


# ------------------------------------------------------------------ Ch 13
def ch13_augment():
    import torch
    sys.path.insert(0, CODE + "ch13"); sys.path.insert(0, CODE + "ch09")
    from augment import random_affine
    from mnist_loader import load_mnist
    (X, y), _, _ = load_mnist()
    torch.manual_seed(1)
    fig, axes = plt.subplots(4, 9, figsize=(6.0, 2.8))
    for r, i in enumerate([0, 7, 22, 51]):
        x = torch.from_numpy(X[i:i + 1])
        axes[r, 0].imshow(X[i].reshape(28, 28), cmap=ink); axes[r, 0].set_title("original" if r == 0 else "", fontsize=7, color=MAG)
        aug = random_affine(x.repeat(8, 1)).numpy()
        for k in range(8):
            axes[r, k + 1].imshow(aug[k].reshape(28, 28), cmap=ink)
        for ax in axes[r]:
            ax.axis("off")
    axes[0, 1].set_title("random variations →", fontsize=7, color=MUTED, loc="left")
    fig.subplots_adjust(wspace=0.05, hspace=0.1)
    save(fig, "ch13_augment")


def ch13_curves():
    h = J("ch13/regularization_histories.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ep = np.arange(1, 41)
    cols = [GOLD, MUTED, VIOLET, NAVY, CYAN, MAG]
    lss = ["-", ":", "--", "-.", "-", "-"]
    mks = ["o", "s", "^", "D", "v", "x"]          # grayscale: marker per series (P3-BOOK-005)
    for (name, hist), c, ls, mk in zip(h.items(), cols, lss, mks):
        kw = dict(color=c, ls=ls, lw=1.4, marker=mk, ms=3.5, markevery=5, mfc="white")
        axes[0].plot(ep, hist["val_loss"], **kw)
        axes[1].plot(ep, np.array(hist["val_acc"]) * 100, **kw)
    axes[0].set_title("(a) validation loss"); axes[0].set_xlabel("epoch"); axes[0].set_ylim(0.05, 0.5)
    axes[0].set_yscale("log")
    axes[1].set_title("(b) validation accuracy (%)"); axes[1].set_xlabel("epoch"); axes[1].set_ylim(92, 97.8)
    handles = [plt.Line2D([], [], color=c, ls=ls, lw=1.5, marker=mk, ms=4, mfc="white") for c, ls, mk in zip(cols, lss, mks)]
    fig.legend(handles, list(h), loc="lower center", ncol=3, fontsize=7.3, bbox_to_anchor=(0.5, -0.36))
    save(fig, "ch13_curves")


def ch13_datasize():
    d = J("ch13/data_size.json")
    n = [r["n"] for r in d]
    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    ax.plot(n, [r["train"] * 100 for r in d], "-o", color=GOLD, ms=4)
    ax.plot(n, [r["val"] * 100 for r in d], "-o", color=CYAN, ms=4)
    ax.fill_between(n, [r["val"] * 100 for r in d], [r["train"] * 100 for r in d], color=MAG, alpha=0.08)
    ax.set_xscale("log"); ax.set_xlabel("training images (log scale)"); ax.set_ylabel("accuracy (%)")
    ax.text(560, 99.0, "training", color=GOLD, fontsize=8)
    ax.text(1300, 87.0, "validation", color=CYAN, fontsize=8)
    ax.text(3000, 96.6, "generalization gap", color=MAG, fontsize=8, rotation=8)
    ax.set_title("More data shrinks the gap")
    save(fig, "ch13_datasize")


# ------------------------------------------------------------------ Ch 14
def ch14_lrfinder():
    d = J("ch14/lr_finder.json")
    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    ax.plot(d["lr"], d["loss"], color=VIOLET)
    ax.set_xscale("log"); ax.set_xlabel("learning rate (log scale)"); ax.set_ylabel("smoothed loss")
    for x, lab, c in [(d["steep"], "steepest descent", CYAN), (d["min"], "lowest loss", MAG),
                      (d["min"] / 10, "suggestion: lowest / 10", GOLD)]:
        ax.axvline(x, color=c, lw=1, ls="--")
        ax.text(x * (0.88 if c == GOLD else 1.12), 2.2 if c != GOLD else 1.0, lab, color=c, fontsize=7.3,
                rotation=90, va="top", ha="right" if c == GOLD else "left")
    ax.set_ylim(0, 2.6)
    ax.set_title("Learning-rate range test (SGD + momentum)")
    save(fig, "ch14_lrfinder")


def ch14_search():
    t = J("ch14/random_search.json")
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    # width by marker shape (and a colour of distinct luminance as a secondary cue)
    wstyle = {64: ("o", NAVY), 128: ("s", CYAN), 256: ("^", GOLD), 512: ("D", "#9FB3C8")}
    for r in t:
        mk, c = wstyle[r["width"]]
        ax.scatter(r["lr"], r["val_acc"] * 100, s=25 + 90 * r["dropout"], marker=mk, color=c,
                   edgecolor=NAVY, linewidth=0.6, zorder=3)
    best = max(t, key=lambda r: r["val_acc"])
    ax.annotate("best", (best["lr"], best["val_acc"] * 100), xytext=(26, -44), textcoords="offset points",
                fontsize=7.5, color=NAVY, arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    ax.set_xscale("log"); ax.set_xlabel("learning rate (log scale)"); ax.set_ylabel("validation accuracy (%)")
    for w, (mk, c) in wstyle.items():
        ax.scatter([], [], marker=mk, color=c, edgecolor=NAVY, linewidth=0.6, s=30, label=f"width {w}")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.set_title("Random search: 12 trials (marker size = dropout)", pad=12)
    save(fig, "ch14_search")


def ch14_monitor():
    m = J("ch14/monitoring.json")
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.2))
    s = np.arange(1, 301)
    floor = 1e-20                                    # the lr 1.0 run's gradient norm is exactly 0.0; log scale needs a floor
    for lr, c, ls in [("0.01", NAVY, "-"), ("1.0", MAG, "--")]:
        d = m[lr]
        axes[0].plot(s, d["loss"], color=c, ls=ls, lw=1.2, label=f"lr {lr}")
        axes[1].plot(s, [max(r[0], floor) for r in d["update_ratio"]], color=c, ls=ls, lw=1.2)
        axes[2].plot(s, np.array(d["dead"]) * 100, color=c, ls=ls, lw=1.2)
        axes[3].plot(s, [max(g[0], floor) for g in d["grad_norm"]], color=c, ls=ls, lw=1.2)
    axes[0].set_title("(a) loss", fontsize=8.5); axes[0].set_ylim(0, 3)
    axes[0].legend(loc="center right", fontsize=7, handlelength=2.4)
    axes[1].set_title("(b) update / weight, layer 1", fontsize=8.5); axes[1].set_yscale("log"); axes[1].set_ylim(1e-6, 1)
    axes[1].axhspan(1e-4, 1e-2, color=GRID, alpha=0.6)
    axes[1].text(10, 2e-4, "healthy zone", fontsize=7, color=MUTED)
    axes[2].set_title("(c) dead units, layer 1 (%)", fontsize=8.5); axes[2].set_ylim(-3, 103)
    axes[3].set_title("(d) gradient norm, layer 1 (log)", fontsize=8.5); axes[3].set_yscale("log")
    axes[3].set_ylim(1e-7, 10); axes[3].set_yticks([1e-6, 1e-4, 1e-2, 1])
    axes[3].text(160, 2.2e-7, "lr 1.0: exactly 0\n(below the axis)", fontsize=6.3, color=MAG, ha="center", va="bottom")
    for ax in axes:
        ax.set_xlabel("step"); ax.set_xticks([0, 100, 200, 300])
    fig.tight_layout(w_pad=0.6)
    save(fig, "ch14_monitor")


def ch14_final():
    h = J("ch14/final_history.json")
    ep = np.arange(1, len(h["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    va = np.array(h["val_acc"]) * 100
    axes[0].plot(ep, va, color=NAVY)
    best = int(np.argmin(h["val_loss"])) + 1         # epoch restored by fit(): lowest validation loss
    axes[0].plot(best, va[best - 1], "o", color=NAVY, ms=6, zorder=4)
    axes[0].annotate(f"restored (epoch {best})", (best, va[best - 1]), xytext=(0, -20), textcoords="offset points",
                     fontsize=7.3, color=NAVY, ha="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkB=3))
    ref = 97.76                                       # Chapter 10 PyTorch run, final validation accuracy (Listing 10.5)
    axes[0].axhline(ref, color=GOLD, ls="--", lw=1)
    axes[0].text(ep[-1], ref - 0.12, "Chapter 10 digit reader (val 97.76%)", fontsize=7.3, color=GOLD, ha="right", va="top")
    axes[0].set_ylim(94.5, 99.6); axes[0].set_xlabel("epoch"); axes[0].set_title("(a) validation accuracy (%)")
    print("ch14_final: lowest val loss at epoch", best, f"val acc {va[best - 1]:.2f}%")
    axes[1].plot(ep, h["lr"], color=VIOLET)
    axes[1].set_xlabel("epoch"); axes[1].set_title("(b) one-cycle learning rate (start of epoch)")
    save(fig, "ch14_final")


GROUPS = {"11": [ch11_paths, ch11_schedules, ch11_mnist],
          "12": [ch12_signal, ch12_bn],
          "13": [ch13_augment, ch13_curves, ch13_datasize],
          "14": [ch14_lrfinder, ch14_search, ch14_monitor, ch14_final]}

if __name__ == "__main__":
    # arguments: chapter numbers ("12 14") and/or function names ("ch14_final"); default: everything.
    # A figure that needs PyTorch or MNIST (ch13_augment) is skipped with a message, not a crash,
    # so the torch-free figures are always regenerated.
    args = sys.argv[1:] or list(GROUPS)
    funcs = [f for a in args for f in (GROUPS.get(a) or [globals()[a]])]
    skipped = []
    for f in funcs:
        try:
            f()
        except (ImportError, FileNotFoundError, OSError) as e:
            skipped.append(f.__name__); print(f"skipped {f.__name__}: {type(e).__name__}: {e}")
    if skipped:
        print("needs PyTorch / MNIST on the author's machine:", ", ".join(skipped))
