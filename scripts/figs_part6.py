"""Figures for Part VI (Chapters 26-28)."""
import sys, json
import numpy as np
import matplotlib.pyplot as plt
from style import CYAN, GOLD, MAG, VIOLET, NAVY, MUTED, GRID, save

CODE = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "code") + "/"  # was /home/claude/book/code/
sys.path.insert(0, CODE)
J = lambda p: json.load(open(CODE + p))


def ch26_speedups():
    r = J("ch26/speedups.json")
    names = list(r)
    base = r[names[0]]
    rel = [r[n] / base for n in names]
    order = np.argsort(rel)
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    cols = [MUTED if names[i].startswith("baseline") else (CYAN if rel[i] > 1.02 else MAG) for i in order]
    ax.barh(range(len(names)), [rel[i] for i in order], color=cols, height=0.62)
    for k, i in enumerate(order):
        ax.text(rel[i] + 0.02, k, f"{rel[i]:.2f}x  ({r[names[i]]:.0f} img/s)", va="center", fontsize=7.5, color=NAVY)
    ax.set_yticks(range(len(names))); ax.set_yticklabels([names[i] for i in order], fontsize=8)
    ax.axvline(1.0, color=NAVY, ls="--", lw=1)
    ax.set_xlim(0, max(rel) * 1.45); ax.set_xlabel("training throughput relative to baseline")
    ax.grid(axis="y", visible=False)
    save(fig, "ch26_speedups")


def ch27_quant():
    r = J("ch27/weight_quant.json")
    bits = [8, 6, 4, 3, 2]
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.plot(bits, [r[f"{b}-channel"] * 100 for b in bits], "-o", color=CYAN, label="per-channel")
    ax.plot(bits, [r[f"{b}-tensor"] * 100 for b in bits], "-s", color=MAG, label="per-tensor")
    base = J("ch27/prune.json")["no fine-tuning"]["0.0"] * 100   # the unquantized float32 model (0% pruned)
    ax.axhline(base, color=NAVY, ls="--", lw=1)
    ax.text(7.9, base - 3.1, f"32-bit: {base:.2f}%", fontsize=7.5, color=NAVY, ha="right")
    ax.set_xticks(bits); ax.invert_xaxis()
    ax.set_xlabel("bits per weight"); ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 100); ax.legend(fontsize=8, loc="lower left")
    save(fig, "ch27_quant")


def ch27_prune():
    r = J("ch27/prune.json")
    a = r["no fine-tuning"]; f = r["fine-tuned 1 epoch"]
    xs = sorted(float(k) for k in a)
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.plot([x * 100 for x in xs], [a[str(x) if str(x) in a else repr(x)] * 100 for x in xs], "-o", color=MAG, label="pruned, no fine-tuning")
    fx = sorted(float(k) for k in f)
    ax.plot([x * 100 for x in fx], [f[str(x)] * 100 for x in fx], "-s", color=CYAN, label="pruned + 1 epoch fine-tuning")
    ax.axhline(92.63, color=NAVY, ls="--", lw=1)
    ax.set_xlabel("weights removed (%)"); ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 100); ax.legend(fontsize=8, loc="lower left")
    save(fig, "ch27_prune")


def ch27_temperature():
    import torch
    sys.path.insert(0, CODE + "ch16")
    from resnet import ResNet
    from wtlkit import mnist_tensors, FASHION_CLASSES
    _, _, (X, y) = mnist_tensors(dataset="fashion", image=True)
    m = ResNet(n=3); m.load_state_dict(torch.load(CODE + "ch16/fashion_resnet20.pt")); m.eval()
    shirts = (y == 6).nonzero().flatten()[:200]
    with torch.no_grad():
        z = m(X[shirts])
    # a correctly classified shirt whose second choice carries visible probability at T=1
    p1 = z.softmax(1)
    ok = (p1.argmax(1) == 6) & (p1[:, 6] > 0.9) & (p1[:, 6] < 0.99)
    k = int(ok.nonzero()[0])
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.3), gridspec_kw={"width_ratios": [0.55, 1, 1]})
    axes[0].imshow(X[shirts[k]].squeeze(), cmap="gray_r"); axes[0].set_axis_off(); axes[0].set_title("input: Shirt", fontsize=8)
    for ax, T, c in [(axes[1], 1, VIOLET), (axes[2], 4, CYAN)]:
        p = (z[k] / T).softmax(0).numpy()
        ax.bar(range(10), p, color=[MAG if i == 6 else c for i in range(10)])
        ax.set_xticks(range(10)); ax.set_xticklabels(FASHION_CLASSES, rotation=60, fontsize=6.5, ha="right")
        ax.set_ylim(0, 1); ax.set_title(f"teacher probabilities, T = {T}", fontsize=8)
        ax.grid(axis="x", visible=False)
    save(fig, "ch27_temperature")


def ch28_throughput():
    r = J("ch28/throughput.json")
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    # one marker and line style per engine, in the order of the Listing 28.3 rows (P2-CH28-004, P3-BOOK-005)
    style = {"PyTorch (eager)": (NAVY, "-", "o"), "PyTorch (compiled)": (CYAN, "--", "s"),
             "ONNX Runtime fp32": (GOLD, ":", "^"), "ONNX Runtime int8": ("#7A8699", "-.", "D")}
    order = [n for n in style if n in r] + [n for n in r if n not in style]
    if len(order) < 4:
        print("ch28_throughput: throughput.json holds", len(order), "engines; re-run code/ch28/ort_benchmark.py for the four-curve figure")
    for k, name in enumerate(order):
        c, ls, mk = style.get(name, (MUTED, "-", "x"))
        bs = sorted(int(b) for b in r[name])
        ys = [r[name][str(b)] for b in bs]
        ax.plot(bs, ys, color=c, ls=ls, marker=mk, ms=5, lw=1.6, label=name)
        ax.annotate(name, (bs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", fontsize=7, color=c, va="center")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks([1, 8, 64, 256]); ax.set_xticklabels(["1", "8", "64", "256"]); ax.set_xlim(0.8, 256 * 4.2)
    ax.set_xlabel("batch size (log scale)"); ax.set_ylabel("images per second (log scale)")
    ax.legend(fontsize=7.5, loc="lower right", handlelength=3)
    save(fig, "ch28_throughput")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        globals()[name]()
