"""Figures for Part V (Chapters 23-25)."""
import sys, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from style import CYAN, GOLD, MAG, VIOLET, NAVY, MUTED, GRID, save

CODE = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "code") + "/"  # was /home/claude/book/code/
sys.path.insert(0, CODE)
J = lambda p: json.load(open(CODE + p))


def ch23_results():
    import torch
    from sklearn.manifold import TSNE
    from wtlkit import FASHION_CLASSES
    r, s = J("ch23/simclr_results.json"), J("ch23/supervised_results.json")
    n = [10, 100, 1000]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    # the 10- and 100-per-class entries are lists of five accuracies (five labeled subsets):
    # plot the mean with min-max error bars (P2-CH23-002); a scalar entry is drawn as is
    def stats(vals):
        a = np.atleast_1d(np.array(vals, float)) * 100
        return a.mean(), a.mean() - a.min(), a.max() - a.mean()
    series = [("SimCLR features + linear", [r[str(k)]["simclr"] for k in n], NAVY, "-", "o"),
              ("supervised from scratch", [s[str(k)] for k in n], GOLD, "--", "s"),
              ("random features + linear", [r[str(k)]["random"] for k in n], MUTED, ":", "^")]
    for k, (lab, vals, c, ls, mk) in enumerate(series):
        m, lo, hi = zip(*(stats(v) for v in vals))
        xs = np.array(n, float) * (1 + 0.06 * (k - 1))        # slight horizontal offset so the bars do not overprint
        ax.errorbar(xs, m, yerr=[lo, hi], color=c, ls=ls, marker=mk, ms=5, lw=1.6, capsize=3, label=lab)
    ax.set_xscale("log"); ax.set_xticks(n); ax.set_xticklabels([str(k) for k in n])
    ax.set_xlabel("labeled images per class"); ax.set_ylabel("test accuracy (%)")
    ax.legend(fontsize=7, loc="lower right"); ax.set_title("(a) accuracy with few labels")
    d = torch.load(CODE + "ch23/simclr_test_features.pt")
    f = d["f"].numpy(); f = (f - f.mean(0)) / (f.std(0) + 1e-6)
    emb = TSNE(n_components=2, random_state=0, init="pca", perplexity=30).fit_transform(f)
    cmap = plt.get_cmap("tab10"); y = d["y"].numpy()
    ax = axes[1]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]      # one marker shape per class (P3-BOOK-005)
    for c in range(10):
        p = emb[y == c]
        ax.scatter(p[:, 0], p[:, 1], s=6, marker=markers[c], color=cmap(c), alpha=0.75, lw=0)
        cx, cy = np.median(p, 0)
        ax.text(cx, cy, FASHION_CLASSES[c], fontsize=6.5, ha="center", va="center", color="black",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=NAVY, lw=0.5, alpha=0.9), zorder=5)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title("(b) SimCLR features (t-SNE)")
    save(fig, "ch23_results")


def ch24_gridworld():
    r = J("ch24/gridworld.json")
    V = np.array(r["V"]); grid = r["grid"]; pol = np.array(r["policy"])
    R, C = V.shape
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    mask = np.array([[grid[i][j] in "#" for j in range(C)] for i in range(R)])
    Vm = np.ma.array(V, mask=mask)
    cm = LinearSegmentedColormap.from_list("v", ["white", "#DCE8EF", "#9FB3C8", CYAN, NAVY])   # light = low, dark = high
    cm.set_bad((0, 0, 0, 0))                                 # walls are drawn separately (hatched)
    vmin, vmax = min(-3, np.floor(V[~mask].min())), max(10, np.ceil(V[~mask].max()))
    im = ax.imshow(Vm, cmap=cm, vmin=vmin, vmax=vmax)
    from matplotlib.patches import Rectangle
    for i in range(R):
        for j in range(C):
            if mask[i, j]:                                   # wall: hatched, white face, black edge
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="white", edgecolor="black",
                                       hatch="//", linewidth=0.8, zorder=2))
    arrows = {0: (0, -0.3), 1: (0, 0.3), 2: (-0.3, 0), 3: (0.3, 0)}
    path = np.array(r["path"])
    ax.plot(path[:, 1], path[:, 0], color="black", lw=2.4, solid_capstyle="round", zorder=3, label="greedy path")
    ax.plot(path[1:-1, 1], path[1:-1, 0], "o", color="black", ms=3.2, zorder=3)   # markers on the intermediate steps
    for i in range(R):
        for j in range(C):
            ch = grid[i][j]
            if ch == "#":
                continue
            dark = V[i, j] > 0.55 * vmax
            if ch in "GX":
                ax.text(j, i, ch, ha="center", va="center", fontsize=12, fontweight="bold",
                        color="white" if dark else NAVY, zorder=6,
                        bbox=dict(boxstyle="circle,pad=0.1", fc="white" if not dark else NAVY, ec="none"))
            else:
                dx, dy = arrows[pol[i, j]]
                ax.annotate("", xy=(j + dx, i + dy), xytext=(j - dx, i - dy),
                            arrowprops=dict(arrowstyle="-|>", color="white" if dark else NAVY, lw=1.1), zorder=4)
            ax.text(j + 0.44, i + 0.44, f"{V[i, j]:.1f}", ha="right", va="bottom", fontsize=5.6,
                    color="white" if dark else NAVY, zorder=5)
            if ch == "S":
                ax.text(j - 0.44, i - 0.42, "S", ha="left", va="top", fontsize=8, fontweight="bold",
                        color="white" if dark else NAVY, zorder=5)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values(): s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, aspect=22); cb.set_label("$V(s)$", fontsize=8)
    cb.ax.tick_params(labelsize=7); cb.outline.set_visible(False)
    save(fig, "ch24_gridworld")


def ch24_cartpole():
    d, r = J("ch24/dqn_returns.json")["returns"], J("ch24/reinforce_returns.json")["returns"]
    ma = lambda x, k=20: np.convolve(x, np.ones(k) / k, mode="valid")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
    for ax, x, c, t in [(axes[0], d, CYAN, "DQN"), (axes[1], r, VIOLET, "REINFORCE")]:
        ax.plot(range(1, len(x) + 1), x, color=c, alpha=0.25, lw=0.8)
        ax.plot(range(20, len(x) + 1), ma(x), color=c, lw=2)
        ax.axhline(500, color=MUTED, ls=":", lw=1)
        ax.set_title(t); ax.set_xlabel("episode")
    axes[0].set_ylabel("return (steps balanced)"); axes[0].set_ylim(0, 520)
    save(fig, "ch24_cartpole")


def ch25_finetune():
    import re
    r = J("ch25/finetune_results.json")
    m = re.search(r"Alice val ([\d.]+) \| Shakespeare val ([\d.]+)", open(CODE + "ch25/finetune_alice.log").read())
    base_a, base_s = float(m.group(1)), float(m.group(2))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=False)
    names = list(r)
    cols = [GOLD, MAG, VIOLET]
    for ax, key, title, base in [(axes[0], "alice", "(a) new domain: Alice", base_a),
                                 (axes[1], "shakespeare", "(b) old domain: Shakespeare", base_s)]:
        vals = [r[n][key] for n in names]
        bars = ax.bar(range(len(names)), vals, color=cols, width=0.62)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=7.5, color=NAVY)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=7.5)
        ax.set_title(title); ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(vals + [base]) * 1.2)
        ax.axhline(base, color=NAVY, ls="--", lw=1)
        ax.text(-0.45, base + 0.05, f"pretrained, before adapting: {base:.2f}", fontsize=7, color=NAVY,
                bbox=dict(fc="white", ec="none", pad=1), zorder=5)
    axes[0].set_ylabel("validation loss")
    save(fig, "ch25_finetune")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        globals()[name]()
