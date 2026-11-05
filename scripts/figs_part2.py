"""Figures for Part II (Chapters 6-9)."""
import sys, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from style import CYAN, GOLD, MAG, VIOLET, NAVY, MUTED, GRID, save

CODE = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "code") + "/"  # was /home/claude/book/code/
for d in ["ch07", "ch09", ""]:
    sys.path.insert(0, CODE + d)
import mlp
from train_spirals import make_spirals

two = ListedColormap(["#FBF1F6", "#B9D3E0"])      # region fills: near-white vs mid-light (grayscale-safe)
three = ListedColormap(["#CDEBF4", "#F5E6C4", "#F7D6EA"])
SIX_LS = ["-", "--", ":", "-.", (0, (5, 1, 1, 1, 1, 1)), (0, (1, 1))]   # six distinct line styles


def classes(ax, X, y, s=16, labels=("class 1", "class 0")):
    """Two classes told apart by marker and fill: class 1 filled circles, class 0 open triangles."""
    ax.scatter(*X[y == 1].T, s=s, marker="o", c=NAVY, lw=0, zorder=3, label=labels[0])
    ax.scatter(*X[y == 0].T, s=s * 1.3, marker="^", facecolor="white", edgecolor=MAG, linewidths=0.7,
               zorder=3, label=labels[1])
CLS3 = [CYAN, GOLD, MAG]
ink = LinearSegmentedColormap.from_list("ink", ["white", NAVY])
div = LinearSegmentedColormap.from_list("div", [MAG, "#F4F4F4", CYAN])


def train_mlp(X, y, sizes, epochs, lr, seed=0, bs=32, snapshots=()):
    params = mlp.init_params(sizes, seed=seed)
    rng = np.random.default_rng(seed)
    snaps, losses = {}, []
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(X))
        for s in range(0, len(X), bs):
            idx = order[s:s + bs]
            P, cache = mlp.forward(X[idx], params)
            params = mlp.sgd_step(params, mlp.backward(P, y[idx], params, cache), lr)
        losses.append(mlp.cross_entropy(mlp.forward(X, params)[0], y))
        if ep in snapshots:
            snaps[ep] = [(W.copy(), b.copy()) for W, b in params]
    return params, snaps, losses


def grid(lim=1.3, n=300):
    xx, yy = np.meshgrid(np.linspace(-lim, lim, n), np.linspace(-lim, lim, n))
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_aspect("equal")


def make_ring(n=400, seed=0):
    rng = np.random.default_rng(seed)
    r_in = rng.uniform(0, 0.45, n // 2); r_out = rng.uniform(0.75, 1.1, n // 2)
    t = rng.uniform(0, 2 * np.pi, n)
    r = np.r_[r_in, r_out]
    X = np.c_[r * np.cos(t), r * np.sin(t)]
    y = np.r_[np.ones(n // 2, int), np.zeros(n // 2, int)]
    return X, y


# ------------------------------------------------------------------ Ch 6
def ch6_hidden_lines():
    X, y = make_ring()
    params, _, _ = train_mlp(X, y, [2, 6, 2], epochs=400, lr=0.1, seed=3)
    W1, b1 = params[0]
    xx, yy, G = grid()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2))
    ax = axes[0]
    classes(ax, X, y, s=9, labels=("inner class", "outer ring"))
    xs = np.linspace(-1.3, 1.3, 50)
    for j in range(W1.shape[1]):                       # one line per hidden neuron, each in its own style
        w, b = W1[:, j], b1[j]
        if abs(w[1]) > abs(w[0]):
            ax.plot(xs, -(w[0] * xs + b) / w[1], color=NAVY, lw=1.1, ls=SIX_LS[j], label=f"neuron {j + 1}")
        else:
            ax.plot(-(w[1] * xs + b) / w[0], xs, color=NAVY, lw=1.1, ls=SIX_LS[j], label=f"neuron {j + 1}")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); clean(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.01), ncol=3, fontsize=6.3, handlelength=2.6,
              columnspacing=0.9, labelspacing=0.3)
    ax.set_title("(a) each hidden neuron draws one line")
    ax = axes[1]
    P = mlp.forward(G, params)[0][:, 1].reshape(xx.shape)
    ax.contourf(xx, yy, P, levels=[0, 0.5, 1], cmap=two)
    ax.contour(xx, yy, P, levels=[0.5], colors=NAVY, linewidths=1.5)
    classes(ax, X, y, s=9)
    acc = (mlp.predict(X, params) == y).mean()
    ax.set_title(f"(b) the output combines them ({acc:.0%})"); clean(ax)
    save(fig, "ch6_hidden_lines")


def ch6_hidden_space():
    """Figure 6.3 (ch6_representation.pdf): the ring data in input space and in the 3-D hidden
    space of a trained 2-3-2 network, with the output layer's decision plane w.h + b = 0."""
    X, y = make_ring(seed=1)
    params, _, _ = train_mlp(X, y, [2, 3, 2], epochs=600, lr=0.1, seed=5)
    W1, b1 = params[0]
    W2, b2 = params[1]                                 # output layer (softmax over two classes)
    H = np.maximum(0, X @ W1 + b1)
    w = W2[:, 1] - W2[:, 0]; b = b2[1] - b2[0]          # class-1 minus class-0 logit: w.h + b = 0 is the boundary
    fig = plt.figure(figsize=(6.6, 3.0))
    ax = fig.add_subplot(1, 2, 1)
    classes(ax, X, y, s=9, labels=("inner class", "outer ring"))
    ax.legend(loc="upper left", fontsize=6.5, handletextpad=0.2, borderaxespad=0.2, frameon=True,
              framealpha=0.92, edgecolor="none")
    clean(ax); ax.set_title("(a) input space: no line separates")
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    ax3.scatter(*H[y == 1].T, s=7, marker="o", c=NAVY, depthshade=False, lw=0)
    ax3.scatter(*H[y == 0].T, s=10, marker="^", facecolor="white", edgecolor=MAG, linewidths=0.6, depthshade=False)
    # the decision plane, solved for the axis along which w is largest and clipped to the axis box
    lim = [(0, H[:, k].max() * 1.05) for k in range(3)]
    k = int(np.argmax(np.abs(w))); i, j = [m for m in range(3) if m != k]
    u, v = np.meshgrid(np.linspace(*lim[i], 30), np.linspace(*lim[j], 30))
    t = -(w[i] * u + w[j] * v + b) / w[k]
    t = np.ma.masked_outside(t, lim[k][0], lim[k][1])
    P = [None] * 3; P[i], P[j], P[k] = u, v, t
    ax3.plot_surface(P[0], P[1], P[2], color="#808080", alpha=0.28, linewidth=0, antialiased=True, zorder=1)
    ax3.set_xlim(*lim[0]); ax3.set_ylim(*lim[1]); ax3.set_zlim(*lim[2])
    ax3.set_xlabel("$h_1$", labelpad=-10); ax3.set_ylabel("$h_2$", labelpad=-10); ax3.set_zlabel("$h_3$", labelpad=-10)
    ax3.set_xticklabels([]); ax3.set_yticklabels([]); ax3.set_zticklabels([])
    ax3.view_init(22, 35)
    ax3.set_title("(b) hidden space: a plane separates", pad=0)
    acc = (mlp.predict(X, params) == y).mean()
    print("ring 2-3-2 accuracy", acc, "plane w =", w.round(2), "b =", round(float(b), 2))
    save(fig, "ch6_representation")


ch6_representation = ch6_hidden_space              # old name kept for callers


def ch6_capacity():
    from sklearn.datasets import make_moons
    X, y = make_moons(300, noise=0.25, random_state=0)
    mu, sd = X.mean(0), X.std(0)
    X = (X - mu) / sd * 0.55
    Xt, yt = make_moons(500, noise=0.25, random_state=1)   # 500 new points, same noise
    Xt = (Xt - mu) / sd * 0.55
    xx, yy, G = grid(1.4)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    accs = {}
    # the 128+128 network is trained longer and harder, until it fits the training noise
    for ax, sizes, epochs, lr, title in zip(axes, [[2, 2], [2, 8, 2], [2, 128, 128, 2]], [600, 1500, 2500], [0.1, 0.1, 0.2],
                                            ["no hidden layer\n(underfits)", "8 hidden units\n(about right)",
                                             "128 + 128 hidden units\n(wiggly, over-flexible)"]):
        params, _, _ = train_mlp(X, y, sizes, epochs=epochs, lr=lr, seed=0)
        P = mlp.forward(G, params)[0][:, 1].reshape(xx.shape)
        cf = ax.contourf(xx, yy, P, levels=[0, 0.5, 1], cmap=two, hatches=[None, "."])
        for coll in getattr(cf, "collections", [cf]):
            coll.set_edgecolor("#7E98A8"); coll.set_linewidth(0)
        ax.contour(xx, yy, P, levels=[0.5], colors=NAVY, linewidths=1.2)
        classes(ax, X, y, s=8)
        tr = (mlp.predict(X, params) == y).mean(); te = (mlp.predict(Xt, params) == yt).mean()
        accs[len(sizes)] = (tr, te)
        ax.set_title(title, fontsize=8.5); clean(ax)
        ax.set_xlabel(f"train {tr:.1%}   test {te:.1%}", fontsize=8, color=NAVY)
    assert accs[4][0] >= accs[3][0] and accs[4][1] <= accs[3][1], accs   # acceptance (P2-CH06-001)
    print("capacity train/test:", accs)
    save(fig, "ch6_capacity")


# ------------------------------------------------------------------ Ch 7
def ch7_spiral_training():
    X, y = make_spirals()
    snaps_at = (1, 10, 50, 300)
    params, snaps, losses = train_mlp(X, y, [2, 64, 64, 3], epochs=300, lr=0.05, snapshots=snaps_at)
    xx, yy, G = grid(1.15)
    fig = plt.figure(figsize=(7.2, 4.3))
    for k, ep in enumerate(snaps_at):
        ax = fig.add_subplot(2, 4, k + 1)
        pr = mlp.predict(G, snaps[ep]).reshape(xx.shape)
        ax.contourf(xx, yy, pr, levels=[-0.5, 0.5, 1.5, 2.5], cmap=three)
        for c in range(3):
            ax.scatter(*X[y == c].T, s=3, c=CLS3[c], lw=0)
        acc = (mlp.predict(X, snaps[ep]) == y).mean()
        ax.set_title(f"epoch {ep}: {acc:.0%}", fontsize=8.5); clean(ax)
    ax = fig.add_subplot(2, 1, 2)
    ax.plot(range(1, 301), losses, color=VIOLET)
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("training loss (log scale)")
    for ep in snaps_at:
        ax.plot(ep, losses[ep - 1], "o", color=NAVY, ms=4)
    ax.set_title("Cross-entropy loss falls by two orders of magnitude")
    fig.tight_layout(h_pad=1.0)
    save(fig, "ch7_spirals")


# ------------------------------------------------------------------ Ch 8
def ch8_momentum_and_regression():
    hist = np.load(CODE + "ch08/spiral_histories.npy", allow_pickle=True).item()
    import tinynet as tn
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax = axes[0]
    # spiral_histories.npy is keyed by (learning rate, momentum); see code/ch08/spirals_tinynet.py
    runs = [((0.05, 0.0), NAVY, "-", "plain SGD, lr 0.05"),
            ((0.5, 0.0), GOLD, ":", "plain SGD, lr 0.5"),
            ((0.05, 0.9), CYAN, "--", "momentum 0.9, lr 0.05")]
    for key, c, ls, lab in runs:
        h = hist[key]["val_loss"]
        ax.plot(range(1, len(h) + 1), h, color=c, ls=ls, lw=1.6, label=lab)
    ax.set_yscale("log"); ax.set_xlim(1, 60); ax.set_xlabel("epoch"); ax.set_ylabel("validation loss (log scale)")
    ax.legend(fontsize=7, loc="upper right", handlelength=2.6)
    ax.set_title("(a) spirals: momentum speeds training")
    ax = axes[1]
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, (400, 1)); y = np.sin(2 * X) + 0.3 * X + rng.normal(0, 0.1, X.shape)
    model = tn.Sequential(tn.Dense(1, 32), tn.Tanh(), tn.Dense(32, 32), tn.Tanh(), tn.Dense(32, 1))
    model.load(CODE + "ch08/curve_model.npz")
    xs = np.linspace(-3, 3, 300)[:, None]
    ax.scatter(X, y, s=5, c=MUTED, alpha=0.6, lw=0, label="training data")
    ax.plot(xs, model.forward(xs, training=False), color=NAVY, lw=2.2, label="network (solid)")
    ax.plot(xs, np.sin(2 * xs) + 0.3 * xs, color=GOLD, lw=1.4, ls="--", label="true curve (dashed)", zorder=3)
    ax.legend(fontsize=7, loc="lower right", handlelength=2.6)
    ax.set_title("(b) regression: the network's fitted curve"); ax.set_xlabel("$x$")
    save(fig, "ch8_momentum_regression")


# ------------------------------------------------------------------ Ch 9
def load():
    from mnist_loader import load_mnist
    return load_mnist()


def ch9_samples(data):
    (X, y), _, _ = data
    fig, axes = plt.subplots(5, 12, figsize=(7.2, 3.3))
    rng = np.random.default_rng(1)
    for r in range(4):
        for c in range(12):
            i = rng.integers(len(X))
            ax = axes[r, c]; ax.imshow(X[i].reshape(28, 28), cmap=ink, vmin=0, vmax=1)
            ax.set_title(str(y[i]), fontsize=7, color=MUTED, pad=1.5); ax.axis("off")
    for c in range(12):
        ax = axes[4, c]; ax.axis("off")
        if c < 10:
            ax.imshow(X[y == c].mean(0).reshape(28, 28), cmap=LinearSegmentedColormap.from_list("m", ["white", VIOLET]))
            ax.set_title(f"mean {c}", fontsize=6.5, color=VIOLET, pad=1.5)
    fig.subplots_adjust(hspace=0.45, wspace=0.1)
    save(fig, "ch9_samples")


def ch9_templates():
    """Figure 9.2 (ch9_weights.pdf): the ten softmax-regression templates, two rows of five,
    one shared diverging colour bar whose two ends differ in luminance (P2-CH09-004)."""
    import tinynet as tn
    m = tn.Sequential(tn.Dense(784, 10)).load(CODE + "ch09/softmax_regression.npz")
    W = m.layers[0].W
    v = 0.9 * np.abs(W).max()                          # symmetric limits; the extremes saturate
    fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.1))
    for d, ax in enumerate(axes.ravel()):
        im = ax.imshow(W[:, d].reshape(28, 28), cmap="gray_r", vmin=-v, vmax=v)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(MUTED); sp.set_linewidth(0.6)
        ax.set_title(str(d), fontsize=9, color=NAVY, pad=2)
    fig.subplots_adjust(left=0.02, right=0.88, top=0.93, bottom=0.03, wspace=0.08, hspace=0.18)
    cax = fig.add_axes([0.905, 0.1, 0.018, 0.78])
    cb = fig.colorbar(im, cax=cax); cb.set_label("weight", fontsize=8); cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7)
    save(fig, "ch9_weights")


ch9_weights = ch9_templates                        # old name kept for callers


def ch9_curves():
    h = np.load(CODE + "ch09/digit_mlp_history.npy", allow_pickle=True).item()
    ep = range(1, len(h["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    ax = axes[0]
    ax.plot(ep, h["loss"], color=CYAN); ax.plot(ep, h["val_loss"], color=MAG, ls="--")
    ax.text(15.3, h["loss"][-1], "training", color=CYAN, fontsize=7.5, va="center")
    ax.text(15.3, h["val_loss"][-1], "validation", color=MAG, fontsize=7.5, va="center")
    best = int(np.argmin(h["val_loss"])) + 1
    ax.axvline(best, color=MUTED, lw=0.8, ls=":")
    ax.text(best + 0.2, 0.2, f"best validation\nloss (epoch {best})", fontsize=7, color=MUTED)
    ax.set_xlim(1, 19); ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_title("(a) loss")
    ax = axes[1]
    ax.plot(ep, np.array(h["acc"]) * 100, color=CYAN); ax.plot(ep, np.array(h["val_acc"]) * 100, color=MAG, ls="--")
    ax.text(15.3, h["acc"][-1] * 100, "training", color=CYAN, fontsize=7.5, va="center")
    ax.text(15.3, h["val_acc"][-1] * 100 - 0.2, "validation", color=MAG, fontsize=7.5, va="center")
    ax.set_xlim(1, 19); ax.set_ylim(95, 100.2); ax.set_xlabel("epoch"); ax.set_ylabel("accuracy (%)")
    ax.set_title("(b) accuracy")
    save(fig, "ch9_curves")


def ch9_gallery(data):
    """Figures 9.4 and 9.5 (ch9_confusion.pdf, ch9_mistakes.pdf); each mistake is labelled with
    its predicted probability to two decimals (P2-CH09-005)."""
    return ch9_confusion_and_mistakes(data)


def ch9_confusion_and_mistakes(data):
    import tinynet as tn
    sys.path.insert(0, CODE + "ch09")
    from predict_digit import build_model
    _, _, (Xt, yt) = data
    model = build_model().load(CODE + "ch09/digit_mlp.npz")
    P = model.predict_proba(Xt); pred = P.argmax(1)
    C = np.zeros((10, 10), int)
    np.add.at(C, (yt, pred), 1)
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    off = C.copy(); np.fill_diagonal(off, 0)
    ax.imshow(np.log1p(off), cmap=LinearSegmentedColormap.from_list("c", ["white", MAG]))
    for i in range(10):
        for j in range(10):
            if i == j:
                ax.text(j, i, C[i, j], ha="center", va="center", fontsize=6.3, color=CYAN, fontweight="bold")
            elif C[i, j]:
                ax.text(j, i, C[i, j], ha="center", va="center", fontsize=6.5, color=NAVY)
    ax.set_xticks(range(10)); ax.set_yticks(range(10)); ax.grid(False)
    ax.set_xlabel("predicted digit"); ax.set_ylabel("true digit")
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title("Confusion matrix on 10,000 test images")
    save(fig, "ch9_confusion")
    wrong = np.where(pred != yt)[0]
    conf = P[wrong, pred[wrong]]
    pick = wrong[np.argsort(-conf)[:16]]
    fig, axes = plt.subplots(2, 8, figsize=(7.2, 2.2))
    for ax, i in zip(axes.ravel(), pick):
        ax.imshow(Xt[i].reshape(28, 28), cmap=ink); ax.axis("off")
        ax.set_title(f"true {yt[i]} → {pred[i]}\n{P[i, pred[i]] * 100:.2f}%", fontsize=6.5, color=NAVY, pad=2)
    fig.subplots_adjust(hspace=0.6)
    save(fig, "ch9_mistakes")
    print("worst confusions:", sorted([(off[i, j], i, j) for i in range(10) for j in range(10)], reverse=True)[:4])


def ch9_experiments():
    res = json.load(open(CODE + "ch09/experiments.json"))
    # the three-seed entry ("baseline, 3 seeds", acc = list) is not a bar: it becomes an error bar on the 256-128 row
    seeds = next((r for r in res if isinstance(r["acc"], list)), None)
    res = [r for r in res if not isinstance(r["acc"], list)]
    is_arch = lambda n: n.startswith(("1 hidden", "2 hidden"))
    names = [r["name"] if is_arch(r["name"]) else "256-128, " + r["name"] for r in res][::-1]
    accs = [r["acc"] * 100 for r in res][::-1]
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    ax.set_xlim(94, 98.5)
    for k, (n, a) in enumerate(zip(names, accs)):
        arch = is_arch(n)
        if a < 50:                                       # diverged run: no bar, a note instead
            ax.text(94.1, k, f"{a:.2f}%: training diverged", color=NAVY, fontsize=7.5, va="center")
            continue
        # architecture bars solid, training-setting bars hatched
        ax.barh(k, a, height=0.62, color=NAVY if arch else "white", edgecolor=NAVY, linewidth=0.8,
                hatch=None if arch else "///")
        ax.text(min(a, 98.5) - 0.08, k, f"{a:.2f}%", va="center", ha="right", fontsize=7.5,
                color="white" if arch else NAVY,
                bbox=None if arch else dict(fc="white", ec="none", pad=1.2))
    if seeds is not None:
        k = names.index("2 hidden layers, 256-128")
        sa = np.array(seeds["acc"]) * 100
        ax.errorbar(sa.mean(), k, xerr=sa.std(), fmt="none", ecolor=GOLD, elinewidth=1.6, capsize=3, zorder=5)
        ax.text(sa.mean(), k - 0.48, f"3 seeds: ±{sa.std():.2f}", color=GOLD, fontsize=6.5, ha="center", va="top")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("validation accuracy after 5 epochs (%)")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend([Patch(fc=NAVY, ec=NAVY), Patch(fc="white", ec=NAVY, hatch="///")],
              ["architecture", "training setting of the 256-128 network"], loc="upper center",
              bbox_to_anchor=(0.45, -0.22), ncol=2, fontsize=7)
    ax.set_title("Same data, different choices")
    save(fig, "ch9_experiments")


def ch9_own_digits():
    sys.path.insert(0, CODE + "ch09")
    from predict_digit import preprocess, build_model
    from PIL import Image
    model = build_model().load(CODE + "ch09/digit_mlp.npz")
    files = ["my_seven", "my_two", "my_zero", "my_four"]
    fig, axes = plt.subplots(2, 4, figsize=(5.6, 2.9))
    for k, f in enumerate(files):
        path = CODE + f"ch09/{f}.png"
        axes[0, k].imshow(Image.open(path), cmap="gray", vmin=0, vmax=255); axes[0, k].axis("off")
        x = preprocess(path)
        p = model.predict_proba(x)[0]
        axes[1, k].imshow(x.reshape(28, 28), cmap=ink); axes[1, k].axis("off")
        axes[1, k].set_title(f"→ {p.argmax()}  ({p.max():.0%})", fontsize=8, color=MAG, pad=2)
    axes[0, 0].set_title("original photo", fontsize=8, color=MUTED, loc="left")
    axes[1, 0].text(-2, -9, "after preprocessing (28×28)", fontsize=8, color=MUTED)
    fig.subplots_adjust(hspace=0.45)
    save(fig, "ch9_own_digits")


def _ch9_mnist_figures():
    """The Chapter 9 figures that need MNIST and the trained .npz files (author's machine)."""
    data = load()
    ch9_samples(data); ch9_templates(); ch9_curves(); ch9_gallery(data); ch9_own_digits()


GROUPS = {"6": [ch6_hidden_lines, ch6_hidden_space, ch6_capacity],
          "7": [ch7_spiral_training],
          "8": [ch8_momentum_and_regression],
          "9": [ch9_experiments, _ch9_mnist_figures]}

if __name__ == "__main__":
    # arguments: chapter numbers ("6 8") and/or function names ("ch9_experiments"); default: everything.
    # Figures that need MNIST or a trained model file are skipped with a message, not a crash.
    args = sys.argv[1:] or list(GROUPS)
    funcs = [f for a in args for f in (GROUPS.get(a) or [globals()[a]])]
    skipped = []
    for f in funcs:
        try:
            f()
        except (ImportError, FileNotFoundError, OSError, RuntimeError) as e:
            skipped.append(f.__name__); print(f"skipped {f.__name__}: {type(e).__name__}: {str(e)[:120]}")
    if skipped:
        print("needs MNIST / trained models on the author's machine:", ", ".join(skipped))
