"""Figures for Part I (Chapters 1-5)."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repository root (/home/claude/book)
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from style import CYAN, GOLD, MAG, VIOLET, NAVY, MUTED, GRID, save

sys.path.insert(0, str(ROOT / "code" / "ch03"))
sys.path.insert(0, str(ROOT / "code" / "ch04"))
sys.path.insert(0, str(ROOT / "code" / "ch05"))

two = ListedColormap(["#FBF1F6", "#B9D3E0"])          # region fills: near-white vs mid-light (distinct in grayscale)
div = LinearSegmentedColormap.from_list("div", [MAG, "#F2F2F2", CYAN])
rng = np.random.default_rng(0)


# ------------------------------------------------------------------ Ch 1
def ch1_boundary_growth():
    """A small network learning the two-spirals problem over time."""
    from sklearn.neural_network import MLPClassifier
    import warnings
    warnings.filterwarnings("ignore")
    n = 300
    t = np.sqrt(rng.uniform(0, 1, n)) * 3 * np.pi
    a = np.c_[t * np.cos(t), t * np.sin(t)] / 3 + rng.normal(0, 0.15, (n, 2))
    b = -a + rng.normal(0, 0.15, (n, 2))
    X = np.vstack([a, b]); y = np.r_[np.zeros(n), np.ones(n)]
    xx, yy = np.meshgrid(np.linspace(-3.6, 3.6, 300), np.linspace(-3.6, 3.6, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    clf = MLPClassifier((32, 32), activation="tanh", learning_rate_init=0.01,
                        max_iter=1, warm_start=True, random_state=0)
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0))
    checkpoints = [1, 30, 150, 1200]
    done = 0
    for ax, cp in zip(axes, checkpoints):
        while done < cp:
            clf.fit(X, y); done += 1
        zz = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=[0, 0.5, 1], cmap=two)
        ax.contour(xx, yy, zz, levels=[0.5], colors=NAVY, linewidths=1)
        # the two classes differ by marker and by fill luminance (dark dots vs light squares)
        ax.scatter(*X[y == 0].T, s=5, marker="o", c=NAVY, lw=0, zorder=3)
        ax.scatter(*X[y == 1].T, s=6, marker="s", facecolor="white", edgecolor=MAG, linewidths=0.5, zorder=3)
        acc = clf.score(X, y)
        ax.set_title(f"after {cp} epoch{'s' if cp > 1 else ''}\naccuracy {acc:.0%}", fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_aspect("equal")
    save(fig, "ch1_spirals")


# ------------------------------------------------------------------ Ch 2
def ch2_vectors():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    # (a) vector addition
    ax = axes[0]
    x, w = np.array([2, 1]), np.array([0.5, 1.5])
    for v, c, lab, o in [(x, CYAN, r"$\mathbf{x}$", (0, 0)), (w, GOLD, r"$\mathbf{w}$", x),
                         (x + w, MAG, r"$\mathbf{x}+\mathbf{w}$", (0, 0))]:
        ax.annotate("", xy=np.add(o, v), xytext=o,
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2, mutation_scale=12))
    ax.text(1.1, 0.25, r"$\mathbf{x}$", color=CYAN, fontsize=12)
    ax.text(2.45, 1.6, r"$\mathbf{w}$", color=GOLD, fontsize=12)
    ax.text(0.55, 1.7, r"$\mathbf{x}+\mathbf{w}$", color=MAG, fontsize=12)
    ax.set_xlim(-0.3, 3); ax.set_ylim(-0.3, 3); ax.set_aspect("equal")
    ax.set_title("(a) adding vectors: tip to tail")
    # (b) dot product as projection
    ax = axes[1]
    a, b = np.array([2.6, 0.6]), np.array([1.2, 1.8])
    proj = (a @ b) / (a @ a) * a
    ax.annotate("", xy=a, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=CYAN, lw=2))
    ax.annotate("", xy=b, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=2))
    ax.plot([b[0], proj[0]], [b[1], proj[1]], ls=":", color=MUTED, lw=1.2)
    ax.plot([0, proj[0]], [0, proj[1]], color=MAG, lw=4, alpha=0.55, solid_capstyle="round")
    ax.text(2.55, 0.25, r"$\mathbf{a}$", color=CYAN, fontsize=12)
    ax.text(1.0, 1.9, r"$\mathbf{b}$", color=GOLD, fontsize=12)
    ax.text(0.5, -0.38, "shadow of b on a", color=MAG, fontsize=7.5)
    ang = np.linspace(np.arctan2(a[1], a[0]), np.arctan2(b[1], b[0]), 30)
    ax.plot(0.5 * np.cos(ang), 0.5 * np.sin(ang), color=NAVY, lw=1)
    ax.text(0.55, 0.42, r"$\theta$", fontsize=11, color=NAVY)
    ax.set_xlim(-0.3, 3); ax.set_ylim(-0.6, 2.4); ax.set_aspect("equal")
    ax.set_title(r"(b) $\mathbf{a}\cdot\mathbf{b}=|\mathbf{a}||\mathbf{b}|\cos\theta$")
    # (c) sign of dot product: similarity
    ax = axes[2]
    ref = np.array([1.0, 0.3]); ref = ref / np.linalg.norm(ref) * 1.6
    angles = np.radians([20, 60, 100, 150, 200, 260, 320])
    ax.annotate("", xy=ref, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=2.4))
    for th in angles:
        v = 1.3 * np.array([np.cos(th), np.sin(th)])
        d = v @ ref
        ax.annotate("", xy=v, xytext=(0, 0),            # dot > 0 solid, dot < 0 dashed
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color=CYAN if d > 0 else MAG,
                                    ls="-" if d > 0 else "--"))
    ax.text(0.95, 1.25, "dot > 0 (solid)\nsimilar", color=CYAN, fontsize=7.5, ha="center")
    ax.text(-1.0, -1.6, "dot < 0 (dashed)\nopposite", color=MAG, fontsize=7.5, ha="center")
    ax.text(1.7, 0.55, r"$\mathbf{w}$", fontsize=12, color=NAVY)
    ax.set_xlim(-1.8, 2.0); ax.set_ylim(-1.8, 1.8); ax.set_aspect("equal")
    ax.set_title("(c) the sign says “same way?”")
    for ax in axes:
        ax.axhline(0, color=GRID, lw=0.8, zorder=0); ax.axvline(0, color=GRID, lw=0.8, zorder=0)
    save(fig, "ch2_vectors")


def ch2_matrix_transform():
    """A 2x2 matrix bends space: grid before/after, plus a nonlinearity."""
    M = np.array([[1.2, 0.8], [-0.4, 1.0]])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    lines = np.linspace(-2, 2, 9)
    tt = np.linspace(-2, 2, 100)

    def draw(ax, fn, title):
        for c in lines:
            for pts, col in [(np.c_[np.full_like(tt, c), tt], CYAN), (np.c_[tt, np.full_like(tt, c)], GOLD)]:
                q = fn(pts)
                ax.plot(q[:, 0], q[:, 1], color=col, lw=0.9, alpha=0.9)
        e = fn(np.array([[1.0, 0], [0, 1.0]]))
        o = fn(np.array([[0.0, 0.0]]))[0]
        for v, c in zip(e, [MAG, VIOLET]):
            ax.annotate("", xy=v, xytext=o, arrowprops=dict(arrowstyle="-|>", color=c, lw=2.2))
        ax.set_title(title); ax.set_aspect("equal"); ax.grid(False)
        ax.set_xlim(-3.4, 3.4); ax.set_ylim(-3.4, 3.4)
        ax.set_xticks([]); ax.set_yticks([])

    draw(axes[0], lambda p: p, "(a) the input space")
    draw(axes[1], lambda p: p @ M.T, r"(b) after $W\mathbf{x}$")
    draw(axes[2], lambda p: np.tanh(p @ M.T) * 2.6, r"(c) after $\tanh(W\mathbf{x})$")
    for ax in axes:
        for sp in ax.spines.values(): sp.set_visible(False)
    save(fig, "ch2_matrix_transform")


def ch2_derivative():
    f = lambda x: x**2 - 2 * x + 3
    df = lambda x: 2 * x - 2
    x = np.linspace(-1.5, 3.5, 200)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax = axes[0]
    ax.plot(x, f(x), color=NAVY)
    for x0, c in [(-1.0, MAG), (1.0, GOLD), (3.0, CYAN)]:   # same x as Listing 2.3
        xs = np.linspace(x0 - 0.9, x0 + 0.9, 10)
        ax.plot(xs, f(x0) + df(x0) * (xs - x0), color=c, lw=1.6)
        ax.plot(x0, f(x0), "o", color=c, ms=6, mec="white", mew=1.2)
        label = "slope 0" if df(x0) == 0 else f"slope {df(x0):+.0f}"
        if x0 == 1.0:                       # centred label at the bottom of the bowl
            ax.text(x0, f(x0) + 0.9, label, color=c, ha="center", fontsize=8)
        elif x0 < 1.0:                      # beside the point, clear of the tangent
            ax.text(x0 + 0.35, f(x0) + 0.6, label, color=c, ha="left", fontsize=8)
        else:
            ax.text(x0 - 0.35, f(x0) + 0.6, label, color=c, ha="right", fontsize=8)
    ax.set_ylim(0, 9); ax.set_title(r"(a) the derivative is the slope of the tangent")
    ax.set_xlabel("$x$"); ax.set_ylabel("$f(x)$")
    ax = axes[1]
    x0, h = 1.8, 1.0
    ax.plot(x, f(x), color=NAVY)
    for hh, al in [(1.0, 0.35), (0.5, 0.6), (0.15, 1.0)]:
        xs = np.array([x0 - 0.6, x0 + hh + 0.6])
        s = (f(x0 + hh) - f(x0)) / hh
        ax.plot(xs, f(x0) + s * (xs - x0), color=MAG, alpha=al, lw=1.4)
        ax.plot(x0 + hh, f(x0 + hh), "o", color=MAG, alpha=al, ms=4)
    ax.plot(x0, f(x0), "o", color=CYAN, ms=6, mec="white", mew=1.2)
    ax.annotate("as h shrinks, the secant\nbecomes the tangent", xy=(2.25, f(2.25)),
                xytext=(-1.3, 7.0), fontsize=8, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    ax.set_ylim(0, 9); ax.set_title(r"(b) $f'(x)=\lim_{h\to 0}\,[f(x+h)-f(x)]/h$")
    ax.set_xlabel("$x$")
    save(fig, "ch2_derivative")


def ch2_gradient_field():
    g = lambda x, y: x**2 + 3 * y**2
    xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-2, 2, 200))
    fig = plt.figure(figsize=(7.2, 2.8))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax3.plot_surface(xx, yy, g(xx, yy), cmap=LinearSegmentedColormap.from_list("s", ["#EAF6FA", CYAN, VIOLET]),
                     linewidth=0, antialiased=True, alpha=0.95, rstride=4, cstride=4)
    ax3.set_xlabel("$x$", labelpad=-8); ax3.set_ylabel("$y$", labelpad=-8); ax3.set_zlabel("$g$", labelpad=-8)
    ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])
    ax3.view_init(30, -60); ax3.set_title("(a) the surface $g(x,y)=x^2+3y^2$", pad=0)
    ax = fig.add_subplot(1, 2, 2)
    ax.contour(xx, yy, g(xx, yy), levels=12, colors=MUTED, linewidths=0.6)
    qx, qy = np.meshgrid(np.linspace(-2.6, 2.6, 9), np.linspace(-1.7, 1.7, 7))
    gx, gy = 2 * qx, 6 * qy
    ax.quiver(qx, qy, -gx, -gy, color=MAG, angles="xy", scale=60, width=0.006)
    ax.plot(0, 0, "*", color=GOLD, ms=12, mec=NAVY, mew=0.6)
    ax.text(0.15, 0.2, "minimum", fontsize=8, color=NAVY)
    ax.set_aspect("equal"); ax.set_title(r"(b) arrows show $-\nabla g$: the way downhill")
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    save(fig, "ch2_gradient")


# ------------------------------------------------------------------ Ch 3
def ch3_boundary_geometry():
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    w, b = np.array([1.0, 2.0]), -2.0
    xx, yy = np.meshgrid(np.linspace(-1, 4, 200), np.linspace(-1, 3, 200))
    zz = w[0] * xx + w[1] * yy + b
    ax.contourf(xx, yy, zz, levels=[-50, 0, 50], cmap=two)
    xs = np.linspace(-1, 4, 10)
    ax.plot(xs, -(w[0] * xs + b) / w[1], color=NAVY, lw=2)
    p0 = np.array([1.0, 0.5])
    ax.annotate("", xy=p0 + 0.45 * w, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=MAG, lw=2.2))
    ax.text(*(p0 + 0.5 * w + [0.05, 0]), r"$\mathbf{w}$", color=MAG, fontsize=13)
    ax.text(2.6, 2.3, "fires (output 1)\n$\\mathbf{w}\\cdot\\mathbf{x}+b>0$", color=CYAN, fontsize=8.5, ha="center")
    ax.text(0.0, -0.6, "silent (output 0)\n$\\mathbf{w}\\cdot\\mathbf{x}+b<0$", color=MAG, fontsize=8.5, ha="center")
    ax.text(3.3, -0.55, "boundary\n$\\mathbf{w}\\cdot\\mathbf{x}+b=0$", color=NAVY, fontsize=8, ha="center")
    ax.set_xlim(-1, 4); ax.set_ylim(-1, 3); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_aspect("equal"); ax.grid(False)
    save(fig, "ch3_boundary")


def ch3_learning_steps():
    from perceptron import Perceptron
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    y = np.array([0, 0, 0, 1])
    p = Perceptron(2, learning_rate=1.0)    # integer steps: every number is exact
    snaps = []
    for epoch in range(6):
        for xi, t in zip(X, y):
            err = t - p.predict(xi)
            if err:
                p.w += p.lr * err * xi; p.b += p.lr * err
                snaps.append((p.w.copy(), p.b, xi.copy()))
    pick_idx = [0, 1, len(snaps) // 2, len(snaps) - 1]
    picks = [snaps[k] for k in pick_idx]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1))
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 120), np.linspace(-0.5, 1.5, 120))
    for i, (ax, (w, b, xi)) in enumerate(zip(axes, picks)):
        zz = w[0] * xx + w[1] * yy + b
        cf = ax.contourf(xx, yy, zz, levels=[-9, 0, 9], cmap=two, hatches=[None, "."])
        for coll in getattr(cf, "collections", [cf]):
            coll.set_edgecolor("#7E98A8"); coll.set_linewidth(0)
        if np.any(w):
            ax.contour(xx, yy, zz, levels=[0], colors=NAVY, linewidths=1.5)
        for pt, t in zip(X, y):
            ax.plot(*pt, "o" if t else "s", color=CYAN if t else MAG, ms=8, mec="white", mew=1)
        ax.plot(*xi, "o", ms=15, mfc="none", mec=GOLD, mew=2)
        ax.set_title(f"update {pick_idx[i] + 1}\n$w$=({w[0]:.0f}, {w[1]:.0f})  $b$={b:.0f}", fontsize=8)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_aspect("equal"); ax.grid(False)
    save(fig, "ch3_learning")


def ch3_xor():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    for ax, (name, y) in zip(axes, [("AND", [0, 0, 0, 1]), ("OR", [0, 1, 1, 1]), ("XOR", [0, 1, 1, 0])]):
        for pt, t in zip(X, y):
            ax.plot(*pt, "o" if t else "s", color=CYAN if t else MAG, ms=11, mec="white", mew=1.2)
        xs = np.linspace(-0.4, 1.4, 10)
        if name == "AND":
            ax.plot(xs, 1.5 - xs, color=NAVY, lw=1.6)
        elif name == "OR":
            ax.plot(xs, 0.5 - xs, color=NAVY, lw=1.6)
        else:
            for sl, ic in [(-1, 0.9), (1, 0.15), (-0.3, 0.6)]:
                ax.plot(xs, ic + sl * (xs - 0.4), color=MUTED, lw=1, ls="--")
            ax.text(0.5, -0.38, "no single line works", color=MAG, ha="center", fontsize=8)
        ax.set_title(name + (" ✓" if name != "XOR" else " ✗"), color=NAVY if name != "XOR" else MAG)
        ax.set_xlim(-0.4, 1.4); ax.set_ylim(-0.5, 1.4); ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_aspect("equal"); ax.grid(False)
        ax.set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    save(fig, "ch3_xor")


# ------------------------------------------------------------------ Ch 4
def ch4_gallery():
    from activations import ACTIVATIONS
    z = np.linspace(-4, 4, 400)
    names = ["sigmoid", "tanh", "relu", "leaky_relu", "gelu"]
    titles = ["Sigmoid", "Tanh", "ReLU", "Leaky ReLU (α=0.1 shown)", "GELU"]
    fig, axes = plt.subplots(1, 5, figsize=(7.4, 1.95), sharey=True)
    for ax, n, t in zip(axes, names, titles):
        f, df = ACTIVATIONS[n]
        if n == "leaky_relu":
            fz, dz = f(z, 0.1), df(z, 0.1)
        else:
            fz, dz = f(z), df(z)
        ax.plot(z, fz, color=CYAN, label="f(z)")
        ax.plot(z, dz, color=MAG, ls="--", lw=1.5, label="f′(z)")
        ax.set_title(t, fontsize=8.5); ax.set_ylim(-1.3, 2.3)
        ax.axhline(0, color=MUTED, lw=0.6); ax.axvline(0, color=MUTED, lw=0.6)
        ax.set_xticks([-4, 0, 4])
    axes[0].text(-3.9, 1.9, "f(z)", color=CYAN, fontsize=8)
    axes[0].text(-3.9, 1.5, "f′(z)", color=MAG, fontsize=8)
    save(fig, "ch4_gallery")


def ch4_linear_collapse():
    """Stacking linear layers is still a line; ReLUs give kinks."""
    rng2 = np.random.default_rng(3)
    x = np.linspace(-3, 3, 400)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2))
    target = np.sin(1.4 * x) + 0.3 * x
    # (a) 3 linear layers
    y = x.copy()
    for _ in range(3):
        y = rng2.normal() * y + rng2.normal()
    axes[0].plot(x, target, color=MUTED, lw=1.6, ls="--")      # target: dashed
    axes[0].plot(x, y, color=MAG)                               # network: solid
    axes[0].set_title("(a) 3 linear layers = 1 line")
    # (b) sum of ReLUs (fitted by least squares)
    for ax, k, title in [(axes[1], 4, "(b) 4 ReLU units"), (axes[2], 24, "(c) 24 ReLU units")]:
        knots = np.linspace(-2.8, 2.8, k)
        F = np.c_[np.ones_like(x), x, np.maximum(0, x[:, None] - knots[None])]
        coef, *_ = np.linalg.lstsq(F, target, rcond=None)
        ax.plot(x, target, color=MUTED, lw=1.4, ls="--", label="target", zorder=3)
        ax.plot(x, F @ coef, color=CYAN, lw=2.4, label="network")
        for kk in knots:
            ax.axvline(kk, color=GOLD, lw=0.5, alpha=0.6)
        ax.set_title(title)
    axes[2].legend(loc="upper left", fontsize=7.5, handlelength=2.2)
    for ax in axes:
        ax.set_ylim(-2, 2); ax.set_xticks([-3, 0, 3])
    save(fig, "ch4_relu_pieces")


def ch4_softmax():
    from activations import softmax
    logits = np.array([2.0, 1.0, 0.1, -1.0])
    labels = ["cat", "dog", "fox", "car"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.1))
    for ax, (vals, title, col) in zip(axes, [(logits, "raw scores (logits)", MUTED),
                                             (softmax(logits), "softmax(z)", CYAN),
                                             (softmax(logits / 0.5), "softmax(z / 0.5): sharper", VIOLET)]):
        bars = ax.bar(labels, vals, color=col, width=0.62)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + (0.04 if v >= 0 else -0.25),
                    f"{v:.2f}", ha="center", fontsize=7.5, color=NAVY)
        ax.set_title(title); ax.axhline(0, color=MUTED, lw=0.6); ax.grid(axis="x", visible=False)
    axes[0].set_ylim(-1.5, 2.6); axes[1].set_ylim(0, 1.05); axes[2].set_ylim(0, 1.05)
    save(fig, "ch4_softmax")


# ------------------------------------------------------------------ Ch 5
def ch5_losses():
    p = np.linspace(0.001, 0.999, 400)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    ax = axes[0]
    e = np.linspace(-3, 3, 300)
    ax.plot(e, e**2, color=CYAN, label="squared error")
    ax.plot(e, np.abs(e), color=GOLD, ls="--", label="absolute error")
    hub = np.where(np.abs(e) < 1, 0.5 * e**2, np.abs(e) - 0.5)
    ax.plot(e, hub, color=VIOLET, ls=":", label="Huber")
    ax.text(1.75, 6.2, "squared", color=CYAN, fontsize=8)
    ax.text(-3.0, 3.35, "absolute", color=GOLD, fontsize=8)
    ax.text(-2.0, 0.85, "Huber", color=VIOLET, fontsize=8)
    ax.set_xlabel(r"error  $\hat y - y$"); ax.set_ylabel("loss"); ax.set_ylim(0, 7.5)
    ax.set_title("(a) regression losses")
    ax = axes[1]
    ax.plot(p, -np.log(p), color=CYAN)
    ax.plot(p, -np.log(1 - p), color=MAG, ls="--")
    ax.text(0.10, 5.3, r"if $y=1$: $-\log\hat p$", color=CYAN, fontsize=8)
    ax.text(0.52, 3.6, r"if $y=0$: $-\log(1-\hat p)$", color=MAG, fontsize=8)
    ax.annotate("confident and wrong:\nhuge penalty", xy=(0.03, -np.log(0.03)), xytext=(0.25, 2.6),
                fontsize=7.5, color=MUTED, arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    ax.set_xlabel(r"predicted probability $\hat p$"); ax.set_ylim(0, 6)
    ax.set_title("(b) binary cross-entropy")
    save(fig, "ch5_losses")


def ch5_learning_rates():
    L = lambda w1, w2: 0.5 * w1**2 + 4 * w2**2
    grad = lambda w: np.array([w[0], 8 * w[1]])
    xx, yy = np.meshgrid(np.linspace(-4.5, 4.5, 200), np.linspace(-4.0, 4.0, 300))
    levels = np.linspace(0, L(4.5, 2.2), 14)      # same contour lines in all three panels
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for ax, lr, title in zip(axes, [0.04, 0.2, 0.26],
                             ["too small (η=0.04)", "about right (η=0.2)", "too large (η=0.26)"]):
        ax.contour(xx, yy, L(xx, yy), levels=levels, colors=MUTED, linewidths=0.5)
        w = np.array([-4.0, 1.5]); path = [w.copy()]
        for _ in range(25):
            w = w - lr * grad(w); path.append(w.copy())
        path = np.array(path)
        col = {0.04: GOLD, 0.2: CYAN, 0.26: MAG}[lr]
        if lr == 0.26:                      # diverging: show only the first 12 steps
            path = path[:13]
        ax.plot(path[:, 0], path[:, 1], "-o", color=col, ms=3, lw=1.2, clip_on=True)
        ax.plot(0, 0, "*", color=NAVY, ms=10)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        # the diverging panel is taller (w2 grows by 1.08 per step: 1.5 * 1.08**12 = 3.8)
        ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.0, 4.0) if lr == 0.26 else ax.set_ylim(-2.2, 2.2)
    save(fig, "ch5_learning_rates")


def ch5_batch_curves():
    from logistic_neuron import make_blobs, train, sigmoid
    X, y = make_blobs()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax = axes[0]
    for (lab, bs, c, ls) in [("full batch", None, GOLD, "-"), ("mini-batch 16", 16, CYAN, "--"), ("SGD (batch 1)", 1, MAG, ":")]:
        _, _, hist = train(X, y, lr=0.1, epochs=30, batch_size=bs)
        ax.plot(range(1, 31), hist, color=c, ls=ls)
        off = {"full batch": 0.0, "mini-batch 16": 0.018, "SGD (batch 1)": -0.018}[lab]
        ax.text(30.5, hist[-1] + off, lab, color=c, fontsize=7.5, va="center")
    ax.set_xlim(1, 41); ax.set_xlabel("epoch"); ax.set_ylabel("training loss")
    ax.set_title("(a) loss per epoch")
    ax = axes[1]
    w, b, _ = train(X, y, lr=0.1, epochs=30, batch_size=16)
    xx, yy = np.meshgrid(np.linspace(-4, 4, 200), np.linspace(-4, 4, 200))
    pp = sigmoid(np.c_[xx.ravel(), yy.ravel()] @ w + b).reshape(xx.shape)
    cs = ax.contourf(xx, yy, pp, levels=np.linspace(0, 1, 11), cmap=div, alpha=0.55)
    ax.contour(xx, yy, pp, levels=[0.5], colors=NAVY, linewidths=1.5)
    ax.scatter(*X[y == 0].T, s=9, c=MAG, edgecolors="white", linewidths=0.3)
    ax.scatter(*X[y == 1].T, s=9, c=CYAN, edgecolors="white", linewidths=0.3)
    cb = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.03); cb.set_label(r"$\hat p$ (class 1)", fontsize=8)
    cb.outline.set_visible(False)
    ax.set_aspect("equal"); ax.grid(False); ax.set_title("(b) the trained neuron's probabilities")
    save(fig, "ch5_neuron_trained")


def ch5_landscape():
    """A bumpy non-convex 1-D loss: local minima, plateau, global minimum."""
    w = np.linspace(-4, 4, 500)
    L = 0.15 * w**2 + np.sin(2 * w) * 0.6 + 0.4 * np.exp(-((w - 1.6) ** 2) * 3) + 1.2
    fig, ax = plt.subplots(figsize=(5.2, 2.2))
    ax.plot(w, L, color=NAVY)
    i_glob = np.argmin(L)
    ax.plot(w[i_glob], L[i_glob], "*", color=GOLD, ms=12, mec=NAVY, mew=0.5)
    ax.annotate("global minimum", xy=(w[i_glob], L[i_glob]), xytext=(w[i_glob] - 2.2, L[i_glob] - 0.15),
                fontsize=8, color=NAVY, arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    # local minima
    from scipy.signal import argrelmin
    for i in argrelmin(L)[0]:
        if i != i_glob:
            ax.plot(w[i], L[i], "o", color=MAG, ms=6)
    ax.text(2.3, 0.62, "local minimum (B gets stuck)", color=MAG, fontsize=7.5, ha="center")
    # two balls rolling downhill with gradient descent (numerical gradient)
    Lf = lambda v: 0.15 * v**2 + np.sin(2 * v) * 0.6 + 0.4 * np.exp(-((v - 1.6) ** 2) * 3) + 1.2
    dL = lambda v: (Lf(v + 1e-5) - Lf(v - 1e-5)) / 2e-5
    for w0, col, lab, dx in [(-2.4, CYAN, "start A", -0.2), (3.7, MAG, "start B", 0.0)]:
        path = [w0]
        for _ in range(25):
            w0 -= 0.08 * dL(w0); path.append(w0)
        path = np.array(path)
        ax.plot(path, Lf(path) + 0.07, "o", color=col, ms=3.5, alpha=0.75)
        ax.text(path[0] + dx, Lf(path[0]) + 0.3, lab, color=col, fontsize=8, ha="center")
    ax.set_xlabel("a weight $w$"); ax.set_ylabel("loss $L(w)$")
    ax.set_title("Real loss landscapes have hills, valleys and traps")
    save(fig, "ch5_landscape")


if __name__ == "__main__":
    ch1_boundary_growth()
    ch2_vectors(); ch2_matrix_transform(); ch2_derivative(); ch2_gradient_field()
    ch3_boundary_geometry(); ch3_learning_steps(); ch3_xor()
    ch4_gallery(); ch4_linear_collapse(); ch4_softmax()
    ch5_losses(); ch5_learning_rates(); ch5_batch_curves(); ch5_landscape()
