"""Figures for Part IV (Chapters 15-22)."""
import sys, json, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from style import CYAN, GOLD, MAG, VIOLET, NAVY, MUTED, GRID, DIVERGING, save

CODE = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "code") + "/"  # was /home/claude/book/code/
for d in ["", "ch09", "ch15", "ch16", "ch17", "ch18", "ch19", "ch20", "ch21"]:
    sys.path.insert(0, CODE + d)
J = lambda p: json.load(open(CODE + p))
ink = LinearSegmentedColormap.from_list("ink", ["white", NAVY])
div = LinearSegmentedColormap.from_list("div", [MAG, "#F4F4F4", CYAN])
GREEN = "#2E9E6A"


def off(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


# ------------------------------------------------------------------ Ch 15
def ch15_filters():
    from conv_scratch import conv2d, KERNELS
    from mnist_loader import load_mnist
    (X, y), _, _ = load_mnist()
    digit = X[0].reshape(28, 28).astype(float)
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0))
    fig.subplots_adjust(left=0.01, right=0.9, top=0.86, bottom=0.03, wspace=0.08)
    axes[0].imshow(digit, cmap=ink); axes[0].set_title(f"input (a {y[0]})", fontsize=8.5)
    outs = {name: conv2d(digit, k, padding=1) for name, k in KERNELS.items()}
    v = max(np.abs(o).max() for n, o in outs.items() if n != "blur")   # one symmetric scale for the signed maps
    for ax, (name, out) in zip(axes[1:], outs.items()):
        if name == "blur":
            ax.imshow(out, cmap=ink, vmin=0, vmax=out.max())
        else:                                     # diverging map whose ends differ in luminance (P3-BOOK-005)
            im = ax.imshow(out, cmap=DIVERGING, vmin=-v, vmax=v)
        ax.set_title(name, fontsize=8.5)
    for ax in axes: off(ax)
    cax = fig.add_axes([0.915, 0.1, 0.014, 0.7])        # shared scale of the two signed edge maps
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("edge response\n(dark +, light \u2212)", fontsize=6.5); cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_visible(False)
    save(fig, "ch15_filters")


def ch15_learned():
    import torch
    from cnn_mnist import DigitCNN
    from mnist_loader import load_mnist
    m = DigitCNN(); m.load_state_dict(torch.load(CODE + "ch15/digit_cnn.pt")); m.eval()
    W = m.features[0][0].weight.detach()[:, 0]            # 32 x 3 x 3
    _, _, (Xt, yt) = load_mnist()
    x = torch.tensor(Xt[1]).view(1, 1, 28, 28)
    with torch.no_grad():
        fm = m.features[0][1](m.features[0][0](x))[0]      # conv + BatchNorm, BEFORE the ReLU (P2-CH15-001)
    fm = fm.numpy()
    fig = plt.figure(figsize=(7.2, 3.3))
    top = fig.add_gridspec(2, 16, left=0.02, right=0.98, top=0.9, bottom=0.6, wspace=0.12, hspace=0.12)
    bot = fig.add_gridspec(2, 8, left=0.02, right=0.98, top=0.5, bottom=0.0, wspace=0.08, hspace=0.08)
    v = float(W.abs().max())                               # one symmetric scale for all 32 kernels
    for k in range(32):
        ax = fig.add_subplot(top[k // 16, k % 16])
        ax.imshow(W[k], cmap=DIVERGING, vmin=-v, vmax=v); off(ax)
    for k in range(16):                                    # each map normalized to its own [min, max]
        ax = fig.add_subplot(bot[k // 8, k % 8])
        f = fm[k]; lo, hi = f.min(), f.max()
        ax.imshow((f - lo) / (hi - lo + 1e-8), cmap=DIVERGING, vmin=0, vmax=1); off(ax)
        for sp in ax.spines.values():                      # thin black frame
            sp.set_visible(True); sp.set_edgecolor("black"); sp.set_linewidth(0.5)
    differ = sum(np.corrcoef(fm[k].ravel(), fm[0].ravel())[0, 1] < 0.8 for k in range(1, 16))
    print(f"ch15_learned: {differ} of 15 maps differ from map 0 (corr < 0.8); acceptance needs at least 8")
    fig.text(0.5, 0.94, "(a) the 32 learned first-layer kernels", ha="center", fontsize=9, color=NAVY, fontweight="semibold")
    fig.text(0.5, 0.535, f"(b) 16 of their feature maps for a test digit '{yt[1]}'", ha="center", fontsize=9, color=NAVY, fontweight="semibold")
    save(fig, "ch15_learned")


# ------------------------------------------------------------------ Ch 16
def ch16_depth():
    h = J("ch16/depth_histories.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ep = np.arange(1, 5)
    # depth by line style (8 dashed, 20 solid), family by marker (plain: open circle, ResNet: filled square)
    style = {"Plain-8": (GOLD, "--", "o"), "ResNet-8": (CYAN, "--", "s"), "Plain-20": (MAG, "-", "o"), "ResNet-20": (NAVY, "-", "s")}
    for name, hist in h.items():
        c, ls, mk = style[name]
        kw = dict(color=c, ls=ls, marker=mk, ms=4, mfc="white" if mk == "o" else c)
        axes[0].plot(ep, hist["loss"], label=name, **kw)
        axes[1].plot(ep, np.array(hist["val_acc"]) * 100, **kw)
    axes[0].set_title("(a) training loss"); axes[0].set_xlabel("epoch"); axes[0].legend(fontsize=7.3)
    axes[1].set_title("(b) validation accuracy (%)"); axes[1].set_xlabel("epoch")
    for ax in axes: ax.set_xticks(ep)
    save(fig, "ch16_depth")


def ch16_fashion():
    import torch
    from resnet import ResNet
    from wtlkit import mnist_tensors, FASHION_CLASSES
    _, _, (Xt, yt) = mnist_tensors(dataset="fashion", image=True)
    m = ResNet(3); m.load_state_dict(torch.load(CODE + "ch16/fashion_resnet20.pt")); m.eval()
    with torch.no_grad():
        P = torch.softmax(m(Xt), 1)
    pred = P.argmax(1)
    fig = plt.figure(figsize=(7.2, 3.3))
    gs = fig.add_gridspec(3, 12, wspace=0.15, hspace=0.55)
    for c in range(10):
        i = (yt == c).nonzero()[0, 0].item()
        ax = fig.add_subplot(gs[c // 5, c % 5]); ax.imshow(Xt[i, 0], cmap=ink); off(ax)
        ax.set_title(FASHION_CLASSES[c], fontsize=7, color=NAVY, pad=2)
    C = np.zeros((10, 10), int); np.add.at(C, (yt.numpy(), pred.numpy()), 1)
    ax = fig.add_subplot(gs[:, 6:])
    offd = C.copy(); np.fill_diagonal(offd, 0)
    ax.imshow(np.log1p(offd), cmap=LinearSegmentedColormap.from_list("c", ["white", MAG]))
    for i in range(10):
        for j in range(10):
            if C[i, j] and (i == j or C[i, j] >= 5):
                ax.text(j, i, C[i, j], ha="center", va="center", fontsize=5.5,
                        color=CYAN if i == j else NAVY, fontweight="bold" if i == j else None)
    short = ["T-sh", "Trou", "Pull", "Dress", "Coat", "Sand", "Shirt", "Snkr", "Bag", "Boot"]
    ax.set_xticks(range(10)); ax.set_xticklabels(short, rotation=90, fontsize=6.5)
    ax.set_yticks(range(10)); ax.set_yticklabels(short, fontsize=6.5); ax.grid(False)
    ax.set_xlabel("predicted", fontsize=7.5); ax.set_ylabel("true", fontsize=7.5)
    ax.set_title("ResNet-20 confusion matrix (test set)", fontsize=8.5)
    fig.text(0.25, 0.27, "Fashion-MNIST: one example per class", ha="center", fontsize=8, color=MUTED)
    save(fig, "ch16_fashion")


def ch16_transfer():
    r = J("ch16/transfer_results.json")
    fig, ax = plt.subplots(figsize=(5.4, 1.9))
    names = list(r)[::-1]; vals = [r[n] * 100 for n in names]
    bars = ax.barh(names, vals, color=[GOLD if "scratch" in n else (CYAN if "frozen" in n else VIOLET) for n in names], height=0.6)
    for b, v in zip(bars, vals):
        ax.text(v - 1, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", ha="right", color="white", fontsize=8)
    ax.set_xlim(0, 100); ax.set_xlabel("target-task test accuracy (%) with 20 labeled images per class")
    ax.grid(axis="y", visible=False)
    save(fig, "ch16_transfer")


# ------------------------------------------------------------------ Ch 17
def ch17_memory():
    r = J("ch17/memory_span.json")
    fig, ax = plt.subplots(figsize=(5.6, 2.4))
    for (name, g), c, ls in zip(r.items(), [NAVY, CYAN, GOLD], ["-", "-.", "--"]):
        ax.plot(range(100), np.maximum(g, 1e-30), color=c, ls=ls, label=name)
    ax.set_yscale("log"); ax.set_ylim(1e-25, 2); ax.set_xlabel("input time step $t$")
    ax.set_ylabel("relative gradient size"); ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("How much does the input at step $t$ influence the output at step 99?")
    save(fig, "ch17_memory")


def ch17_forecast():
    r = J("ch17/forecast.json")
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    days = np.arange(len(r["true"]))
    ax.plot(days, r["true"], color="#9FB3C8", lw=1.1, label="actual")             # light solid (grayscale-safe)
    ax.plot(days, r["pred"], color=NAVY, lw=1.0, ls="--", label="LSTM forecast")  # dark dashed
    ax.set_xlim(0, len(days)); ax.set_xlabel("day of 1990"); ax.set_ylabel("min. temperature (°C)")
    ax.legend(fontsize=7.5, ncol=2, loc="upper center")
    ax.set_title("One-day-ahead forecasts on the test year")
    save(fig, "ch17_forecast")


# ------------------------------------------------------------------ Ch 18
def ch18_positional():
    d, T = 64, 100
    pos = np.arange(T)[:, None]; i = np.arange(d // 2)[None]
    ang = pos / 10000 ** (2 * i / d)
    pe = np.zeros((T, d)); pe[:, 0::2] = np.sin(ang); pe[:, 1::2] = np.cos(ang)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), gridspec_kw={"width_ratios": [1.3, 1]})
    axes[0].imshow(pe.T, cmap=div, aspect="auto", vmin=-1, vmax=1)
    axes[0].set_xlabel("position in the sequence"); axes[0].set_ylabel("embedding dimension")
    axes[0].set_title("(a) sinusoidal positional encodings"); axes[0].grid(False)
    for k, c, ls in zip([0, 4, 16], [NAVY, CYAN, GOLD], ["-", "--", ":"]):   # grayscale: line style per dim
        axes[1].plot(pe[:, 2 * k], color=c, ls=ls, lw=1.4, label=f"dim {2 * k}")
    axes[1].set_xlabel("position"); axes[1].set_ylim(-1.1, 1.6)
    axes[1].legend(fontsize=7, loc="upper center", ncol=3)
    axes[1].set_title("(b) fast and slow waves")
    save(fig, "ch18_positional")


def ch18_compare():
    a, b = J("ch17/char_lstm_log.json"), J("ch18/mini_gpt_log.json")
    fig, ax = plt.subplots(figsize=(5.4, 2.4))
    for log, c, lab, mk in [(a, "#9FB3C8", "LSTM", "o"), (b, NAVY, "mini-GPT", "s")]:   # grayscale: light/open vs dark/square
        s = [r[0] for r in log]
        ax.plot(s, [r[2] for r in log], "-", marker=mk, color=c, ms=4, mec=NAVY, label=f"{lab} (validation)")
        ax.plot(s, [r[1] for r in log], ":", color=c, lw=1.2)
    ax.set_xlabel("training step"); ax.set_ylabel("cross-entropy (nats/char)")
    ax.legend(fontsize=7.5); ax.set_ylim(1.2, 4.5)
    ax.set_title("Same data, steps and batch size (dotted: training)")
    save(fig, "ch18_compare")


def ch18_attention():
    import torch
    from mini_gpt import MiniGPT, chars, encode
    m = MiniGPT(len(chars)); m.load_state_dict(torch.load(CODE + "ch18/mini_gpt.pt")); m.eval()
    text = "ROMEO:\nBut soft, what light"
    with torch.no_grad():
        m(encode(text)[None])
    labels = [c if c != "\n" else "⏎" for c in text]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0))
    titles = ["previous character", "spaces and punctuation", "longer range"]
    for ax, (layer, head), sub in zip(axes, [(0, 3), (3, 1), (3, 3)], titles):
        W = m.blocks[layer].attn.last_weights[0, head].numpy()
        ax.imshow(W, cmap=LinearSegmentedColormap.from_list("a", ["white", VIOLET]), vmin=0, vmax=1)
        ax.set_xticks(range(len(text))); ax.set_xticklabels(labels, fontsize=6, family="DejaVu Sans Mono")
        ax.set_yticks(range(len(text))); ax.set_yticklabels(labels, fontsize=6, family="DejaVu Sans Mono")
        ax.tick_params(length=0, pad=1); ax.grid(False)
        ax.set_title(f"layer {layer + 1}, head {head + 1}\n({sub})", fontsize=8.2)
    save(fig, "ch18_attention")


# ------------------------------------------------------------------ Ch 19
def ch19_recon():
    import torch
    from autoencoder import AutoEncoder
    from wtlkit import mnist_tensors
    m = AutoEncoder(16); m.load_state_dict(torch.load(CODE + "ch19/autoencoder.pt")); m.eval()
    _, _, (Xt, _) = mnist_tensors()
    _, _, (Xf, _) = mnist_tensors(dataset="fashion")
    d = torch.load(CODE + "ch19/denoise_examples.pt")
    with torch.no_grad():
        rd, rf = m(Xt[:8]), m(Xf[:8])
    rows = [("digits", Xt[:8]), ("rebuilt", rd), ("clothes", Xf[:8]), ("rebuilt", rf),
            ("noisy", d["noisy"]), ("denoised", d["denoised"])]
    fig, axes = plt.subplots(6, 8, figsize=(5.4, 4.3))
    for r, (lab, imgs) in enumerate(rows):
        for c in range(8):
            axes[r, c].imshow(imgs[c].view(28, 28), cmap=ink, vmin=0, vmax=1); off(axes[r, c])
        axes[r, 0].set_ylabel(lab, fontsize=7.5, color=MAG if lab in ("clothes", "noisy") else NAVY,
                              rotation=0, ha="right", va="center")
    fig.subplots_adjust(wspace=0.05, hspace=0.08)
    save(fig, "ch19_recon")


def ch19_anomaly():
    r = J("ch19/anomaly_scores.json")
    fig, ax = plt.subplots(figsize=(5.6, 2.3))
    bins = np.geomspace(1e-3, 0.3, 60)
    ax.hist(r["digits"], bins=bins, color=CYAN, alpha=0.75, label="MNIST digits (normal)")
    ax.hist(r["clothes"], bins=bins, color=MAG, alpha=0.6, label="Fashion-MNIST clothes (anomalies)")
    ax.axvline(r["threshold"], color=NAVY, ls="--", lw=1.2)
    ax.text(r["threshold"] * 1.08, ax.get_ylim()[1] * 0.85, "threshold", fontsize=7.5, color=NAVY)
    ax.set_xscale("log"); ax.set_xlabel("reconstruction error (MSE, log scale)"); ax.set_ylabel("images")
    ax.legend(fontsize=7.3, loc="upper left")
    save(fig, "ch19_anomaly")


def ch19_vae():
    import torch
    from vae import VAE
    from wtlkit import mnist_tensors
    m = VAE(); m.load_state_dict(torch.load(CODE + "ch19/vae.pt")); m.eval()
    _, _, (Xt, yt) = mnist_tensors()
    with torch.no_grad():
        mu = m(Xt)[1].numpy()
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    ax = axes[0]
    for d in range(10):
        p = mu[yt.numpy() == d]
        ax.scatter(p[:, 0], p[:, 1], s=1.2, color=cmap(d), alpha=0.6)
        cx, cy = np.median(p, 0)
        ax.text(cx, cy, str(d), fontsize=10, fontweight="bold", color="white",
                ha="center", va="center", bbox=dict(boxstyle="circle,pad=0.15", fc=cmap(d), ec="none"))
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect("equal")
    ax.set_title("(a) test digits in the 2-D latent space"); ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$")
    n = 15; g = np.linspace(-2.5, 2.5, n)
    zz = torch.tensor([[x, y] for y in g[::-1] for x in g], dtype=torch.float32)
    with torch.no_grad():
        imgs = torch.sigmoid(m.dec(zz)).view(n, n, 28, 28).numpy()
    canvas = np.vstack([np.hstack(list(row)) for row in imgs])
    axes[1].imshow(canvas, cmap=ink); off(axes[1])
    axes[1].set_title("(b) decoding a grid of latent points")
    save(fig, "ch19_vae")


# ------------------------------------------------------------------ Ch 20
def ch20_gan():
    r = J("ch20/gan_run.json")
    fig = plt.figure(figsize=(7.2, 3.1))
    gs = fig.add_gridspec(4, 25, wspace=0.08, hspace=0.08)
    for k, ep in enumerate(["1", "5", "30"]):
        imgs = np.array(r["snapshots"][ep]).reshape(-1, 28, 28)
        for i in range(16):
            ax = fig.add_subplot(gs[i // 4, k * 8 + (i % 4) * 2:k * 8 + (i % 4) * 2 + 2])
            ax.imshow(imgs[i], cmap=ink, vmin=0, vmax=1); off(ax)
        fig.text(0.155 + k * 0.32, 0.92, f"after epoch {ep}", ha="center", fontsize=9, color=NAVY, fontweight="semibold")
    save(fig, "ch20_gan")


def ch20_gan_curves():
    log = np.array(J("ch20/gan_run.json")["log"])
    fig, ax = plt.subplots(figsize=(5.4, 2.2))
    ax.plot(log[:, 0], log[:, 3], color=NAVY, marker="o", ms=3, mfc="white", label="D(real images)")
    ax.plot(log[:, 0], log[:, 4], color=MAG, ls="--", marker="s", ms=3, label="D(generated images)")
    ax.axhline(0.5, color=MUTED, ls=":", lw=1)
    ax.set_ylim(0, 1); ax.set_xlabel("epoch"); ax.set_ylabel("mean probability 'real'")
    ax.legend(fontsize=7.5); ax.set_title("The discriminator's verdicts during training")
    save(fig, "ch20_gan_curves")


def ch20_diffusion():
    r = J("ch20/diffusion_frames.json")
    fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.1))
    for ax, t in zip(axes[0], ["0", "20", "60", "120", "199"]):   # T = 200 (P5-CH20-201 fallback)
        p = np.array(r["forward"][t]); ax.scatter(p[:, 0], p[:, 1], s=0.6, color=CYAN, alpha=0.6)
        ax.set_title(f"t = {t}", fontsize=8.5)
    for ax, t in zip(axes[1], ["199", "150", "100", "50", "0"]):
        p = np.array(r["reverse"][t]); ax.scatter(p[:, 0], p[:, 1], s=0.6, color=MAG, alpha=0.6)
        ax.set_title(f"t = {t}", fontsize=8.5)
    for ax in axes.ravel():
        ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2); ax.set_aspect("equal"); off(ax)
    axes[0, 0].set_ylabel("forward: add noise", fontsize=8, color=CYAN)
    axes[1, 0].set_ylabel("reverse: denoise", fontsize=8, color=MAG)
    for ax in axes[:, 0]:
        ax.yaxis.set_visible(True); ax.set_yticks([])
    save(fig, "ch20_diffusion")


# ------------------------------------------------------------------ Ch 21
def ch21_karate():
    r = J("ch21/karate.json")
    pos = {int(k): v for k, v in r["pos"].items()}
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for i, j in r["edges"]:
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color=GRID, lw=0.9, zorder=1)
    # the two predicted factions differ by marker shape AND fill: faction 0 filled circles, faction 1 hollow squares
    for i, p in pos.items():
        f0 = r["pred"][i] == 0
        wrong = r["pred"][i] != r["y"][i]
        ax.scatter(*p, s=190 if i in (0, 33) else 100, marker="o" if f0 else "s",
                   facecolor=NAVY if f0 else "white", edgecolor=MAG if wrong else NAVY,
                   linewidth=2.6 if i in (0, 33) else (1.6 if wrong else 0.9), zorder=3)
        ax.text(*p, str(i), fontsize=5.5, ha="center", va="center", color="white" if f0 else NAVY, zorder=4)
    for i in [k for k in pos if r["pred"][k] != r["y"][k]]:        # label the mistake directly
        ax.annotate(f"member {i}: predicted faction {r['pred'][i]},\ntrue faction {r['y'][i]}", pos[i],
                    xytext=(-62, 28), textcoords="offset points", fontsize=6.5, color=MAG, ha="center",
                    arrowprops=dict(arrowstyle="->", color=MAG, lw=0.8, shrinkB=7))
    off(ax)
    ax.scatter([], [], marker="o", color=NAVY, s=50, label="predicted faction 0 (filled circles)")
    ax.scatter([], [], marker="s", facecolor="white", edgecolor=NAVY, s=50, label="predicted faction 1 (hollow squares)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=6.5, handletextpad=0.3, columnspacing=1.2)
    ax.set_title("Predicted factions (thick rings: the two labeled members)", fontsize=8)
    save(fig, "ch21_karate")


def ch21_cora():
    r = J("ch21/cora_results.json")
    fig, ax = plt.subplots(figsize=(5.0, 1.6))
    names = list(r)[::-1]
    m = [np.mean(r[n]) * 100 for n in names]; s = [np.std(r[n], ddof=1) * 100 for n in names]
    bars = ax.barh(names, m, color=[VIOLET if "GCN" in n else GOLD for n in names], height=0.55)
    for b, v in zip(bars, m):
        ax.text(v - 3, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", ha="right", color="white", fontsize=8)
    ax.set_xlim(0, 100); ax.set_xlabel("Cora test accuracy (%), mean of 5 seeds"); ax.grid(axis="y", visible=False)
    save(fig, "ch21_cora")


# ------------------------------------------------------------------ Ch 22
def ch22_permuted():
    r = J("ch22/inductive_bias.json")
    from mnist_loader import load_mnist
    import torch
    (X, y), _, _ = load_mnist()
    perm = np.array(r["perm"])
    fig = plt.figure(figsize=(7.2, 2.5))
    gs = fig.add_gridspec(2, 9, wspace=0.3, hspace=0.25)
    for k in range(3):
        a = fig.add_subplot(gs[0, k]); a.imshow(X[k].reshape(28, 28), cmap=ink); off(a)
        b = fig.add_subplot(gs[1, k]); b.imshow(X[k][perm].reshape(28, 28), cmap=ink); off(b)
        if k == 0:
            a.set_ylabel("normal", fontsize=8, color=NAVY); b.set_ylabel("shuffled", fontsize=8, color=MAG)
            a.yaxis.set_visible(True); b.yaxis.set_visible(True); a.set_yticks([]); b.set_yticks([])
    ax = fig.add_subplot(gs[:, 4:])
    res = r["results"]
    groups = ["MLP", "CNN"]; x = np.arange(2)
    normal = [res[f"{g}, normal"] * 100 for g in groups]; shuf = [res[f"{g}, shuffled"] * 100 for g in groups]
    b1 = ax.bar(x - 0.18, normal, 0.34, color=NAVY, label="normal pixels")
    b2 = ax.bar(x + 0.18, shuf, 0.34, color="#9FB3C8", hatch="///", edgecolor=NAVY, lw=0.6, label="shuffled pixels")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{b.get_height():.2f}",
                ha="center", fontsize=7.5, color=NAVY)
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylim(90, 100.5)
    ax.set_ylabel("test accuracy (%)"); ax.legend(fontsize=7.3, loc="upper left"); ax.grid(axis="x", visible=False)
    save(fig, "ch22_permuted")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        globals()[name]()
