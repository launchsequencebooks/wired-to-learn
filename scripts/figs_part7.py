"""Figures for Part VII (Chapters 29-36)."""
import sys, json
import numpy as np
import matplotlib.pyplot as plt
from style import CYAN, GOLD, MAG, VIOLET, NAVY, MUTED, GRID, save

CODE = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "code") + "/"  # was /home/claude/book/code/
sys.path.insert(0, CODE)
J = lambda p: json.load(open(CODE + p))


def ch29_coverage():
    r = J("ch29/coverage_curve.json")
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    ax.plot(np.array(r["coverage"]) * 100, np.array(r["accuracy"]) * 100, color=CYAN, lw=2)
    for target, col in [(99, MAG)]:
        cov = np.array(r["coverage"]) * 100; acc = np.array(r["accuracy"]) * 100
        i = np.nonzero(acc >= target)[0].max()
        ax.plot(cov[i], acc[i], "o", color=col); ax.annotate(f"{cov[i]:.0f}% automated\nat {acc[i]:.1f}% accuracy", (cov[i], acc[i]),
                                                             xytext=(cov[i] - 45, acc[i] - 3.5), fontsize=7.5, color=NAVY,
                                                             arrowprops=dict(arrowstyle="->", color=NAVY))
    ax.set_xlabel("share of items handled automatically (%)"); ax.set_ylabel("accuracy on those items (%)")
    ax.set_ylim(90, 100.3); ax.set_xlim(0, 101)
    save(fig, "ch29_coverage")


def ch31_models():
    names = ["TF-IDF +\nlogistic reg.", "bag of\nembeddings", "Transformer", "ensemble"]
    b, t, e = J("ch31/baseline_results.json"), J("ch31/triage_results.json"), J("ch31/ensemble_results.json")
    rows = [b, t["bag of embeddings"], t["Transformer"], e]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    for off, key, col, lab in [(-0.18, "acc", CYAN, "77 intents"), (0.18, "team", GOLD, "6 teams")]:
        v = [r[key] * 100 for r in rows]
        ax.bar(x + off, v, 0.36, color=col, label=lab)
        for xi, vi in zip(x + off, v):
            ax.text(xi, vi + 0.3, f"{vi:.1f}", ha="center", fontsize=7, color=NAVY)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.5); ax.set_ylim(80, 100)
    ax.set_ylabel("test accuracy (%)"); ax.legend(fontsize=7.5, loc="upper left", ncol=2); ax.grid(axis="x", visible=False)
    save(fig, "ch31_models")


def ch32_detectors():
    import re
    log = open(CODE + "ch32/detectors.log").read()
    rows = re.findall(r"^(.+?)\s+precision ([\d.]+)\s+recall ([\d.]+)", log, re.M)
    ceiling = float(re.search(r"best possible recall ([\d.]+) from order", log).group(1))
    names = [r[0].replace(" (top-3)", "").replace("rule: ", "rule:\n").replace("autoencoder on", "autoencoder on\n").replace("LSTM next-event", "LSTM\nnext-event") for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    for off, k, col, lab, hat in [(-0.18, 1, NAVY, "precision", ""), (0.18, 2, "#9FB3C8", "recall", "///")]:
        v = [float(r[k]) for r in rows]
        ax.bar(x + off, v, 0.36, color=col, hatch=hat, edgecolor=NAVY, lw=0.6, label=lab)
    ax.axhline(ceiling, color=MAG, ls="--", lw=1.2)
    ax.text(-0.45, ceiling + 0.02, "best possible recall", color=MAG, fontsize=7, ha="left")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.5); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7.5, loc="upper right", bbox_to_anchor=(1, 0.88)); ax.grid(axis="x", visible=False)
    save(fig, "ch32_detectors")


def ch33_week():
    import pandas as pd
    sys.path.insert(0, CODE + "ch33")
    from demand_data import TIME, df
    r = J("ch33/forecasts.json")
    q = np.expm1(np.array(r["forecasts"]["GRU"])).clip(min=0)
    origins = r["origins"]
    start = [i for i, o in enumerate(origins) if TIME[o] == pd.Timestamp("2012-12-10")][0]
    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    for d in range(7):
        i = start + d; t = TIME[origins[i]:origins[i] + 24]
        ax.fill_between(t, q[i, :, 0], q[i, :, 2], color=CYAN, alpha=0.25, lw=0, label="80% interval" if d == 0 else None)
        ax.plot(t, q[i, :, 1], color=CYAN, lw=1.6, label="GRU forecast (median)" if d == 0 else None)
        ax.plot(t, df["cnt"].values[origins[i]:origins[i] + 24], ".", color=MAG, ms=3, label="actual" if d == 0 else None)
    ax.set_ylabel("rentals per hour"); ax.legend(fontsize=7, loc="upper left", ncol=3)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b")); ax.tick_params(axis="x", labelsize=7)
    save(fig, "ch33_week")


def fig34_2():
    """Figure 34.2 (ch34_embeddings.pdf); see ch34_embeddings."""
    return ch34_embeddings()


def ch34_embeddings():
    import torch, torch.nn.functional as F
    from sklearn.manifold import TSNE
    sys.path.insert(0, CODE + "ch34")
    import os; os.chdir(CODE + "ch34")
    from rec_data import ITEMS, TRAIN, N_ITEMS
    from recommenders import MFWithGenres
    m = MFWithGenres(); m.load_state_dict(torch.load("mf_genres.pt")); m.eval()
    cnt = torch.bincount(TRAIN[:, 1], minlength=N_ITEMS)
    top = cnt.argsort(descending=True)[:600]
    with torch.no_grad():
        V = F.normalize(m.item_vectors(top), dim=1).numpy()
    emb = TSNE(n_components=2, random_state=0, init="pca", perplexity=30, metric="cosine").fit_transform(V)
    # four genre groups plus "other", each with its own marker shape and a colour of distinct luminance
    groups = [("Children's / Animation", ("Children's", "Animation"), "D", GOLD),
              ("Action / Thriller / Sci-Fi", ("Action", "Thriller", "Sci-Fi"), "s", NAVY),
              ("Comedy", ("Comedy",), "^", CYAN),
              ("Drama / Romance", ("Drama", "Romance"), "o", "#9FB3C8")]
    def group_of(genres):
        for name, members, _, _ in groups:              # first matching group wins (children's before action)
            if any(x in members for x in genres):
                return name
        return "other"
    g = [group_of(ITEMS[i][1]) for i in top.tolist()]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name, _, mk, c in groups:
        idx = [j for j, x in enumerate(g) if x == name]
        ax.scatter(emb[idx, 0], emb[idx, 1], s=14, marker=mk, color=c, edgecolor="white", linewidth=0.3, label=name, alpha=0.9)
    other = [j for j, x in enumerate(g) if x == "other"]
    ax.scatter(emb[other, 0], emb[other, 1], s=14, marker="x", color=MUTED, linewidth=0.7, label="other")
    # at most eight named films, each with a short leader line; labels are pushed apart (adjustText-style)
    names = ["Star Wars (1977)", "Return of the Jedi (1983)", "Toy Story (1995)", "Lion King, The (1994)",
             "Fargo (1996)", "Titanic (1997)", "Godfather, The (1972)", "Independence Day (ID4) (1996)"]
    titles = [ITEMS[i][0] for i in top.tolist()]
    pts = [(n.replace(", The", "").rsplit(" (", 1)[0], emb[titles.index(n)]) for n in names if n in titles][:8]
    span = emb.max(0) - emb.min(0)
    rng = np.random.default_rng(0)
    lab = np.array([p + span * 0.06 * rng.choice([-1, 1], 2) for _, p in pts])
    for _ in range(400):                              # simple repulsion between labels and from every point
        for a in range(len(lab)):
            push = np.zeros(2)
            for b_ in range(len(lab)):
                if a != b_:                               # labels are wide: ask for twice the separation in x
                    d = lab[a] - lab[b_]; dist = np.linalg.norm(d / span / [2.0, 1.0]) + 1e-9
                    if dist < 0.12: push += d / span / dist * (0.12 - dist)
            d = lab[a] - pts[a][1]; dist = np.linalg.norm(d / span) + 1e-9
            if dist > 0.16: push -= d / span / dist * (dist - 0.16)      # keep the leader line short
            if dist < 0.05: push += d / span / dist * (0.05 - dist)
            lab[a] += push * span * 0.5
    for (name, p), l in zip(pts, lab):
        ax.annotate(name, p, xytext=l, textcoords="data", fontsize=6.8, color="black", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                    arrowprops=dict(arrowstyle="-", color="black", lw=0.6, shrinkA=0, shrinkB=2), zorder=6)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.legend(fontsize=6.5, loc="lower left", ncol=5, markerscale=1.3, columnspacing=0.8, handletextpad=0.2)
    save(fig, "ch34_embeddings")


def ch36_calibration():
    r = J("ch36/calibration.json")["ticket ensemble"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
    for ax, key, title in [(axes[0], "before", "before (ECE 15%)"), (axes[1], "after", f"after, T = {r['T']:.2f} (ECE 1%)")]:
        t = np.array(r[key])
        ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
        ax.plot(t[:, 0], t[:, 1], "-", color=CYAN, lw=1.2)
        ax.scatter(t[:, 0], t[:, 1], s=8 + 300 * t[:, 2] / t[:, 2].max(), color=CYAN, zorder=3, alpha=0.8)
        ax.set_title(title, fontsize=9); ax.set_xlabel("confidence"); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("accuracy")
    save(fig, "ch36_calibration")


def ch36_drift():
    """Figure 36.2: four panels from code/ch36/drift.json (input PSI, confidence PSI, prediction PSI,
    weekly accuracy), one series per panel, the 0.2 alert line dashed (grayscale-safe)."""
    r = J("ch36/drift.json")
    w = [x["week"] for x in r]
    has_pred = all("psi_pred" in x for x in r)
    if not has_pred:
        print("ch36_drift: drift.json has no psi_pred column; re-run code/ch36/drift.py for the four-panel figure")
    panels = [("input PSI (brightness)", "psi_input", 1), ("confidence PSI", "psi_conf", 1),
              ("prediction PSI", "psi_pred", 1), ("weekly accuracy (%)", "accuracy", 100)]
    panels = [p for p in panels if p[1] in r[0]]          # until the re-run, the interim figure has three panels
    fig, axes = plt.subplots(1, len(panels), figsize=(7.4, 2.3))
    psi_max = max(x[k] for x in r for k in ("psi_input", "psi_conf", "psi_pred") if k in x) * 1.15
    for k, (ax, (title, key, scale)) in enumerate(zip(axes, panels)):
        title = f"({'abcd'[k]}) {title}"
        ax.axvspan(4.5, 8.5, color=GRID, alpha=0.7, lw=0)
        ax.plot(w, [x[key] * scale for x in r], "-o", color=NAVY, ms=4, lw=1.6)
        if key.startswith("psi"):
            ax.axhline(0.2, color=MAG, ls="--", lw=1); ax.set_ylim(0, max(0.65, psi_max))
            if key == "psi_input":
                ax.text(1.0, 0.215, "alert line (0.2)", fontsize=6.5, color=MAG)
        else:
            ax.set_ylim(85, 96)
        ax.set_title(title, fontsize=8.5); ax.set_xlabel("week"); ax.set_xticks(w)
    axes[1].text(4.7, 0.6, "new camera\nfrom week 5", fontsize=6.5, color=MUTED, va="top")
    fig.tight_layout(w_pad=0.5)
    save(fig, "ch36_drift")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        globals()[name]()
