# Eight training images and two random views of each: what the contrastive loss sees.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wtlkit import mnist_tensors, set_seed
from simclr import augment

set_seed(0)
(X_tr, _), _, _ = mnist_tensors(dataset="fashion", image=True)
x = X_tr[:8]
rows = [("original", x), ("view 1", augment(x.clone())), ("view 2", augment(x.clone()))]
fig, axes = plt.subplots(3, 8, figsize=(7.2, 2.9))
for r, (name, imgs) in enumerate(rows):
    for c in range(8):
        axes[r, c].imshow(imgs[c, 0], cmap="gray", vmin=0, vmax=1)
        axes[r, c].axis("off")
    axes[r, 0].set_title(name, loc="left", fontsize=8)
fig.savefig("ch23_views.png", dpi=200, bbox_inches="tight")   # copy into figs/
