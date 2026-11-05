# Use the trained digit reader on an image file of a handwritten digit.
# The image is cleaned up the way MNIST digits were prepared:
# dark-on-light is inverted, the digit is cropped, scaled to fit a 20x20 box,
# and centered by its center of mass in a 28x28 frame.
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image                             # pip install pillow
import tinynet as tn


def build_model():
    return tn.Sequential(tn.Dense(784, 256), tn.ReLU(), tn.Dense(256, 128), tn.ReLU(),
                         tn.Dense(128, 10))


def preprocess(path):
    img = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    if img.mean() > 0.5:                       # dark ink on light paper -> invert
        img = 1.0 - img
    img[img < 0.2] = 0.0                       # remove faint background noise
    rows, cols = np.nonzero(img)
    img = img[rows.min():rows.max() + 1, cols.min():cols.max() + 1]   # crop
    h, w = img.shape
    scale = 20.0 / max(h, w)                   # longest side -> 20 pixels
    new = (max(1, round(w * scale)), max(1, round(h * scale)))
    img = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize(new, Image.LANCZOS),
                     dtype=np.float32) / 255.0
    canvas = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - img.shape[0]) // 2, (28 - img.shape[1]) // 2
    canvas[top:top + img.shape[0], left:left + img.shape[1]] = img
    # shift so the center of mass sits at the middle (14, 14)
    ys, xs = np.indices(canvas.shape)
    cy, cx = (ys * canvas).sum() / canvas.sum(), (xs * canvas).sum() / canvas.sum()
    canvas = np.roll(canvas, (round(14 - cy), round(14 - cx)), axis=(0, 1))
    return canvas.reshape(1, 784)


if __name__ == "__main__":
    model = build_model().load(Path(__file__).with_name("digit_mlp.npz"))
    for path in sys.argv[1:]:
        p = model.predict_proba(preprocess(path))[0]
        top3 = np.argsort(p)[::-1][:3]
        print(f"{Path(path).name}: digit {top3[0]}  "
              f"(confidence {p[top3[0]]:.1%}; next: "
              + ", ".join(f"{d} {p[d]:.1%}" for d in top3[1:]) + ")")
