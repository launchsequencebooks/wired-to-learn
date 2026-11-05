# Confusion matrix and the most confident mistakes of the saved digit reader.
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import tinynet as tn
from mnist_loader import load_mnist
from predict_digit import build_model

_, _, (X_te, y_te) = load_mnist()
model = build_model().load(Path(__file__).with_name("digit_mlp.npz"))
P = model.predict_proba(X_te); pred = P.argmax(axis=1)
cm = np.zeros((10, 10), dtype=int)
np.add.at(cm, (y_te, pred), 1)              # rows: true digit, columns: predicted
print(cm)
wrong = np.flatnonzero(pred != y_te)
n_conf = int((P[wrong].max(axis=1) > 0.99).sum())
print(f"{len(wrong)} mistakes, {n_conf} of them made with more than 99% confidence")
worst = wrong[np.argsort(-P[wrong, pred[wrong]])[:8]]
for i in worst:
    print(f"image {i}: true {y_te[i]} read as {pred[i]} ({P[i, pred[i]]:.2%} sure)")
