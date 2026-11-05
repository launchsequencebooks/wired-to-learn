# Eight weeks in production: from week 5, a supplier's new camera makes the photos darker.
import sys, json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import numpy as np
from wtlkit import mnist_tensors
from wtlapps import OnnxModel

model = OnnxModel(root / "ch28" / "resnet20_int8.onnx")
_, (X_ref, _), (X, y) = mnist_tensors(dataset="fashion", image=True)
X_ref, X, y = X_ref.numpy()[:5000], X.numpy(), y.numpy()


def psi(reference, current, bins=10):
    """Population stability index: how far a distribution has moved (above 0.2 usually means 'investigate')."""
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference) + 1e-4
    c = np.histogram(current, edges)[0] / len(current) + 1e-4
    return float(np.sum((c - r) * np.log(c / r)))


brightness = lambda imgs: imgs.reshape(len(imgs), -1).mean(1)
class_shares = lambda p: np.bincount(p.argmax(1), minlength=10) / len(p) + 1e-4
p_ref = model.predict_proba(X_ref)
ref_bright, ref_conf, ref_pred = brightness(X_ref), p_ref.max(1), class_shares(p_ref)
rows = []
print("week  brightness  input PSI  mean confidence  confidence PSI  prediction PSI  accuracy  alert")
for week in range(1, 9):
    imgs, labels = X[(week - 1) * 1250:week * 1250].copy(), y[(week - 1) * 1250:week * 1250]
    if week >= 5:
        imgs[:600] *= 0.6 if week < 7 else 0.4               # half the photos now come from the new camera
    p = model.predict_proba(imgs)
    b, c, r = brightness(imgs), p.max(1), ref_pred
    shares = class_shares(p)                                 # the other label-free output: predicted classes
    pred_psi = float(np.sum((shares - r) * np.log(shares / r)))
    row = {"week": week, "brightness": float(b.mean()), "psi_input": psi(ref_bright, b),
           "confidence": float(c.mean()), "psi_conf": psi(ref_conf, c), "psi_pred": pred_psi,
           "accuracy": float((p.argmax(1) == labels).mean())}
    alert = "DRIFT" if max(row["psi_input"], row["psi_conf"], row["psi_pred"]) > 0.2 else ""
    rows.append(row)
    print(f"{week:4d}  {row['brightness']:10.3f}  {row['psi_input']:9.3f}  {row['confidence']:15.1%}  "
          f"{row['psi_conf']:14.3f}  {row['psi_pred']:14.3f}  {row['accuracy']:8.1%}  {alert}")
json.dump(rows, open("drift.json", "w"), indent=1)

# once labels arrive (here: weeks 7-8), compare photos from the new camera with the rest
imgs, labels = X[7500:10000].copy(), y[7500:10000]
new = np.zeros(len(imgs), bool); new[:600] = new[1250:1850] = True
imgs[new] *= 0.4
ok = model.predict_proba(imgs).argmax(1) == labels
print(f"weeks 7-8, once labeled: accuracy {ok[new].mean():.1%} on new-camera photos, {ok[~new].mean():.1%} on the others")
