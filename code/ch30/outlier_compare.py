# Which signal tells 'not clothing' apart? 500 digits vs 2,000 clothes, at 1% false alarms.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.metrics import roc_auc_score
from wtlkit import mnist_tensors
from wtlapps import OnnxModel
from organizer import OutlierDetector

_, (X_va, _), (X, _) = mnist_tensors(dataset="fashion", image=True)
_, _, (D, _) = mnist_tensors(dataset="mnist", image=True)
normal, clothes, digits = X_va[3000:6000].numpy(), X[:2000].numpy(), D[:500].numpy()
det = OutlierDetector(OnnxModel("embed_resnet.onnx"))
clf = OnnxModel(Path(__file__).resolve().parents[1] / "ch28" / "resnet20_int8.onnx")


def nn_distance(x):
    e = det.embedder(x) - det.mean
    return 1 - ((e / np.linalg.norm(e, axis=1, keepdims=True)) @ det.ref.T).max(1)


signals = {"1 - classifier confidence": lambda x: 1 - clf.predict_proba(x).max(1),
           "autoencoder error": lambda x: det.ae(x)[:, 0],
           "distance to nearest known": nn_distance,
           "both combined": det.score}
for name, f in signals.items():
    threshold = np.quantile(f(normal), 0.99)              # 1% of normal clothes will be flagged
    s_c, s_d = f(clothes), f(digits)
    auc = roc_auc_score(np.r_[np.zeros(len(s_c)), np.ones(len(s_d))], np.r_[s_c, s_d])
    print(f"{name:26s} AUC {auc:.3f}   digits caught {np.mean(s_d > threshold):6.1%}   "
          f"clothes flagged {np.mean(s_c > threshold):.1%}")
