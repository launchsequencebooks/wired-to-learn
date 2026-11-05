# Gradient-check TinyNet: every layer type and every loss.
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # find tinynet/
import numpy as np
import tinynet as tn

rng = np.random.default_rng(0)
X = rng.normal(size=(8, 5))
cases = {
    "ReLU + softmax CE": (tn.Sequential(tn.Dense(5, 7, rng=rng), tn.ReLU(), tn.Dense(7, 3, rng=rng)),
                          tn.SoftmaxCrossEntropy(), rng.integers(0, 3, 8)),
    "Tanh + binary CE": (tn.Sequential(tn.Dense(5, 6, rng=rng), tn.Tanh(), tn.Dense(6, 1, rng=rng)),
                         tn.BinaryCrossEntropy(), rng.integers(0, 2, 8)),
    "Sigmoid + MSE":    (tn.Sequential(tn.Dense(5, 4, rng=rng), tn.Sigmoid(), tn.Dense(4, 2, rng=rng)),
                         tn.MSE(), rng.normal(size=(8, 2))),
    "Dropout + softmax CE": (tn.Sequential(tn.Dense(5, 7, rng=rng), tn.ReLU(),
                                           tn.Dropout(0.5, rng=rng),
                                           tn.Dense(7, 3, rng=rng)),
                             tn.SoftmaxCrossEntropy(), rng.integers(0, 3, 8)),
}
for name, (model, loss, y) in cases.items():
    err = tn.gradient_check(model, loss, X, y)
    verdict = "PASS" if err < 1e-5 else "FAIL"
    print(f"{name:21s} worst relative error {err:.1e}  {verdict}")
