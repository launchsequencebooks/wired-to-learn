# Three spirals with TinyNet: a validation split, momentum, and a fair comparison
# (plain SGD at ten times the learning rate, the same effective step as momentum 0.9).
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ch07"))
import numpy as np
import tinynet as tn
from train_spirals import make_spirals

X, y = make_spirals(n_per_class=300)
X_tr, X_val, y_tr, y_val = tn.train_test_split(X, y, test_fraction=0.2)


def build(seed=0):
    rng = np.random.default_rng(seed)
    return tn.Sequential(tn.Dense(2, 64, rng=rng), tn.ReLU(),
                         tn.Dense(64, 64, rng=rng), tn.ReLU(),
                         tn.Dense(64, 3, rng=rng))


build().summary()
histories = {}
for lr, momentum in [(0.05, 0.0), (0.5, 0.0), (0.05, 0.9)]:
    model = build()
    opt = tn.SGD(model.params, model.grads, lr=lr, momentum=momentum)
    print(f"\n--- lr = {lr}, momentum = {momentum}")
    histories[(lr, momentum)] = model.fit(X_tr, y_tr, tn.SoftmaxCrossEntropy(), opt,
                                          epochs=60, batch_size=32,
                                          val_data=(X_val, y_val), verbose=20)
np.save("spiral_histories.npy", histories, allow_pickle=True)
