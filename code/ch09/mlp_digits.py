# The digit reader: a 784-256-128-10 network trained with TinyNet.
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import tinynet as tn
from mnist_loader import load_mnist

(X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_mnist()
rng = np.random.default_rng(0)
model = tn.Sequential(
    tn.Dense(784, 256, rng=rng), tn.ReLU(),
    tn.Dense(256, 128, rng=rng), tn.ReLU(),
    tn.Dense(128, 10, rng=rng),
)
model.summary()
opt = tn.SGD(model.params, model.grads, lr=0.05, momentum=0.9)
history = model.fit(X_tr, y_tr, tn.SoftmaxCrossEntropy(), opt, epochs=15,
                    batch_size=64, val_data=(X_va, y_va), verbose=3)
print(f"test accuracy: {model.accuracy(X_te, y_te):.2%}")
model.save("digit_mlp.npz")
np.save("digit_mlp_history.npy", history, allow_pickle=True)
