# Baseline: a single Dense layer + softmax ("softmax regression").
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import tinynet as tn
from mnist_loader import load_mnist

(X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_mnist()

model = tn.Sequential(tn.Dense(784, 10, init="xavier", rng=np.random.default_rng(0)))
opt = tn.SGD(model.params, model.grads, lr=0.1)
model.fit(X_tr, y_tr, tn.SoftmaxCrossEntropy(), opt, epochs=10, batch_size=64,
          val_data=(X_va, y_va), verbose=5)
print(f"test accuracy: {model.accuracy(X_te, y_te):.2%}")
model.save("softmax_regression.npz")
