# Chapter 1's XOR network, rewritten with TinyNet.
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import tinynet as tn

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 1, 1, 0])

rng = np.random.default_rng(0)
model = tn.Sequential(tn.Dense(2, 4, init="xavier", rng=rng), tn.Tanh(),
                      tn.Dense(4, 1, init="xavier", rng=rng))
opt = tn.SGD(model.params, model.grads, lr=0.5)
model.fit(X, y, tn.BinaryCrossEntropy(), opt, epochs=300, batch_size=4, verbose=100)
print("predictions:", model.predict(X), " probabilities of class 1:",
      model.predict_proba(X)[:, 0].round(3))
