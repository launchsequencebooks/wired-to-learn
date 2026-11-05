# Regression with TinyNet: learn a noisy curve with MSE loss.
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import tinynet as tn

rng = np.random.default_rng(0)
X = rng.uniform(-3, 3, (400, 1))
y = np.sin(2 * X) + 0.3 * X + rng.normal(0, 0.1, X.shape)     # target curve + noise

model = tn.Sequential(tn.Dense(1, 32, rng=rng), tn.Tanh(),
                      tn.Dense(32, 32, rng=rng), tn.Tanh(),
                      tn.Dense(32, 1, rng=rng))
opt = tn.SGD(model.params, model.grads, lr=0.05, momentum=0.9)
loss = tn.MSE()
for epoch in range(1, 301):
    for Xb, yb in tn.batches(X, y, 32, rng=rng):
        loss.forward(model.forward(Xb), yb)
        model.backward(loss.backward())
        opt.step()
    if epoch in (1, 50, 300):
        mse = loss.forward(model.forward(X, training=False), y)
        print(f"epoch {epoch:3d}  MSE {mse:.4f}")
model.save("curve_model.npz")
