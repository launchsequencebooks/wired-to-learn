# Gradient checking: compare backprop's gradients with numerical ones.
import numpy as np
from mlp import init_params, forward, backward, cross_entropy

rng = np.random.default_rng(0)
X = rng.normal(size=(5, 4))            # 5 examples, 4 features
y = rng.integers(0, 3, size=5)         # 3 classes
params = init_params([4, 6, 5, 3], seed=1)

P, cache = forward(X, params)
grads = backward(P, y, params, cache)

def loss_with(params):
    return cross_entropy(forward(X, params)[0], y)

h = 1e-6
for layer, ((W, b), (dW, db)) in enumerate(zip(params, grads)):
    for name, theta, g in [("W", W, dW), ("b", b, db)]:
        num = np.zeros_like(theta)
        for idx in np.ndindex(theta.shape):          # every single parameter
            old = theta[idx]
            theta[idx] = old + h; up = loss_with(params)
            theta[idx] = old - h; dn = loss_with(params)
            theta[idx] = old
            num[idx] = (up - dn) / (2 * h)
        rel = np.linalg.norm(g - num) / (np.linalg.norm(g) + np.linalg.norm(num))
        print(f"layer {layer+1} {name}: shape {str(theta.shape):8s} relative error {rel:.1e}")
