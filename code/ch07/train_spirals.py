# Train our NumPy MLP on three intertwined spirals.
import numpy as np
from mlp import init_params, forward, backward, cross_entropy, sgd_step, predict


def make_spirals(n_per_class=200, classes=3, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(classes):
        r = np.linspace(0.05, 1, n_per_class)
        t = np.linspace(c * 4, (c + 1) * 4, n_per_class) + rng.normal(0, noise, n_per_class)
        X.append(np.c_[r * np.sin(t), r * np.cos(t)])
        y.append(np.full(n_per_class, c))
    return np.vstack(X), np.concatenate(y)


if __name__ == "__main__":
    X, y = make_spirals()
    params = init_params([2, 64, 64, 3], seed=0)
    rng = np.random.default_rng(0)
    for epoch in range(1, 301):
        order = rng.permutation(len(X))
        for start in range(0, len(X), 32):              # mini-batches of 32
            idx = order[start:start + 32]
            P, cache = forward(X[idx], params)
            grads = backward(P, y[idx], params, cache)
            params = sgd_step(params, grads, lr=0.05)
        if epoch in (1, 10, 50, 100, 300):
            loss = cross_entropy(forward(X, params)[0], y)
            acc = (predict(X, params) == y).mean()
            print(f"epoch {epoch:3d}  loss {loss:.4f}  accuracy {acc:.1%}")
    np.savez("spiral_params.npz", *[a for Wb in params for a in Wb])
