# One sigmoid neuron trained with gradient descent: logistic regression
# from scratch. Supports full-batch, mini-batch and stochastic training.
import numpy as np


def make_blobs(n=200, seed=0):
    """Two overlapping clouds of 2-D points with labels 0 and 1."""
    rng = np.random.default_rng(seed)
    X0 = rng.normal([-1.0, -1.0], 0.9, (n // 2, 2))
    X1 = rng.normal([1.2, 1.0], 0.9, (n // 2, 2))
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    return X, y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def bce(p, y, eps=1e-12):
    """Binary cross-entropy averaged over the batch."""
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def train(X, y, lr=0.5, epochs=100, batch_size=None, seed=1):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    batch_size = batch_size or n            # None -> full batch
    history = []
    for epoch in range(epochs):
        order = rng.permutation(n)          # shuffle each epoch
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            Xb, yb = X[idx], y[idx]
            p = sigmoid(Xb @ w + b)         # forward
            error = p - yb                  # dL/dz for sigmoid + BCE
            w -= lr * Xb.T @ error / len(idx)
            b -= lr * error.mean()
        history.append(bce(sigmoid(X @ w + b), y))
    return w, b, history


if __name__ == "__main__":
    X, y = make_blobs()
    for label, bs in [("full batch", None), ("mini-batch 16", 16), ("SGD (1)", 1)]:
        w, b, hist = train(X, y, lr=0.1, epochs=30, batch_size=bs)
        acc = ((sigmoid(X @ w + b) >= 0.5) == y).mean()
        print(f"{label:14s} loss={hist[-1]:.4f}  accuracy={acc:.1%}  "
              f"w={w.round(2)} b={b:.2f}")
