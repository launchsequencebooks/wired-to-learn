# A complete multilayer perceptron with backpropagation, in plain NumPy.
# Hidden layers use ReLU; the output layer uses softmax + cross-entropy.
import numpy as np


def init_params(layer_sizes, seed=0):
    rng = np.random.default_rng(seed)
    return [(rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_out)), np.zeros(n_out))
            for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]


def softmax(Z):
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


def forward(X, params):
    """Return class probabilities and a cache of every layer's (A_prev, Z)."""
    cache, A = [], X
    for i, (W, b) in enumerate(params):
        Z = A @ W + b
        cache.append((A, Z))                       # remember for backward
        A = np.maximum(0, Z) if i < len(params) - 1 else softmax(Z)
    return A, cache


def cross_entropy(P, y):
    """Mean negative log-probability of the true classes y (integers)."""
    N = len(y)
    return -np.mean(np.log(P[np.arange(N), y] + 1e-12))


def backward(P, y, params, cache):
    """Gradients of the mean cross-entropy w.r.t. every W and b."""
    N = len(y)
    grads = [None] * len(params)
    dZ = P.copy()
    dZ[np.arange(N), y] -= 1                       # (1) output: P - Y
    dZ /= N
    for i in reversed(range(len(params))):
        A_prev = cache[i][0]                       # input to this layer
        W, _ = params[i]
        dW = A_prev.T @ dZ                         # (2) weight gradient
        db = dZ.sum(axis=0)                        # (3) bias gradient
        grads[i] = (dW, db)
        if i > 0:
            dA_prev = dZ @ W.T                     # (4) send blame backward
            Z_prev = cache[i - 1][1]
            dZ = dA_prev * (Z_prev > 0)            # through the ReLU
    return grads


def sgd_step(params, grads, lr):
    return [(W - lr * dW, b - lr * db) for (W, b), (dW, db) in zip(params, grads)]


def predict(X, params):
    return forward(X, params)[0].argmax(axis=1)
