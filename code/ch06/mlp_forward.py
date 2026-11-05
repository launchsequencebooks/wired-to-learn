# A forward pass for a fully connected network of any shape,
# applied to a whole batch at once - and why vectorization matters.
import time
import numpy as np


def init_params(layer_sizes, seed=0):
    """One (W, b) pair per layer. He initialization keeps signals stable."""
    rng = np.random.default_rng(seed)
    params = []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        W = rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_out))
        b = np.zeros(n_out)
        params.append((W, b))
    return params


def forward(X, params):
    """ReLU hidden layers, softmax output. X has shape (N, n_in)."""
    A = X
    for i, (W, b) in enumerate(params):
        Z = A @ W + b
        if i < len(params) - 1:
            A = np.maximum(0, Z)                       # hidden: ReLU
        else:
            E = np.exp(Z - Z.max(axis=1, keepdims=True))
            A = E / E.sum(axis=1, keepdims=True)       # output: softmax
    return A


def forward_loops(X, params):
    """The same computation with explicit Python loops (don't do this!)."""
    out = []
    for x in X:                                        # every example
        a = list(x)
        for i, (W, b) in enumerate(params):
            z = [sum(a[k] * W[k, j] for k in range(len(a))) + b[j]
                 for j in range(W.shape[1])]           # every neuron
            if i < len(params) - 1:
                a = [max(0.0, v) for v in z]
            else:
                m = max(z); e = [np.exp(v - m) for v in z]; s = sum(e)
                a = [v / s for v in e]
        out.append(a)
    return np.array(out)


if __name__ == "__main__":
    sizes = [784, 128, 64, 10]
    params = init_params(sizes)
    n_params = sum(W.size + b.size for W, b in params)
    print("layer sizes:", sizes, "| parameters:", n_params)

    X = np.random.default_rng(1).random((64, 784))     # a fake batch of 64 images
    t0 = time.perf_counter(); P = forward(X, params); t_vec = time.perf_counter() - t0
    t0 = time.perf_counter(); Q = forward_loops(X[:4], params); t_loop = time.perf_counter() - t0
    print("output shape:", P.shape, "| rows sum to 1:", np.allclose(P.sum(1), 1))
    print("same answers:", np.allclose(P[:4], Q))
    per_ex_vec, per_ex_loop = t_vec / 64, t_loop / 4
    print(f"speed-up from vectorization: about {per_ex_loop / per_ex_vec:,.0f}x")
