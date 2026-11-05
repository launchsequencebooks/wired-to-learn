"""Check a whole model's backpropagation against numerical gradients."""
import numpy as np


def _reseed(model, seed):
    for layer in model.layers:                  # make Dropout masks repeatable
        if hasattr(layer, "rng"):
            layer.rng = np.random.default_rng(seed)


def gradient_check(model, loss, X, y, h=1e-6, max_checks=40, seed=0):
    """Return the worst relative error over a random sample of parameters."""
    rng = np.random.default_rng(seed)
    _reseed(model, seed)
    loss.forward(model.forward(X), y)
    model.backward(loss.backward())
    analytic = [g.copy() for g in model.grads]
    worst = 0.0
    for p, g in zip(model.params, analytic):
        for _ in range(min(max_checks, p.size)):
            i = tuple(rng.integers(0, s) for s in p.shape)
            old = p[i]
            p[i] = old + h; _reseed(model, seed); up = loss.forward(model.forward(X), y)
            p[i] = old - h; _reseed(model, seed); dn = loss.forward(model.forward(X), y)
            p[i] = old
            num = (up - dn) / (2 * h)
            rel = abs(num - g[i]) / max(abs(num) + abs(g[i]), 1e-12)
            worst = max(worst, rel)
    return worst
