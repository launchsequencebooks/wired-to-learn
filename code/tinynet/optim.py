"""TinyNet optimizers. They update parameter arrays in place."""
import numpy as np


class SGD:
    """Stochastic gradient descent with optional momentum and weight decay.
    Momentum (Chapter 11) keeps a running 'velocity' that smooths the steps."""

    def __init__(self, params, grads, lr=0.01, momentum=0.0, weight_decay=0.0):
        self.params, self.grads = params, grads
        self.lr, self.momentum, self.wd = lr, momentum, weight_decay
        self.velocity = [np.zeros_like(p) for p in params]

    def step(self):
        for p, g, v in zip(self.params, self.grads, self.velocity):
            g = g + self.wd * p if self.wd else g
            v *= self.momentum
            v -= self.lr * g
            p += v                                   # in place: the layer sees it
