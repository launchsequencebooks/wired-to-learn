"""TinyNet layers. Every layer has forward() and backward(); layers with
weights also expose params and grads as parallel lists of arrays."""
import numpy as np


class Layer:
    params, grads = (), ()

    def forward(self, X, training=True):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError


class Dense(Layer):
    """Fully connected layer: Y = X W + b."""

    def __init__(self, n_in, n_out, init="he", rng=None):
        rng = rng or np.random.default_rng()
        scale = np.sqrt(2.0 / n_in) if init == "he" else np.sqrt(1.0 / n_in)
        self.W = rng.normal(0.0, scale, (n_in, n_out))
        self.b = np.zeros(n_out)
        self.dW, self.db = np.zeros_like(self.W), np.zeros_like(self.b)
        self.params, self.grads = [self.W, self.b], [self.dW, self.db]

    def forward(self, X, training=True):
        self.X = X                                  # cache for backward
        return X @ self.W + self.b

    def backward(self, dout):
        self.dW[...] = self.X.T @ dout              # write into the same arrays
        self.db[...] = dout.sum(axis=0)             # so the optimizer sees them
        return dout @ self.W.T


class ReLU(Layer):
    def forward(self, X, training=True):
        self.mask = X > 0
        return X * self.mask

    def backward(self, dout):
        return dout * self.mask


class Sigmoid(Layer):
    def forward(self, X, training=True):
        self.out = 1.0 / (1.0 + np.exp(-X))
        return self.out

    def backward(self, dout):
        return dout * self.out * (1.0 - self.out)


class Tanh(Layer):
    def forward(self, X, training=True):
        self.out = np.tanh(X)
        return self.out

    def backward(self, dout):
        return dout * (1.0 - self.out ** 2)


class Dropout(Layer):
    """Randomly zeroes a fraction p of activations during training only
    (inverted dropout: survivors are scaled by 1/(1-p)). See Chapter 13."""

    def __init__(self, p=0.5, rng=None):
        self.p, self.rng = p, rng or np.random.default_rng()

    def forward(self, X, training=True):
        if not training or self.p == 0:
            self.mask = None
            return X
        self.mask = (self.rng.random(X.shape) >= self.p) / (1.0 - self.p)
        return X * self.mask

    def backward(self, dout):
        return dout if self.mask is None else dout * self.mask
