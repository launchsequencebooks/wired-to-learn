"""TinyNet losses. forward() returns a scalar loss; backward() returns the
gradient of that loss with respect to the network's raw outputs."""
import numpy as np


class SoftmaxCrossEntropy:
    """Softmax + categorical cross-entropy, fused for a simple, stable gradient.
    Targets are integer class labels."""

    def forward(self, logits, y):
        Z = logits - logits.max(axis=1, keepdims=True)
        self.P = np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True)
        self.y = y
        N = len(y)
        return -np.mean(np.log(self.P[np.arange(N), y] + 1e-12))

    def backward(self):
        N = len(self.y)
        d = self.P.copy()
        d[np.arange(N), self.y] -= 1                 # P - Y
        return d / N


class BinaryCrossEntropy:
    """Sigmoid + binary cross-entropy on a single logit column; targets 0/1."""

    def forward(self, logits, y):
        self.p = 1.0 / (1.0 + np.exp(-logits))
        self.y = y.reshape(self.p.shape).astype(float)
        p = np.clip(self.p, 1e-12, 1 - 1e-12)
        return -np.mean(self.y * np.log(p) + (1 - self.y) * np.log(1 - p))

    def backward(self):
        return (self.p - self.y) / len(self.y)


class MSE:
    """Mean squared error for regression; targets have the output's shape."""

    def forward(self, pred, y):
        self.diff = pred - y.reshape(pred.shape)
        return np.mean(self.diff ** 2)

    def backward(self):
        return 2.0 * self.diff / self.diff.size
