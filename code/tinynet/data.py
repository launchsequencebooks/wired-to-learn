"""Data helpers: mini-batches, splits and scaling."""
import numpy as np


def batches(X, y, batch_size=32, shuffle=True, rng=None):
    """Yield (X_batch, y_batch) pairs covering the data once (one epoch)."""
    idx = np.arange(len(X))
    if shuffle:
        (rng or np.random.default_rng()).shuffle(idx)
    for start in range(0, len(X), batch_size):
        j = idx[start:start + batch_size]
        yield X[j], y[j]


def train_test_split(X, y, test_fraction=0.2, seed=0):
    idx = np.random.default_rng(seed).permutation(len(X))
    n_test = int(len(X) * test_fraction)
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], X[te], y[tr], y[te]


class Standardizer:
    """Rescale each feature to mean 0 and standard deviation 1, using
    statistics learned from the training set only."""

    def fit(self, X):
        self.mean, self.std = X.mean(axis=0), X.std(axis=0) + 1e-8
        return self

    def transform(self, X):
        return (X - self.mean) / self.std
