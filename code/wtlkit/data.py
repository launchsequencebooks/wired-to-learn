"""MNIST / Fashion-MNIST as PyTorch tensors and DataLoaders."""
import sys
from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ch09"))
from mnist_loader import load_mnist  # noqa: E402
from .datasets import load_fashion  # noqa: E402


def mnist_tensors(n_train=None, dataset="mnist", image=False):
    """Return (X_tr, y_tr), (X_va, y_va), (X_te, y_te) as torch tensors.
    n_train optionally keeps only the first n_train training images;
    image=True reshapes to (N, 1, 28, 28) for convolutional networks."""
    load = load_mnist if dataset == "mnist" else load_fashion
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load()
    if n_train is not None:
        X_tr, y_tr = X_tr[:n_train], y_tr[:n_train]

    def t(X, y):
        X = torch.from_numpy(X)
        return (X.view(-1, 1, 28, 28) if image else X), torch.from_numpy(y)
    return t(X_tr, y_tr), t(X_va, y_va), t(X_te, y_te)


def make_loaders(batch_size=64, n_train=None, seed=0, dataset="mnist", image=False):
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = mnist_tensors(n_train, dataset, image)
    g = torch.Generator().manual_seed(seed)
    train = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True, generator=g)
    val = DataLoader(TensorDataset(X_va, y_va), batch_size=1024)
    test = DataLoader(TensorDataset(X_te, y_te), batch_size=1024)
    return train, val, test
