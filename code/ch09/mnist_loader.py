"""Load MNIST: 70,000 handwritten digits (28x28 grayscale).
Looks for code/data/mnist.npz; downloads and checks it on first use if missing."""
from pathlib import Path
import hashlib
import urllib.request
import numpy as np

URLS = ["https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
        "https://s3.amazonaws.com/img-datasets/mnist.npz"]        # mirror
SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
PATH = Path(__file__).resolve().parents[1] / "data" / "mnist.npz"


def load_mnist(flatten=True, val_size=10_000):
    if not PATH.exists():
        download_mnist()
    with np.load(PATH) as f:
        X_train, y_train = f["x_train"], f["y_train"].astype(int)
        X_test, y_test = f["x_test"], f["y_test"].astype(int)
    # scale pixels from 0..255 to 0..1 and (optionally) flatten 28x28 -> 784
    prep = lambda X: (X.reshape(len(X), -1) if flatten else X).astype(np.float32) / 255.0
    X_train, X_test = prep(X_train), prep(X_test)
    # hold out the last val_size training images for validation
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def download_mnist():
    """Fetch mnist.npz (11 MB) into a temporary file, verify its SHA-256, then move
    it to PATH. Tries each URL in turn; a failed or corrupt download is deleted."""
    PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PATH.with_suffix(".part")
    for url in URLS:
        try:
            print(f"downloading MNIST (11 MB) from {url} ...")
            urllib.request.urlretrieve(url, tmp)
            digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
            if digest == SHA256:
                tmp.replace(PATH)
                return
            print(f"checksum mismatch ({digest[:12]}...), trying the next mirror")
        except OSError as err:                       # network error, HTTP error, ...
            print(f"download failed: {err}")
        finally:
            tmp.unlink(missing_ok=True)
    raise RuntimeError("Could not download MNIST. Download mnist.npz manually from "
                       + URLS[0] + " and place it at " + str(PATH))


if __name__ == "__main__":
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_mnist()
    print("train", X_tr.shape, "validation", X_va.shape, "test", X_te.shape)
    print("pixel range:", round(float(X_tr.min()), 2), "to", round(float(X_tr.max()), 2),
          "| mean pixel:", round(float(X_tr.mean()), 3))
    print("training images per digit:", np.bincount(y_tr))
