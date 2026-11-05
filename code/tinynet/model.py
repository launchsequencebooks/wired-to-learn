"""TinyNet's Sequential model: stack layers, train with fit(), predict."""
import time
import numpy as np
from .data import batches


class Sequential:
    def __init__(self, *layers):
        self.layers = list(layers)

    # ---------------------------------------------------------------- core
    def forward(self, X, training=True):
        for layer in self.layers:
            X = layer.forward(X, training)
        return X

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    @property
    def params(self):
        return [p for layer in self.layers for p in layer.params]

    @property
    def grads(self):
        return [g for layer in self.layers for g in layer.grads]

    # ---------------------------------------------------------- training
    def fit(self, X, y, loss, optimizer, epochs=10, batch_size=32,
            val_data=None, verbose=1, seed=0):
        rng = np.random.default_rng(seed)
        history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
        classify = np.issubdtype(np.asarray(y).dtype, np.integer)   # acc needs labels
        for epoch in range(1, epochs + 1):
            t0, total, n = time.perf_counter(), 0.0, 0
            for Xb, yb in batches(X, y, batch_size, rng=rng):
                out = self.forward(Xb, training=True)
                total += loss.forward(out, yb) * len(Xb)
                n += len(Xb)
                self.backward(loss.backward())
                optimizer.step()
            history["loss"].append(total / n)
            history["acc"].append(self.accuracy(X, y) if classify else float("nan"))
            if val_data is not None:
                vl, va = self.evaluate(*val_data, loss)
                history["val_loss"].append(vl)
                history["val_acc"].append(va)
            if verbose and (epoch % verbose == 0 or epoch == epochs):
                msg = f"epoch {epoch:3d}  loss {history['loss'][-1]:.4f}"
                if classify:
                    msg += f"  acc {history['acc'][-1]:.2%}"
                if val_data is not None:
                    msg += f"  val_loss {vl:.4f}"
                    if classify:
                        msg += f"  val_acc {va:.2%}"
                print(msg + f"  ({time.perf_counter() - t0:.1f}s)")
        return history

    # -------------------------------------------------------- inference
    def predict_logits(self, X, batch_size=1024):
        return np.vstack([self.forward(X[i:i + batch_size], training=False)
                          for i in range(0, len(X), batch_size)])

    def predict_proba(self, X):
        Z = self.predict_logits(X)
        if Z.shape[1] == 1:                       # one logit: sigmoid
            return 1.0 / (1.0 + np.exp(-Z))
        E = np.exp(Z - Z.max(axis=1, keepdims=True))
        return E / E.sum(axis=1, keepdims=True)   # several logits: softmax

    def predict(self, X):
        Z = self.predict_logits(X)
        if Z.shape[1] == 1:                       # one logit: class 1 iff logit > 0
            return (Z[:, 0] > 0).astype(int)
        return Z.argmax(axis=1)

    def accuracy(self, X, y):
        return float((self.predict(X) == np.asarray(y).reshape(-1)).mean())

    def evaluate(self, X, y, loss):
        classify = np.issubdtype(np.asarray(y).dtype, np.integer)
        acc = self.accuracy(X, y) if classify else float("nan")
        return float(loss.forward(self.predict_logits(X), y)), acc

    # ------------------------------------------------------------ utils
    def summary(self):
        print(f"{'layer':<22}{'output':>10}{'params':>10}")
        total, width = 0, None
        for layer in self.layers:
            n = sum(p.size for p in layer.params)
            if hasattr(layer, "W"):
                width = layer.W.shape[1]
            total += n
            print(f"{type(layer).__name__:<22}{width if width else '':>10}{n:>10,}")
        print(f"{'total parameters':<32}{total:>10,}")

    def save(self, path):
        np.savez(path, *self.params)

    def load(self, path):
        with np.load(path) as f:
            saved = [f[key] for key in f.files]
        if (len(saved) != len(self.params)
                or any(s.shape != p.shape for s, p in zip(saved, self.params))):
            raise ValueError(f"{path}: checkpoint shapes {[s.shape for s in saved]} "
                             f"do not match model {[p.shape for p in self.params]}")
        for p, s in zip(self.params, saved):
            p[...] = s
        return self
