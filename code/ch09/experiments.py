# Small experiments: how do width, depth, momentum, learning rate and
# batch size change the result? Each run trains for 5 epochs.
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import tinynet as tn
from mnist_loader import load_mnist

(X_tr, y_tr), (X_va, y_va), _ = load_mnist()


def run(hidden=(256, 128), lr=0.05, momentum=0.9, batch_size=64, epochs=5, seed=0):
    rng = np.random.default_rng(seed)
    layers, n_in = [], 784
    for h in hidden:
        layers += [tn.Dense(n_in, h, rng=rng), tn.ReLU()]
        n_in = h
    model = tn.Sequential(*layers, tn.Dense(n_in, 10, rng=rng))
    opt = tn.SGD(model.params, model.grads, lr=lr, momentum=momentum)
    hist = model.fit(X_tr, y_tr, tn.SoftmaxCrossEntropy(), opt, epochs=epochs,
                     batch_size=batch_size, val_data=(X_va, y_va), verbose=0,
                     seed=seed)
    n_params = sum(p.size for p in model.params)
    return hist["val_acc"][-1], hist["val_loss"][-1], n_params


experiments = [
    ("1 hidden layer, 32 units",   dict(hidden=(32,))),
    ("1 hidden layer, 128 units",  dict(hidden=(128,))),
    ("1 hidden layer, 512 units",  dict(hidden=(512,))),
    ("2 hidden layers, 256-128",   dict()),
    ("  ... without momentum",     dict(momentum=0.0)),
    ("  ... without momentum, lr 0.5", dict(momentum=0.0, lr=0.5)),
    ("  ... learning rate 0.5",    dict(lr=0.5)),
    ("  ... batch size 16",        dict(batch_size=16)),
    ("  ... batch size 512",       dict(batch_size=512)),
]
results = []
print(f"{'configuration':<32}{'params':>9}{'val acc':>9}{'val loss':>10}")
for name, kw in experiments:
    acc, loss, n = run(**kw)
    results.append({"name": name.strip(" ."), "acc": acc, "loss": loss, "params": int(n)})
    print(f"{name:<32}{n:>9,}{acc:>9.2%}{loss:>10.4f}")
# the noise floor: the same baseline with three different seeds
accs = np.array([run(seed=s)[0] for s in (0, 1, 2)])
print(f"baseline over 3 seeds: val acc {accs.mean():.2%} +/- {accs.std():.2%}")
results.append({"name": "baseline, 3 seeds", "acc": accs.tolist()})
json.dump(results, open("experiments.json", "w"), indent=1)
