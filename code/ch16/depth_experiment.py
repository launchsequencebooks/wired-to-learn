# Plain vs residual networks at two depths on Fashion-MNIST (20,000 training images),
# averaged over three seeds so that a difference can be told from run-to-run noise.
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from wtlkit import make_loaders, set_seed, fit
from resnet import ResNet

SEEDS = [0, 1, 2]
histories = {}                                   # seed-0 histories, for Figure 16.2
print(f"final epoch, mean +/- std over {len(SEEDS)} seeds (time per seed)")
print(f"{'network':12s}{'train loss':>21s}{'val acc':>20s}{'time':>8s}")
for n in [1, 3]:
    for residual in [False, True]:
        name = f"{'ResNet' if residual else 'Plain'}-{6 * n + 2}"
        losses, accs, t0 = [], [], time.perf_counter()
        for seed in SEEDS:
            set_seed(seed)
            tr, va, _ = make_loaders(batch_size=128, n_train=20000, dataset="fashion",
                                     image=True)
            model = ResNet(n, residual=residual)
            opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9,
                                  weight_decay=5e-4)
            hist = fit(model, tr, va, opt, epochs=4, verbose=0)
            losses.append(hist["loss"][-1]); accs.append(hist["val_acc"][-1])
            if seed == 0:
                histories[name] = hist
        print(f"{name:12s}{np.mean(losses):>10.4f} +/- {np.std(losses):.4f}"
              f"{np.mean(accs):>10.2%} +/- {np.std(accs):.2%}"
              f"{(time.perf_counter() - t0) / len(SEEDS):>7.0f}s")
json.dump(histories, open("depth_histories.json", "w"))
