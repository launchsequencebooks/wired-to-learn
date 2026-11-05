# Magnitude pruning: remove the smallest weights, then (optionally) fine-tune.
import sys, json, copy
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path += [str(root), str(root / "ch16")]
import torch
from torch.nn.utils import prune
from resnet import ResNet
from fashion_resnet import flip_and_shift
from wtlkit import make_loaders, mnist_tensors, accuracy, set_seed, train_one_epoch, benchmark

_, _, (X_te, y_te) = mnist_tensors(dataset="fashion", image=True)
trained = ResNet(n=3)
trained.load_state_dict(torch.load(root / "ch16" / "fashion_resnet20.pt"))


def pruned_copy(amount):
    model = copy.deepcopy(trained)
    layers = [(m, "weight") for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    # global pruning: one threshold for the whole network, so layers lose different fractions
    prune.global_unstructured(layers, pruning_method=prune.L1Unstructured, amount=amount)
    return model, layers


results = {"no fine-tuning": {}, "fine-tuned 1 epoch": {}}
for amount in [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
    model, _ = pruned_copy(amount)
    results["no fine-tuning"][amount] = accuracy(model, X_te, y_te)
    print(f"{amount:4.0%} of weights removed   test accuracy {results['no fine-tuning'][amount]:.2%}")

set_seed(0)
train, _, _ = make_loaders(batch_size=128, dataset="fashion", image=True)
for amount in [0.7, 0.9, 0.95]:
    model, layers = pruned_copy(amount)                    # the masks keep pruned weights at zero
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    train_one_epoch(model, train, opt, torch.nn.CrossEntropyLoss(), augment=flip_and_shift)
    results["fine-tuned 1 epoch"][amount] = accuracy(model, X_te, y_te)
    for m, name in layers:
        prune.remove(m, name)                              # make the zeros permanent
    zeros = sum((m.weight == 0).sum().item() for m, _ in layers) / sum(m.weight.numel() for m, _ in layers)
    print(f"{amount:4.0%} removed, fine-tuned   test accuracy {results['fine-tuned 1 epoch'][amount]:.2%}"
          f"   (weights that are zero: {zeros:.1%})")
json.dump(results, open("prune.json", "w"), indent=1)

# zeros in a dense tensor are still multiplied: no speed-up without sparse kernels
X = X_te[:256]
dense_ms = benchmark(lambda: trained.eval()(X))
sparse_ms = benchmark(lambda: model.eval()(X))
print(f"inference, 256 images: dense {dense_ms:.0f} ms, 95%-pruned {sparse_ms:.0f} ms")
