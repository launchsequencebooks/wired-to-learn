# Training ResNet-20 on all of Fashion-MNIST.
import sys, json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import torch
from wtlkit import make_loaders, set_seed, fit, evaluate, count_params
from resnet import ResNet


def flip_and_shift(X):
    """Augmentation for clothes: random horizontal flips and small shifts."""
    flip = torch.rand(len(X)) < 0.5
    X = torch.where(flip[:, None, None, None], X.flip(3), X)
    X = torch.nn.functional.pad(X, (2, 2, 2, 2))
    i, j = torch.randint(0, 5, (2,)).tolist()
    return X[:, :, i:i + 28, j:j + 28]


if __name__ == "__main__":
    set_seed(0)
    tr, va, te = make_loaders(batch_size=128, dataset="fashion", image=True)
    model = ResNet(n=3)
    print(f"ResNet-20 parameters: {count_params(model):,}")
    EPOCHS = 8
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, epochs=EPOCHS, steps_per_epoch=len(tr))
    hist = fit(model, tr, va, opt, epochs=EPOCHS, scheduler=sched, augment=flip_and_shift, verbose=2)
    print(f"test accuracy: {evaluate(model, te)[1]:.2%}")
    torch.save(model.state_dict(), "fashion_resnet20.pt")
    json.dump(hist, open("fashion_history.json", "w"))
