# A convolutional digit reader: can it beat the 99.05% of Chapter 14?
import sys, json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root)); sys.path.insert(0, str(root / "ch13"))
import torch
from torch import nn
from wtlkit import make_loaders, set_seed, fit, evaluate, count_params
from augment import random_affine


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def block(c_in, c_out):                          # conv -> BN -> ReLU, twice, then pool
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(),
                nn.MaxPool2d(2))
        self.features = nn.Sequential(block(1, 32), block(32, 64))   # 28x28 -> 14x14 -> 7x7
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                                        nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
                                        nn.Dropout(0.3), nn.Linear(128, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


def augment_images(X):                                   # reuse Chapter 13's augmentation
    return random_affine(X.view(len(X), -1)).view_as(X)


if __name__ == "__main__":
    set_seed(0)
    train_loader, val_loader, test_loader = make_loaders(batch_size=128, image=True)
    model = DigitCNN()
    print(f"parameters: {count_params(model):,}")
    EPOCHS = 10
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, epochs=EPOCHS,
                                                steps_per_epoch=len(train_loader))
    hist = fit(model, train_loader, val_loader, opt, epochs=EPOCHS, scheduler=sched,
               augment=augment_images, verbose=2)
    test_loss, test_acc = evaluate(model, test_loader)
    print(f"test accuracy: {test_acc:.2%}  (test error {1 - test_acc:.2%}, "
          f"{round((1 - test_acc) * 10000)} of 10,000 digits wrong)")
    torch.save(model.state_dict(), "digit_cnn.pt")
    json.dump(hist, open("cnn_history.json", "w"))
