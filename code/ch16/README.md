# Chapter 16 · Modern CNNs and Transfer Learning

*Part IV: The Network Zoo* · [All chapters](../../README.md#chapters)

Run every command from this folder, inside your activated environment.

| Listing | File | What it does | Expected output |
|---|---|---|---|
| 16.1 | [`gradient_flow.py`](gradient_flow.py) | Gradient reaching the first layer, with and without skip connections. | printed in the book |
| 16.2 | [`resnet.py`](resnet.py) | A ResNet for 28×28 images, with an option to remove the skip connections. | printed in the book |
| 16.3 | [`depth_experiment.py`](depth_experiment.py) | Plain and residual networks at two depths. | [`depth_experiment.log`](depth_experiment.log) |
| 16.4 | [`fashion_resnet.py`](fashion_resnet.py) | Training ResNet-20 on Fashion-MNIST. | [`fashion_resnet.log`](fashion_resnet.log) |
| 16.5 | [`transfer.py`](transfer.py) | Transfer learning to new classes with 20 examples each. | [`transfer.log`](transfer.log) |

Found a problem? [Report it](../../../../issues/new/choose) and include the listing number.
