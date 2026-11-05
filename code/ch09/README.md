# Chapter 9 · Project: Reading Handwritten Digits

*Part II: Building a Brain from Scratch* · [All chapters](../../README.md#chapters)

Run every command from this folder, inside your activated environment.

| Listing | File | What it does | Expected output |
|---|---|---|---|
| 9.1 | [`mnist_loader.py`](mnist_loader.py) | Loading MNIST. | printed in the book |
| 9.2 | [`softmax_regression.py`](softmax_regression.py) | A one-layer baseline. | printed in the book |
| 9.3 | [`mlp_digits.py`](mlp_digits.py) | The 784–256–128–10 digit reader. | printed in the book |
| 9.4 | [`errors.py`](errors.py) | Confusion matrix and the most confident mistakes (run `mlp_digits.py` first). | printed in the book |
| 9.5 | [`experiments.py`](experiments.py) | Controlled experiments on the digit reader. | printed in the book |
| 9.6 | [`predict_digit.py`](predict_digit.py) | Reading digits from image files. | printed in the book |

`mnist_loader.py` downloads `mnist.npz` into `code/data/` on first use and checks its SHA-256; if no mirror is reachable it tells you where to put a manual download. The four `my_*.png` test images are rendered, not photographed; real photos are Exercise 9.6.

Found a problem? [Report it](../../../../issues/new/choose) and include the listing number.
