# Chapter 10 · From TinyNet to PyTorch

*Part III: Training Like a Pro* · [All chapters](../../README.md#chapters)

Run every command from this folder, inside your activated environment.

| Listing | File | What it does | Expected output |
|---|---|---|---|
| 10.1 | [`tensors.py`](tensors.py) | Tensor basics and NumPy interoperability. | printed in the book |
| 10.2 | [`autograd_demo.py`](autograd_demo.py) | Autograd on Chapter 7's tiny network. | printed in the book |
| 10.3 | [`grad_accumulation.py`](grad_accumulation.py) | What happens without zero_grad(). | printed in the book |
| 10.4 | [`modules.py`](modules.py) | Defining networks with nn.Module and nn.Sequential. | printed in the book |
| 10.5 | [`mnist_pytorch.py`](mnist_pytorch.py) | The digit reader in PyTorch. | printed in the book |
| 10.6 | [`tinynet_vs_torch.py`](tinynet_vs_torch.py) | Comparing TinyNet with PyTorch, output by output and gradient by gradient. Needs `../ch09/digit_mlp.npz`, written by `python mlp_digits.py` in `code/ch09` (about 3 minutes); run that first if the file is missing. | printed in the book |
| 10.7 | [`wtlkit/train.py`](../wtlkit/train.py) | The wtlkit training helpers. | printed in the book |

Found a problem? [Report it](../../../../issues/new/choose) and include the listing number.
