# Chapter 27 · Quantization, Pruning and Distillation

*Part VI: Maximum Performance* · [All chapters](../../README.md#chapters)

Run every command from this folder, inside your activated environment.

| Listing | File | What it does | Expected output |
|---|---|---|---|
| 27.1 | [`quantize.py`](quantize.py) | Affine quantization from scratch. | [`quantize.log`](quantize.log) |
| 27.2 | [`weight_quant.py`](weight_quant.py) | Quantizing ResNet-20's weights to fewer and fewer bits. | [`weight_quant.log`](weight_quant.log) |
| 27.3 | [`prune.py`](prune.py) | Global magnitude pruning, with and without fine-tuning. | [`prune.log`](prune.log) |
| 27.4 | [`distill.py`](distill.py) | Knowledge distillation from ResNet-20 into a small CNN. | [`distill.log`](distill.log) |
| 27.5 | [`distill_unlabeled.py`](distill_unlabeled.py) | Distillation with few labels and many unlabeled images. | [`distill_unlabeled.log`](distill_unlabeled.log) |

Found a problem? [Report it](../../../../issues/new/choose) and include the listing number.
