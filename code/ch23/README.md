# Chapter 23 · Self-Supervised and Contrastive Learning

*Part V: Learning Beyond Labels* · [All chapters](../../README.md#chapters)

Run every command from this folder, inside your activated environment.

| Listing | File | What it does | Expected output |
|---|---|---|---|
| 23.1 | [`ntxent.py`](ntxent.py) | The NT-Xent contrastive loss. | printed in the book |
| 23.2 | [`simclr.py`](simclr.py) | SimCLR pretraining and linear evaluation on Fashion-MNIST. | [`simclr.log`](simclr.log) |
| 23.3 | [`supervised_baseline.py`](supervised_baseline.py) | Supervised training on the labeled images only. | [`supervised_baseline.log`](supervised_baseline.log) |
| Fig. 23.2 | [`show_views.py`](show_views.py) | Eight training images and two random views of each (imports `augment` from `simclr.py`; no training needed). | `ch23_views.png` |

Found a problem? [Report it](../../../../issues/new/choose) and include the listing number.
