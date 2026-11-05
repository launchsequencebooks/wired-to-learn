# Chapter 11 · Optimizers: From Momentum to Adam

*Part III: Training Like a Pro* · [All chapters](../../README.md#chapters)

Run every command from this folder, inside your activated environment.

| Listing | File | What it does | Expected output |
|---|---|---|---|
| 11.1 | [`optimizer_paths.py`](optimizer_paths.py) | Six optimizers from scratch, racing on a narrow valley. | printed in the book |
| 11.2 | [`my_adam.py`](my_adam.py) | Adam and AdamW from scratch, checked against PyTorch. | printed in the book |
| 11.3 | [`schedules.py`](schedules.py) | Learning-rate schedules in PyTorch. | one line per schedule with the learning rate at steps 0, 100, 500 and 999 (plotted as Figure 11.2) |
| 11.4 | [`compare_optimizers.py`](compare_optimizers.py) | Comparing seven optimizers on MNIST over three seeds (0, 1, 2); prints mean ± half-range of the final validation accuracy. | printed in the book |

Found a problem? [Report it](../../../../issues/new/choose) and include the listing number.
