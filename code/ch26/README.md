# Chapter 26 · Profiling and Faster Training

*Part VI: Maximum Performance* · [All chapters](../../README.md#chapters)

Run every command from this folder, inside your activated environment.

| Listing | File | What it does | Expected output |
|---|---|---|---|
| 26.1 | [`wtlkit/perf.py`](../wtlkit/perf.py) | Measuring speed and size (`wtlkit/perf.py`). | printed in the book |
| 26.2 | [`profile_step.py`](profile_step.py) | Profiling a training step. | [`profile_step.log`](profile_step.log) |
| 26.3 | [`speedups.py`](speedups.py) | Measuring the effect of each speed-up (`python speedups.py cuda` repeats it on a GPU; a setting that cannot run, e.g. `torch.compile` without a C++ compiler, is printed as `skipped`). | [`speedups.log`](speedups.log) |
| 26.4 | [`grad_accum.py`](grad_accum.py) | Gradient accumulation, checked against a real batch. | [`grad_accum.log`](grad_accum.log) |
| 26.5 | [`checkpointing.py`](checkpointing.py) | Gradient checkpointing with `torch.utils.checkpoint`. | [`checkpointing.log`](checkpointing.log) |

Found a problem? [Report it](../../../../issues/new/choose) and include the listing number.
