# Chapter 36 · Testing, Monitoring and MLOps

*Part VII: Building Intelligent Applications* · [All chapters](../../README.md#chapters)

Run every command from this folder, inside your activated environment.

## Before you start

The tests (36.1–36.2), the calibration (36.3), the drift monitor (36.4) and the registry
(36.5) load trained files from three earlier chapters:

| File | Trained by | Training time |
|---|---|---|
| `ch28/resnet20_int8.onnx` (and `resnet20.onnx`) | Listings 28.1–28.2, `export_onnx.py`, `quantize_onnx.py` (committed) | seconds |
| `ch16/fashion_resnet20.pt` | Listing 16.4, `fashion_resnet.py` | about 10 minutes on a 2-thread laptop CPU |
| `ch31/bag.pt`, `ch31/triage_parts.pkl`, `ch31/ticket_index.npz` | Listings 31.4 and 31.6, `train_triage.py`, `build_triage.py` | about 6 min, plus a few minutes |

If you have not run those chapters, fetch the files with
`python scripts/fetch_artifacts.py ch36` from the `code` folder.

| Listing | File | What it does | Expected output |
|---|---|---|---|
| 36.1 | [`test_clothing_model.py`](test_clothing_model.py) | Tests for the clothing classifier. | printed in the book |
| 36.2 | [`run_tests.py`](run_tests.py) | Running the test suite and reporting. | [`run_tests.log`](run_tests.log) |
| 36.3 | [`calibration.py`](calibration.py) | Measuring and fixing calibration. | [`calibration.log`](calibration.log) |
| 36.4 | [`drift.py`](drift.py) | Monitoring for data drift. | [`drift.log`](drift.log) |
| 36.5 | [`registry.py`](registry.py) | A minimal model registry with release gates and model cards. | [`registry.log`](registry.log) |
| 36.6 | [`model-ci.yml`](model-ci.yml) | A continuous-integration workflow for the model. | printed in the book |

Found a problem? [Report it](../../../../issues/new/choose) and include the listing number.
