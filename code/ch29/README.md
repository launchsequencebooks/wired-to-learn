# Chapter 29 · Neural Networks as Application Components

*Part VII: Building Intelligent Applications* · [All chapters](../../README.md#chapters)

Run every command from this folder, inside your activated environment.

## Before you start

Chapter 29 trains nothing. Listing 29.4 (`routing_demo.py`) loads the quantized
classifier of Chapter 28, which is committed:

| File | Produced by | Time |
|---|---|---|
| `ch28/resnet20_int8.onnx` | Listing 28.2, `quantize_onnx.py` (from `ch16/fashion_resnet20.pt`) | committed; rebuilding it needs the ResNet-20 of Chapter 16 (about 10 min on a 2-thread laptop CPU, or `python scripts/fetch_artifacts.py ch28`) |

`routing_demo.py` also writes `router_99.json`, the confidence threshold for the 99%
target, which Chapter 30's organizer reads. A copy is committed so that Chapter 30 runs
without Chapter 29; re-running `routing_demo.py` refreshes it.

| Listing | File | What it does | Expected output |
|---|---|---|---|
| 29.1 | [`wtlapps/model.py`](../wtlapps/model.py) | A trained network as an application component. | printed in the book |
| 29.2 | [`wtlapps/index.py`](../wtlapps/index.py) | Similarity search over embeddings. | printed in the book |
| 29.3 | [`wtlapps/routing.py`](../wtlapps/routing.py) | Routing predictions between the model and a person. | printed in the book |
| 29.4 | [`routing_demo.py`](routing_demo.py) | How much can the classifier handle on its own? | [`routing_demo.log`](routing_demo.log) |

**Supporting files:** [`router_99.json`](router_99.json) (written by `routing_demo.py`, read by `../ch30/organizer.py`)

Found a problem? [Report it](../../../../issues/new/choose) and include the listing number.
