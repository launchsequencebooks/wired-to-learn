# Chapter 30 · Project: A Smart Photo Organizer

*Part VII: Building Intelligent Applications* · [All chapters](../../README.md#chapters)

Run every command from this folder, inside your activated environment.

## Before you start

Listing 30.1 (`build_components.py`) loads two networks trained earlier in the book,
and Listings 30.3-30.5 load the components it writes:

| File | Trained by | Training time |
|---|---|---|
| `ch16/fashion_resnet20.pt` | Listing 16.4, `ch16/fashion_resnet.py` | about 10 min on a 2-thread laptop CPU |
| `ch23/simclr_encoder.pt` | Listing 23.2, `ch23/simclr.py` | about 13 min on a 2-thread laptop CPU |
| `ch30/outlier_reference.npz` | Listing 30.1, `build_components.py` (needs the two files above) | a few minutes |
| `ch29/router_99.json` | Listing 29.4, `ch29/routing_demo.py` | seconds; a copy is committed |

The three `.onnx` components and `ch28/resnet20_int8.onnx` are committed. Either run
the chapters in order, or fetch the trained files with
`python scripts/fetch_artifacts.py ch30` from the `code` folder.

For the web interface, first write a demo folder with `python make_demo_photos.py`
(280 garments and 20 scanned digits in `photos/`), then `streamlit run app.py`.

| Listing | File | What it does | Expected output |
|---|---|---|---|
| 30.1 | [`build_components.py`](build_components.py) | Packaging the networks as components. | [`build_components.log`](build_components.log) |
| 30.2 | [`outlier_compare.py`](outlier_compare.py) | Comparing signals for ``this is not clothing''. | [`outlier_compare.log`](outlier_compare.log) |
| 30.3 | [`organizer.py`](organizer.py) | The Smart Photo Organizer. | printed in the book |
| 30.4 | [`evaluate.py`](evaluate.py) | Testing the organizer end to end. | [`evaluate.log`](evaluate.log) |
| 30.5 | [`app.py`](app.py) | The organizer's web interface. | printed in the book |

**Supporting files:** [`make_demo_photos.py`](make_demo_photos.py)

Found a problem? [Report it](../../../../issues/new/choose) and include the listing number.
