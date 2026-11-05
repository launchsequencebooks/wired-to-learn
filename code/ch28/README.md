# Chapter 28 · ONNX, Inference Engines and Serving

*Part VI: Maximum Performance* · [All chapters](../../README.md#chapters)

Run every command from this folder, inside your activated environment.

| Listing | File | What it does | Expected output |
|---|---|---|---|
| 28.1 | [`export_onnx.py`](export_onnx.py) | Exporting ResNet-20 to ONNX and checking it. | [`export_onnx.log`](export_onnx.log) |
| 28.2 | [`quantize_onnx.py`](quantize_onnx.py) | Static int8 quantization with ONNX Runtime. | [`quantize_onnx.log`](quantize_onnx.log) |
| 28.3 | [`ort_benchmark.py`](ort_benchmark.py) | Comparing PyTorch and ONNX Runtime. | [`ort_benchmark.log`](ort_benchmark.log) |
| 28.4 | [`serve.py`](serve.py) | A FastAPI prediction service. | printed in the book |
| 28.5 | [`test_service.py`](test_service.py) | Testing the service end to end. | [`test_service.log`](test_service.log) |
| 28.6 | [`Dockerfile`](Dockerfile) | A container image for the service. | printed in the book |

**Supporting files:** [`requirements.txt`](requirements.txt)

Found a problem? [Report it](../../../../issues/new/choose) and include the listing number.
