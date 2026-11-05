# PyTorch (eager, compiled) vs ONNX Runtime (fp32, int8): accuracy, latency, throughput.
import sys, json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path += [str(root), str(root / "ch16")]
import numpy as np
import torch
import onnxruntime as ort
from resnet import ResNet
from wtlkit import mnist_tensors, benchmark

_, _, (X_te, y_te) = mnist_tensors(dataset="fashion", image=True)
model = ResNet(n=3)
model.load_state_dict(torch.load(root / "ch16" / "fashion_resnet20.pt"))
model.eval()


def ort_runner(path):
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return lambda X: session.run(None, {"image": X.numpy()})[0]


@torch.no_grad()
def torch_runner(X):
    return model(X).numpy()


engines = {"PyTorch (eager)": torch_runner,
           "ONNX Runtime fp32": ort_runner("resnet20.onnx"),
           "ONNX Runtime int8": ort_runner("resnet20_int8.onnx")}
try:                                           # the fair PyTorch baseline (Chapter 26)
    compiled = torch.compile(model)

    @torch.no_grad()
    def compiled_runner(X):
        return compiled(X).numpy()

    compiled_runner(X_te[:1])                      # compile now, so failures show here
    engines["PyTorch (compiled)"] = compiled_runner
except Exception as e:                         # no C++ compiler, unsupported platform
    print(f"torch.compile unavailable ({type(e).__name__}); skipping the compiled row")

curves = {}
for name, run in engines.items():
    preds = np.concatenate([run(X_te[i:i + 500]).argmax(1) for i in range(0, len(X_te), 500)])
    acc = (preds == y_te.numpy()).mean()
    curves[name] = {b: b / benchmark(lambda: run(X_te[:b]),  # one measurement per batch
                                     warmup=10 if b == 1 else 3,  # size (Fig. 28.2)
                                     repeat=200 if b == 1 else 20) * 1000
                    for b in [1, 8, 64, 256]}
    one = 1000 / curves[name][1]               # single-image latency from the same run
    print(f"{name:18s} test accuracy {acc:.2%}   1 image {one:5.2f} ms   "
          f"batch 256 {curves[name][256]:6.0f} img/s")
json.dump(curves, open("throughput.json", "w"), indent=1)
