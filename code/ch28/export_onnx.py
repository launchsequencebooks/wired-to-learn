# Export ResNet-20 to ONNX and check that ONNX Runtime gives the same answers.
import sys, json
from collections import Counter
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path += [str(root), str(root / "ch16")]
import numpy as np
import torch
import onnx
import onnxruntime as ort
from resnet import ResNet
from wtlkit import FASHION_CLASSES

model = ResNet(n=3)
model.load_state_dict(torch.load(root / "ch16" / "fashion_resnet20.pt"))
model.eval()                                               # export the inference behaviour of BatchNorm

example = torch.randn(1, 1, 28, 28)
torch.onnx.export(model, (example,), "resnet20.onnx", input_names=["image"], output_names=["logits"],
                  dynamic_shapes={"x": {0: torch.export.Dim("batch")}},   # any batch size
                  dynamo=True, external_data=False, verbose=False)

graph = onnx.load("resnet20.onnx")
onnx.checker.check_model(graph)                            # is the file a valid ONNX model?
onnx.helper.set_model_props(graph, {"classes": json.dumps(FASHION_CLASSES),   # labels travel with the model
                                    "preprocessing": "grayscale 28x28, pixel/255"})
onnx.save(graph, "resnet20.onnx")
ops = Counter(node.op_type for node in graph.graph.node)
print(f"ONNX opset {graph.opset_import[0].version}, {len(graph.graph.node)} nodes:",
      ", ".join(f"{k} x{v}" for k, v in ops.most_common()))
print(f"file size {Path('resnet20.onnx').stat().st_size / 1e6:.2f} MB")

session = ort.InferenceSession("resnet20.onnx", providers=["CPUExecutionProvider"])
torch.manual_seed(0)
X = torch.randn(64, 1, 28, 28)
with torch.no_grad():
    expected = model(X).numpy()
got = session.run(["logits"], {"image": X.numpy()})[0]
print(f"batch of {len(X)}: largest difference from PyTorch {np.abs(got - expected).max():.1e}, "
      f"same predictions: {(got.argmax(1) == expected.argmax(1)).all()}")
