# Post-training int8 quantization with ONNX Runtime, calibrated on 512 training images.
import sys
import onnx
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from onnxruntime.quantization import (quantize_static, CalibrationDataReader, QuantFormat,
                                      QuantType, quant_pre_process)
from wtlkit import mnist_tensors

(X_tr, _), _, _ = mnist_tensors(dataset="fashion", image=True)


class Calibration(CalibrationDataReader):
    """Feeds sample inputs so the quantizer can record each activation's range."""

    def __init__(self, X, batch=64):
        self.batches = iter([{"image": X[i:i + batch].numpy()} for i in range(0, len(X), batch)])

    def get_next(self):
        return next(self.batches, None)


quant_pre_process("resnet20.onnx", "resnet20_prep.onnx")   # shape inference and graph clean-up
quantize_static("resnet20_prep.onnx", "resnet20_int8.onnx", Calibration(X_tr[:512]),
                quant_format=QuantFormat.QDQ, per_channel=True,
                weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)
fp32, int8 = onnx.load("resnet20.onnx"), onnx.load("resnet20_int8.onnx")
onnx.helper.set_model_props(int8, {p.key: p.value for p in fp32.metadata_props})   # keep the labels
onnx.save(int8, "resnet20_int8.onnx")
from collections import Counter
ops = Counter(n.op_type for n in int8.graph.node)        # what does a QDQ graph look like?
print(", ".join(f"{k} x{v}" for k, v in ops.most_common(4)))
for f in ["resnet20.onnx", "resnet20_int8.onnx"]:
    print(f"{f:20s} {Path(f).stat().st_size / 1e6:.2f} MB")
