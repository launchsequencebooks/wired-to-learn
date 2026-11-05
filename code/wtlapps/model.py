"""A trained network as an application component: load once, predict, describe itself."""
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort


class OnnxModel:
    """Wraps an ONNX file. Labels and other metadata travel inside the file (Chapter 28)."""

    def __init__(self, path, threads=None):
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        meta = self.session.get_modelmeta().custom_metadata_map
        self.labels = json.loads(meta["classes"]) if "classes" in meta else None
        self.meta = meta
        self.name = Path(path).stem

    def __call__(self, x, batch_size=512):
        """Raw outputs (logits or embeddings) for a float32 array, in batches."""
        x = np.asarray(x, dtype=np.float32)
        return np.concatenate([self.session.run(None, {self.input: x[i:i + batch_size]})[0]
                               for i in range(0, len(x), batch_size)])

    def predict_proba(self, x):
        z = self(x)
        z = np.exp(z - z.max(1, keepdims=True))
        return z / z.sum(1, keepdims=True)
