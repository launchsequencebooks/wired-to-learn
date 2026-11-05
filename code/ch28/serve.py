# A prediction service for the clothing classifier (run: uvicorn serve:app --port 8000).
import io
import json
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile

MODEL = Path(__file__).with_name("resnet20_int8.onnx")
VERSION = "resnet20-int8-v1"

app = FastAPI(title="Wired to Learn clothing classifier")
opts = ort.SessionOptions()
opts.intra_op_num_threads = 1                          # one core per worker; see 28.4.2
session = ort.InferenceSession(str(MODEL), opts,         # load once, at start-up
                               providers=["CPUExecutionProvider"])
CLASSES = json.loads(session.get_modelmeta().custom_metadata_map["classes"])     # labels from the model file


def preprocess(data: bytes) -> np.ndarray:
    """Any image file -> a (1, 1, 28, 28) float32 array, exactly as in training."""
    img = Image.open(io.BytesIO(data)).convert("L").resize((28, 28))
    return (np.asarray(img, dtype=np.float32) / 255.0)[None, None]


@app.get("/health")
def health():
    return {"status": "ok", "model": VERSION}


@app.post("/predict")
def predict(file: UploadFile = File(...)):   # plain def: FastAPI runs it in a thread
    try:                                     # pool, so the model never blocks the event
        x = preprocess(file.file.read())     # loop (ONNX Runtime is thread-safe)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="the upload is not an image")
    t0 = time.perf_counter()
    logits = session.run(None, {"image": x})[0][0]
    p = np.exp(logits - logits.max()); p /= p.sum()          # softmax
    top = np.argsort(p)[::-1][:3]
    return {"label": CLASSES[top[0]], "confidence": round(float(p[top[0]]), 4),
            "top3": [{"label": CLASSES[i], "p": round(float(p[i]), 4)} for i in top],
            "model": VERSION, "inference_ms": round((time.perf_counter() - t0) * 1000, 2)}
