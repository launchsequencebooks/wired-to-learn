# Exercise the service in-process: correctness, error handling and latency.
import io, sys, time
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from serve import app
from wtlkit import mnist_tensors, FASHION_CLASSES

client = TestClient(app)
print("GET /health ->", client.get("/health").json())


def png(x):                                                # a test image as a PNG file, like a real upload
    buf = io.BytesIO()
    Image.fromarray((x.squeeze().numpy() * 255).round().astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


_, _, (X_te, y_te) = mnist_tensors(dataset="fashion", image=True)
r = client.post("/predict", files={"file": ("item.png", png(X_te[0]), "image/png")}).json()
print(f"true label {FASHION_CLASSES[y_te[0]]!r} ->", {k: r[k] for k in ["label", "confidence", "top3"]})

bad = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
print("not an image ->", bad.status_code, bad.json())

latencies, correct = [], 0
for i in range(500):
    t0 = time.perf_counter()
    r = client.post("/predict", files={"file": ("item.png", png(X_te[i]), "image/png")}).json()
    latencies.append((time.perf_counter() - t0) * 1000)
    correct += r["label"] == FASHION_CLASSES[y_te[i]]
p50, p95, p99 = np.percentile(latencies, [50, 95, 99])
print(f"500 requests: accuracy {correct / 500:.1%}, latency p50 {p50:.1f} ms, p95 {p95:.1f} ms, p99 {p99:.1f} ms")
