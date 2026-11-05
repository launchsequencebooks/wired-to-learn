# A minimal model registry: every model version is recorded, checked by gates, and documented.
import hashlib, json, shutil, subprocess, sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import numpy as np
from wtlkit import mnist_tensors, FASHION_CLASSES
from wtlapps import OnnxModel

REG = Path("registry"); REG.mkdir(exist_ok=True)
INDEX = REG / "index.json"
_, _, (X_te, y_te) = mnist_tensors(dataset="fashion", image=True)
X_te, y_te = X_te.numpy(), y_te.numpy()


def evaluate(path):
    pred = OnnxModel(path)(X_te).argmax(1)
    per_class = {c: float((pred[y_te == i] == i).mean()) for i, c in enumerate(FASHION_CLASSES)}
    return {"accuracy": float((pred == y_te).mean()), "worst_class": min(per_class, key=per_class.get),
            "worst_class_accuracy": min(per_class.values()), "size_mb": round(Path(path).stat().st_size / 1e6, 2)}


def register(path, version, notes):
    """Copy the model into the registry with its hash, metrics, notes, data and code lineage."""
    index = json.loads(INDEX.read_text()) if INDEX.exists() else {"versions": {}, "production": None}
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]    # which exact file is this?
    shutil.copy(path, REG / f"{version}.onnx")
    data = root / "data" / "fashion" / "train-images-idx3-ubyte.gz"         # which data trained it?
    try:                                                                     # which code revision?
        code = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:                                                          # no git on this machine
        code = ""
    index["versions"][version] = {
        "sha256": digest, "notes": notes, "metrics": evaluate(path),
        "data": {"name": "fashion-mnist", "sha256": hashlib.sha256(data.read_bytes()).hexdigest()[:12]},
        "code": code or "unknown"}
    INDEX.write_text(json.dumps(index, indent=1))
    return index


def promote(version, max_accuracy_drop=0.002, min_class_accuracy=0.70):
    """Release gates: no meaningful loss of accuracy, and no class left behind."""
    index = json.loads(INDEX.read_text())
    cand = index["versions"][version]["metrics"]
    prod = index["versions"][index["production"]]["metrics"] if index["production"] else None
    checks = {"accuracy": prod is None or cand["accuracy"] >= prod["accuracy"] - max_accuracy_drop,
              "every class": cand["worst_class_accuracy"] >= min_class_accuracy}
    if all(checks.values()):
        index["production"] = version
        INDEX.write_text(json.dumps(index, indent=1))
    return checks


def model_card(version):
    """A short, standard description of the model for its users (Mitchell et al., 2019)."""
    v = json.loads(INDEX.read_text())["versions"][version]
    m = v["metrics"]
    return "\n".join([f"# Model card: clothing classifier {version}",
                      f"- File hash: {v['sha256']}; size {m['size_mb']} MB. {v['notes']}",
                      f"- Trained on {v['data']['name']} (data hash {v['data']['sha256']}), "
                      f"code commit {v['code']}.",
                      f"- Intended use: tagging product photos of 10 garment types (28x28 grayscale).",
                      f"- Test accuracy {m['accuracy']:.2%}; weakest class: {m['worst_class']} ({m['worst_class_accuracy']:.1%}).",
                      "- Known limitations: confident on blank and non-garment images (use an outlier detector);",
                      "  accuracy drops on darker photos than those in training (monitor input brightness)."])


if INDEX.exists():
    INDEX.unlink()                                          # start the demonstration from an empty registry
register(root / "ch28" / "resnet20.onnx", "v1", "ResNet-20, float32 (Chapter 28).")
print("v1 gates:", promote("v1"))
register(root / "ch28" / "resnet20_int8.onnx", "v2", "ResNet-20, int8 static quantization (Chapter 28).")
print("v2 gates:", promote("v2"), "-> production is", json.loads(INDEX.read_text())["production"])
print()
print(model_card("v2"))
Path("registry/model_card_v2.md").write_text(model_card("v2"))
