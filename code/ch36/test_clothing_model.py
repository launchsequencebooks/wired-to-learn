# Tests for the clothing classifier of Chapter 28 (run: pytest -q test_clothing_model.py).
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import numpy as np
import pytest
from wtlkit import mnist_tensors, FASHION_CLASSES
from wtlapps import OnnxModel

MODEL = root / "ch28" / "resnet20_int8.onnx"


@pytest.fixture(scope="module")
def model():
    return OnnxModel(MODEL)


@pytest.fixture(scope="module")
def test_data():
    _, _, (X, y) = mnist_tensors(dataset="fashion", image=True)
    return X.numpy(), y.numpy()


def test_labels_match_training_data(model):             # the bug of Chapter 28 can never come back
    assert model.labels == FASHION_CLASSES


def test_output_shape_and_probabilities(model, test_data):
    p = model.predict_proba(test_data[0][:8])
    assert p.shape == (8, 10) and np.allclose(p.sum(1), 1, atol=1e-5)


def test_accuracy_does_not_regress(model, test_data):   # a quality gate: never ship a worse model
    X, y = test_data
    assert (model(X).argmax(1) == y).mean() >= 0.92


def test_batch_and_single_agree(model, test_data):      # batching must not change answers
    X = test_data[0][:16]
    single = np.concatenate([model(X[i:i + 1]) for i in range(16)])
    assert np.array_equal(single.argmax(1), model(X).argmax(1))


def test_latency_budget(model, test_data):              # a slower model must not ship unnoticed
    import time
    X = test_data[0][:256]
    model(X)                                            # warm up
    t0 = time.perf_counter(); model(X); per_item = (time.perf_counter() - t0) / 256
    assert per_item < 0.002, f"{per_item * 1e3:.2f} ms per image (budget 2 ms)"


def test_small_shift_invariance(model, test_data):      # a garment moved by one pixel is the same garment
    X = test_data[0][:1000]
    same = model(X).argmax(1) == model(np.roll(X, 1, axis=3)).argmax(1)
    assert same.mean() >= 0.95


@pytest.mark.parametrize("cls", range(10))
def test_every_class_is_learned(model, test_data, cls): # slices: no class may be left behind
    X, y = test_data
    acc = (model(X[y == cls]).argmax(1) == cls).mean()
    assert acc >= 0.75, f"{FASHION_CLASSES[cls]}: {acc:.1%}"


@pytest.mark.xfail(strict=True, reason="known limitation, documented in the model card: "
                   "screened by the organizer's outlier detector (Chapter 30)")
def test_blank_image_is_not_confident(model):          # minimum functionality: no garment, no confidence
    p = model.predict_proba(np.zeros((1, 1, 28, 28), np.float32))[0]
    assert p.max() < 0.9, f"blank -> {model.labels[p.argmax()]} ({p.max():.0%})"
