"""Measuring speed and size: the tools of Part VI."""
import io
import statistics
import time
import torch


def benchmark(fn, warmup=3, repeat=10):
    """Median wall-clock time of fn() in milliseconds. Warm-up runs are
    discarded: the first calls pay one-off costs (memory allocation,
    kernel selection, compilation) that steady-state runs do not."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def model_size_mb(model):
    """Size of the model's saved state_dict in megabytes."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


@torch.no_grad()
def accuracy(model, X, y, batch_size=1000):
    """Accuracy of model on tensors X, y (evaluated in mini-batches)."""
    model.eval()
    correct = 0
    for i in range(0, len(X), batch_size):
        correct += (model(X[i:i + batch_size]).argmax(1) == y[i:i + batch_size]).sum().item()
    return correct / len(X)
