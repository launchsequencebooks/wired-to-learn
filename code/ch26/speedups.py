# Measuring training throughput of ResNet-20 under different settings.
import sys, json, copy
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path += [str(root), str(root / "ch16")]
import torch
from resnet import ResNet
from wtlkit import benchmark

torch.manual_seed(0)
base = ResNet(n=3)
device = sys.argv[1] if len(sys.argv) > 1 else "cpu"   # "cuda" runs it on a GPU
# 16-bit format for autocast: bfloat16 where the hardware has it, float16 on older GPUs
half = torch.bfloat16
if device == "cuda" and not torch.cuda.is_bf16_supported():
    half = torch.float16                                   # e.g. a Colab T4


def throughput(batch=128, channels_last=False, bf16=False, compiled=False, threads=2):
    torch.set_num_threads(threads)
    model = copy.deepcopy(base).to(device)
    X = torch.randn(batch, 1, 28, 28, device=device)
    y = torch.randint(0, 10, (batch,), device=device)
    if channels_last:                                      # store images as N,H,W,C
        model = model.to(memory_format=torch.channels_last)
        X = X.to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    fwd = torch.compile(model) if compiled else model

    def step():
        opt.zero_grad()
        with torch.autocast(device, dtype=half, enabled=bf16):
            loss = torch.nn.functional.cross_entropy(fwd(X), y)
        loss.backward()
        opt.step()
        if device == "cuda":
            torch.cuda.synchronize()               # GPU work is asynchronous: wait
    ms = benchmark(step, warmup=3, repeat=10)
    return batch / ms * 1000                       # images per second


settings = {"baseline (batch 128)": {},
            "batch 32": {"batch": 32},
            "batch 512": {"batch": 512},
            "1 thread instead of 2": {"threads": 1},
            "channels_last": {"channels_last": True},
            "bfloat16 autocast": {"bf16": True},
            "torch.compile": {"compiled": True},
            "compile + channels_last": {"compiled": True, "channels_last": True}}
results = {}
for name, kw in settings.items():
    try:
        results[name] = throughput(**kw)
    except Exception as e:            # torch.compile needs a C compiler; not on Windows
        print(f"{name:24s} skipped ({type(e).__name__})"); continue
    ratio = results[name] / results["baseline (batch 128)"]
    print(f"{name:24s} {results[name]:6.0f} img/s   {ratio:.2f}x")
json.dump(results, open("speedups.json", "w"), indent=1)   # only the settings that ran
