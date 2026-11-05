# Where does the time go in one training step of ResNet-20?
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path += [str(root), str(root / "ch16")]
import torch
from torch.profiler import profile, ProfilerActivity
from resnet import ResNet

torch.manual_seed(0)
model = ResNet(n=3)
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
X, y = torch.randn(128, 1, 28, 28), torch.randint(0, 10, (128,))


def step():
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(X), y)
    loss.backward()
    opt.step()


for _ in range(3):                                         # warm up first
    step()
with profile(activities=[ProfilerActivity.CPU]) as prof:
    for _ in range(5):
        step()

# group the low-level operations into a few families we can reason about
families = {"convolution (forward)": ["aten::mkldnn_convolution", "aten::thnn_conv2d", "aten::_slow_conv2d_forward"],
            "convolution (backward)": ["convolution_backward"],
            "batch norm": ["batch_norm"],
            "ReLU": ["relu", "clamp_min", "threshold_backward"],
            "optimizer (SGD)": ["_foreach", "aten::add_", "aten::mul_"]}
totals = dict.fromkeys(families, 0.0)
everything = 0.0
for e in prof.key_averages():
    everything += e.self_cpu_time_total
    for fam, keys in families.items():
        if any(k in e.key for k in keys):
            totals[fam] += e.self_cpu_time_total
            break
step_ms = everything / 5 / 1000
print(f"one training step (batch 128): {step_ms:.0f} ms")
for fam, t in totals.items():
    print(f"  {fam:24s} {100 * t / everything:5.1f}% of time")
print(f"  {'everything else':24s} {100 * (everything - sum(totals.values())) / everything:5.1f}% of time")
