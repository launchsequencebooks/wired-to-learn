# Batch normalization from scratch, checked against nn.BatchNorm1d,
# plus layer normalization for comparison.
import torch
from torch import nn


class MyBatchNorm(nn.Module):
    def __init__(self, n, momentum=0.1, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n))     # learnable scale
        self.beta = nn.Parameter(torch.zeros(n))     # learnable shift
        self.register_buffer("run_mean", torch.zeros(n))
        self.register_buffer("run_var", torch.ones(n))
        self.momentum, self.eps = momentum, eps

    def forward(self, x):
        if self.training:                            # use this batch's statistics
            mean, var = x.mean(0), x.var(0, unbiased=False)
            with torch.no_grad():                    # remember them for evaluation
                n = x.shape[0]
                self.run_mean.lerp_(mean, self.momentum)
                self.run_var.lerp_(var * n / (n - 1), self.momentum)
        else:                                        # use the running averages
            mean, var = self.run_mean, self.run_var
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


torch.manual_seed(0)
mine, ref = MyBatchNorm(4), nn.BatchNorm1d(4)
for step in range(5):                                # a few training batches
    x = torch.randn(32, 4) * 3 + 7                   # mean 7, std 3
    out_m, out_r = mine(x), ref(x)
print("train mode, max difference:", f"{(out_m - out_r).abs().max().item():.1e}")
print("output mean per feature :", (out_m.mean(0).detach().numpy().round(3) + 0.0))
print("output std per feature  :", out_m.std(0, unbiased=False).detach().numpy().round(3))
mine.eval(); ref.eval()
x = torch.randn(8, 4) * 3 + 7
print("eval mode, max difference:", f"{(mine(x) - ref(x)).abs().max().item():.1e}")
print("running mean:", mine.run_mean.numpy().round(2))

ln = nn.LayerNorm(4)                                 # normalizes each example instead
z = ln(torch.randn(3, 4) * 5 + 2)
print("LayerNorm: per-example means", (z.mean(1).detach().numpy().round(3) + 0.0))
