# The classic PyTorch bug: forgetting optimizer.zero_grad().
import torch

w = torch.tensor(2.0, requires_grad=True)
for step in range(3):
    loss = (3 * w) ** 2          # d(loss)/dw = 18 w = 36 at w = 2
    loss.backward()
    print(f"step {step}: w.grad = {w.grad.item():.0f}")   # grows: 36, 72, 108!
w.grad.zero_()                   # what optimizer.zero_grad() does for every parameter
print("after zeroing:", w.grad.item())
