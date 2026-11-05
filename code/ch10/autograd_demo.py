# Autograd: PyTorch records operations and runs backpropagation for you.
# Same tiny network as Chapter 7 - compare the gradients!
import torch

torch.set_default_dtype(torch.float64)         # double precision, like NumPy
x, y = torch.tensor(1.5), torch.tensor(1.0)
w1 = torch.tensor(0.8, requires_grad=True)     # 'track my gradient'
b1 = torch.tensor(-0.2, requires_grad=True)
w2 = torch.tensor(1.2, requires_grad=True)
b2 = torch.tensor(0.3, requires_grad=True)

a1 = torch.sigmoid(w1 * x + b1)                # forward pass builds a graph
y_hat = w2 * a1 + b2
L = 0.5 * (y_hat - y) ** 2
print(f"L = {L.item():.5f}   (graph node: {L.grad_fn.__class__.__name__})")

L.backward()                                   # backward pass, automatically
for name, p in [("w1", w1), ("b1", b1), ("w2", w2), ("b2", b2)]:
    print(f"dL/d{name} = {p.grad.item():.5f}")
