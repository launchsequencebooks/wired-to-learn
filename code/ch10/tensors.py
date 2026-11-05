# PyTorch tensors: NumPy arrays with two superpowers (GPUs and autograd).
import numpy as np
import torch

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])      # from a Python list
b = torch.ones(2, 2)                              # like np.ones
c = torch.randn(3, 4)                             # random normal, like rng.normal
print("a:", a, sep="\n")
print("shape", tuple(c.shape), "| dtype", c.dtype, "| device", c.device)

print("a @ b =", (a @ b).tolist())                # matrix product, same operator
print("a * b =", (a * b).tolist())                # element-wise, same as NumPy
print("mean of each column:", a.mean(dim=0).tolist())   # 'dim' plays the role of 'axis'

x = np.arange(6, dtype=np.float32).reshape(2, 3)
t = torch.from_numpy(x)                           # shares memory with x
t[0, 0] = 100.0
print("NumPy array changed too:", x[0, 0])
print("back to NumPy:", (t * 2).numpy()[0])

device = "cuda" if torch.cuda.is_available() else "cpu"
print("computing on:", device)
print("on device:", c.to(device).device)
