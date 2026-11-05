# Backpropagation by hand on the smallest possible network:
#   x -> [w1, b1] -> sigmoid -> [w2, b2] -> y_hat,   L = 1/2 (y_hat - y)^2
import numpy as np

sigmoid = lambda z: 1 / (1 + np.exp(-z))
x, y = 1.5, 1.0
w1, b1, w2, b2 = 0.8, -0.2, 1.2, 0.3

def loss(w1, b1, w2, b2):
    a1 = sigmoid(w1 * x + b1)
    return 0.5 * (w2 * a1 + b2 - y) ** 2

# ---- forward pass: compute and remember every intermediate value
z1 = w1 * x + b1          # 1.0
a1 = sigmoid(z1)          # 0.7311
y_hat = w2 * a1 + b2      # 1.1773
L = 0.5 * (y_hat - y)**2

# ---- backward pass: one local derivative per step, multiplied backward
dL_dyhat = y_hat - y              # dL/dy_hat
dL_dw2 = dL_dyhat * a1            # y_hat = w2*a1 + b2
dL_db2 = dL_dyhat * 1.0
dL_da1 = dL_dyhat * w2
dL_dz1 = dL_da1 * a1 * (1 - a1)   # sigmoid'(z1) = a1 (1 - a1)
dL_dw1 = dL_dz1 * x               # z1 = w1*x + b1
dL_db1 = dL_dz1 * 1.0

print(f"forward : z1={z1:.4f} a1={a1:.4f} y_hat={y_hat:.4f} L={L:.5f}")
print(f"backward: dL/dw2={dL_dw2:.5f} dL/db2={dL_db2:.5f} "
      f"dL/dw1={dL_dw1:.5f} dL/db1={dL_db1:.5f}")

# ---- check every gradient numerically
h, p = 1e-6, [w1, b1, w2, b2]
numeric = []
for i in range(4):
    up, dn = list(p), list(p)
    up[i] += h; dn[i] -= h
    numeric.append((loss(*up) - loss(*dn)) / (2 * h))
print("numeric :", np.round(numeric, 5))
