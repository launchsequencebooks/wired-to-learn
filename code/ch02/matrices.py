# Matrices: a whole layer of neurons computed in one line.
import numpy as np

# 3 neurons, each with 2 weights -> W has shape (2, 3)
W = np.array([[ 0.2, -0.5,  1.0],
              [ 0.7,  0.1, -0.3]])
b = np.array([0.1, 0.0, -0.2])     # one bias per neuron

x = np.array([1.0, 2.0])           # one example
print("one example :", x @ W + b)  # 3 outputs, one per neuron

X = np.array([[1.0, 2.0],          # a batch of 4 examples, shape (4, 2)
              [0.0, 1.0],
              [3.0, -1.0],
              [0.5, 0.5]])
Z = X @ W + b                      # (4,2) @ (2,3) -> (4,3); b is broadcast
print("batch shape :", Z.shape)
print(Z.round(2))
print("W.T (rows and columns swapped):\n", W.T)
