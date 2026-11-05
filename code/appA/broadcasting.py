# Broadcasting: operating on arrays of different shapes without loops.
import numpy as np

X = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])               # 2 examples, 3 features
b = np.array([10.0, 20.0, 30.0])              # one bias per feature
print("X + b:\n", X + b)                      # b is repeated for every row

mean = X.mean(axis=0)                         # per-feature mean, shape (3,)
std = X.std(axis=0)
print("standardized:\n", ((X - mean) / std).round(3))

col = np.array([[1.0], [2.0]])                # shape (2, 1): one number per row
print("X * col:\n", X * col)                  # scales each row by its own number

try:
    X + np.array([1.0, 2.0])                  # shapes (2,3) and (2,) do not line up
except ValueError as e:
    print("error:", e)
