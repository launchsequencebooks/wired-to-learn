# NumPy arrays: creating, shaping, indexing and computing with them.
import numpy as np

x = np.array([1.0, 2.0, 3.0])                 # a vector
W = np.arange(6, dtype=float).reshape(2, 3)   # a 2x3 matrix: 0..5 laid out row by row
print("x:", x, "shape", x.shape)
print("W:\n", W, "shape", W.shape, "dtype", W.dtype)

print("W[1, 2] =", W[1, 2], "  row 0:", W[0], "  column 1:", W[:, 1])
print("x > 1.5 ->", x > 1.5, "  x[x > 1.5] ->", x[x > 1.5])   # boolean masks select elements

print("element-wise x * x:", x * x)
print("matrix-vector W @ x:", W @ x)          # (2x3) @ (3,) -> (2,)
print("sum of all:", W.sum(), " column sums:", W.sum(axis=0), " row means:", W.mean(axis=1))

rng = np.random.default_rng(0)                # a seeded random generator: same numbers every run
print("random normals:", rng.normal(size=3).round(3))

v = W[0]                                      # a VIEW: shares memory with W
v[0] = 99
print("after changing the view, W[0, 0] =", W[0, 0], "(use .copy() to avoid this)")
