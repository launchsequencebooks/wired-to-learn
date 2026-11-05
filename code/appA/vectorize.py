# Why NumPy: the same computation with a Python loop and with one vectorized expression.
import time
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(2000, 300))
w = rng.normal(size=300)

t0 = time.perf_counter()
loop = [sum(X[i, j] * w[j] for j in range(300)) for i in range(2000)]
t_loop = time.perf_counter() - t0

t0 = time.perf_counter()
vec = X @ w
t_vec = time.perf_counter() - t0

print(f"same result: {np.allclose(loop, vec)}")
print(f"Python loops {t_loop * 1000:.0f} ms, NumPy {t_vec * 1000:.2f} ms (about {t_loop / t_vec:,.0f}x faster)")
