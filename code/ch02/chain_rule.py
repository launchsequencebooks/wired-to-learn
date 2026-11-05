# The chain rule in action: y = sin(x^2). Compare the hand-derived
# derivative with a numerical estimate.
import numpy as np

x = 1.3
u = x**2            # inner function
y = np.sin(u)       # outer function

dy_du = np.cos(u)   # derivative of the outer, evaluated at the inner
du_dx = 2 * x       # derivative of the inner
dy_dx = dy_du * du_dx   # chain rule: multiply along the path

h = 1e-6
numeric = (np.sin((x + h)**2) - np.sin((x - h)**2)) / (2 * h)
print(f"chain rule : {dy_dx:.6f}")
print(f"numerical  : {numeric:.6f}")
