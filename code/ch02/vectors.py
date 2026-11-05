# Vectors in NumPy: the "lists of numbers" that neural networks are made of.
import numpy as np

x = np.array([2.0, 1.0])          # an input with two features
w = np.array([0.5, -1.5])         # a weight for each feature

print("x + w        =", x + w)                 # element-wise sum
print("3 * x        =", 3 * x)                 # scaling
print("x * w        =", x * w)                 # element-wise product
print("dot(x, w)    =", np.dot(x, w))          # 2*0.5 + 1*(-1.5)
print("x @ w        =", x @ w)                 # same thing, nicer syntax
print("length |x|   =", np.linalg.norm(x))     # sqrt(2^2 + 1^2)

# The angle between two vectors comes straight from the dot product
cos_theta = (x @ w) / (np.linalg.norm(x) * np.linalg.norm(w))
print("angle (deg)  =", np.degrees(np.arccos(cos_theta)).round(1))
