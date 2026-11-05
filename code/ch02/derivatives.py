# Derivatives by experiment: nudge the input, watch the output.
import numpy as np

def f(x):
    return x**2 - 2*x + 3

def numerical_derivative(f, x, h=1e-5):
    """Central difference: slope of the line through f(x-h) and f(x+h)."""
    return (f(x + h) - f(x - h)) / (2 * h)

for x in [-1.0, 1.0, 3.0]:
    exact = 2*x - 2                      # from the power rule
    approx = numerical_derivative(f, x)
    print(f"x={x:+.1f}  exact={exact:+.4f}  numerical={approx:+.4f}")

# Gradient of a function of two variables: one partial derivative each
def g(v):
    x, y = v
    return x**2 + 3*y**2

def numerical_gradient(g, v, h=1e-5):
    grad = np.zeros_like(v)
    for i in range(len(v)):
        step = np.zeros_like(v); step[i] = h
        grad[i] = (g(v + step) - g(v - step)) / (2 * h)
    return grad

v = np.array([1.0, -2.0])
print("numerical gradient:", numerical_gradient(g, v).round(4))
print("exact gradient    :", np.array([2*v[0], 6*v[1]]))
