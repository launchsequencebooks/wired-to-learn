# Activation functions and their derivatives - the toolkit we reuse
# throughout the book. Each derivative is written in terms of the input z.
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def d_sigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)                 # elegant: uses its own output

def tanh(z):
    return np.tanh(z)

def d_tanh(z):
    return 1 - np.tanh(z) ** 2

def relu(z):
    return np.maximum(0, z)

def d_relu(z):
    return np.where(z > 0, 1.0, 0.0)   # 0 for z<=0, 1 for z>0

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def d_leaky_relu(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)

def gelu(z):                           # tanh approximation (used by BERT/GPT)
    return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))

def d_gelu(z):                         # derivative of the tanh approximation
    c = np.sqrt(2 / np.pi)
    u = c * (z + 0.044715 * z**3)
    t = np.tanh(u)
    return 0.5 * (1 + t) + 0.5 * z * (1 - t**2) * c * (1 + 3 * 0.044715 * z**2)

def softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)   # stability trick
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


ACTIVATIONS = {
    "sigmoid": (sigmoid, d_sigmoid),
    "tanh": (tanh, d_tanh),
    "relu": (relu, d_relu),
    "leaky_relu": (leaky_relu, d_leaky_relu),
    "gelu": (gelu, d_gelu),
}

if __name__ == "__main__":
    # Gradient check: compare each analytic derivative with a numerical one
    z = np.linspace(-4, 4, 41) + 0.013          # avoid the kink exactly at 0
    for name, (f, df) in ACTIVATIONS.items():
        numeric = (f(z + 1e-6) - f(z - 1e-6)) / 2e-6
        err = np.max(np.abs(numeric - df(z)))
        print(f"{name:11s} max derivative error: {err:.2e}")

    logits = np.array([2.0, 1.0, 0.1])
    print("softmax:", softmax(logits).round(3), "sum =", softmax(logits).sum())
    print("big logits:", softmax(np.array([1000.0, 999.0])).round(3))
