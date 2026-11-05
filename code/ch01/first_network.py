# Your first neural network: it learns XOR in about 30 lines of NumPy.
# Don't worry if some lines look mysterious - by Chapter 7 you will
# be able to write every one of them yourself.
import numpy as np

rng = np.random.default_rng(42)

# The four XOR examples: output is 1 when exactly one input is 1
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

# A tiny network: 2 inputs -> 4 hidden neurons -> 1 output
W1 = rng.normal(0, 1, (2, 4)); b1 = np.zeros(4)
W2 = rng.normal(0, 1, (4, 1)); b2 = np.zeros(1)
sigmoid = lambda z: 1 / (1 + np.exp(-z))

for step in range(5001):
    # forward pass: make a prediction (@ is matrix multiplication, Chapter 2)
    h = np.tanh(X @ W1 + b1)
    p = sigmoid(h @ W2 + b2)
    # backward pass: how should each weight change to reduce the error?
    dz2 = (p - y) / len(X)
    dW2, db2 = h.T @ dz2, dz2.sum(0)   # .T flips rows and columns (Chapter 2)
    dz1 = (dz2 @ W2.T) * (1 - h**2)
    dW1, db1 = X.T @ dz1, dz1.sum(0)
    # learn: nudge every weight a little bit downhill
    for param, grad in ((W1, dW1), (b1, db1), (W2, dW2), (b2, db2)):
        param -= 1.0 * grad
    if step % 1000 == 0:
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        print(f"step {step:4d}  loss {loss:.4f}")

print("predictions:", p.round(3).ravel())
