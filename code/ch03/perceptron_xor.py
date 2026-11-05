# XOR: the problem that froze neural-network research for a decade.
import numpy as np
from perceptron import Perceptron                 # run from inside code/ch03

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

p = Perceptron(2)
result = p.fit(X, y_xor, epochs=100)
print("converged:", result)                  # None -> never converged
print("predictions:", p.predict(X), "targets:", y_xor)
print("accuracy:", (p.predict(X) == y_xor).mean())
