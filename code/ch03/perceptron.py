# The perceptron (Rosenblatt, 1958) - the first neuron that could learn.
import numpy as np


class Perceptron:
    def __init__(self, n_inputs, learning_rate=1.0):
        self.w = np.zeros(n_inputs)   # one weight per input
        self.b = 0.0                  # bias: how easily the neuron fires
        self.lr = learning_rate

    def predict(self, x):
        z = x @ self.w + self.b       # weighted sum
        return np.where(z >= 0, 1, 0) # step activation: fire or stay silent

    def fit(self, X, y, epochs=20, verbose=False):
        for epoch in range(epochs):
            mistakes = 0
            for xi, target in zip(X, y):
                error = target - self.predict(xi)   # -1, 0 or +1
                if error != 0:
                    self.w += self.lr * error * xi  # tilt the boundary
                    self.b += self.lr * error       # shift the boundary
                    mistakes += 1
            if verbose:
                print(f"epoch {epoch+1:2d}: mistakes={mistakes} "
                      f"w={self.w.round(2)} b={self.b:.2f}")
            if mistakes == 0:                       # converged
                return epoch + 1
        return None


if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    for name, y in [("AND", [0, 0, 0, 1]), ("OR", [0, 1, 1, 1])]:
        p = Perceptron(2)
        epochs = p.fit(X, np.array(y), verbose=(name == "AND"))
        print(f"{name}: learned in {epochs} epochs ->", p.predict(X))
