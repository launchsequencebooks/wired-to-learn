# A forward pass through a 2-3-2 network, small enough to check by hand.
import numpy as np

relu = lambda z: np.maximum(0, z)

def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

x = np.array([1.0, 2.0])                      # one input example

W1 = np.array([[ 0.5, -1.0,  0.3],            # 2 inputs -> 3 hidden neurons
               [ 0.2,  0.4,  0.1]])
b1 = np.array([0.1, 0.0, 0.2])
W2 = np.array([[ 1.0, -0.5],                  # 3 hidden -> 2 outputs
               [ 0.7,  0.2],
               [-0.4,  0.9]])
b2 = np.array([0.0, 0.1])

z1 = x @ W1 + b1;  a1 = relu(z1)              # hidden layer
z2 = a1 @ W2 + b2; a2 = softmax(z2)           # output layer

for name, v in [("z1", z1), ("a1", a1), ("z2", z2), ("a2", a2)]:
    print(f"{name} = {np.round(v, 4)}")
print("predicted class:", a2.argmax())
