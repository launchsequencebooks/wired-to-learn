# Why deep sigmoid networks stall: backprop multiplies one activation
# derivative per layer. Watch what happens to that product with depth.
import numpy as np
from activations import d_sigmoid, d_tanh, d_relu

z = 1.0   # a typical pre-activation value for an active neuron
per_layer = {"sigmoid": d_sigmoid(z), "tanh": d_tanh(z), "relu": d_relu(z)}
print("derivative at z=1:", {k: round(float(v), 3) for k, v in per_layer.items()})
for depth in [1, 5, 10, 20]:
    print(f"{depth:2d} layers -> " + "  ".join(
        f"{k}={v**depth:.1e}" for k, v in per_layer.items()))
