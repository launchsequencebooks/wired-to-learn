# Matplotlib: a figure with two curves, labels and a legend, saved to a file.
import numpy as np
import matplotlib
matplotlib.use("Agg")                         # draw to files, no window needed
import matplotlib.pyplot as plt

z = np.linspace(-4, 4, 200)                   # 200 evenly spaced points
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(z, 1 / (1 + np.exp(-z)), label="sigmoid")
ax.plot(z, np.maximum(0, z), label="ReLU")
ax.set_xlabel("z"); ax.set_ylabel("activation"); ax.set_ylim(-0.2, 2)
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("activations.png", dpi=200)
print("saved activations.png")
