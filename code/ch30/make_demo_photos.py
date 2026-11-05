# Write a demo photo folder: 280 clothing photos and 20 stray digits, as PNG files.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from wtlkit import mnist_tensors

_, _, (X, _) = mnist_tensors(dataset="fashion", image=True)
_, _, (D, _) = mnist_tensors(dataset="mnist", image=True)
out = Path("photos"); out.mkdir(exist_ok=True)
for i in range(280):
    Image.fromarray((X[2000 + i, 0].numpy() * 255).astype(np.uint8)).save(out / f"photo_{i:03d}.png")
for i in range(20):
    Image.fromarray((D[500 + i, 0].numpy() * 255).astype(np.uint8)).save(out / f"scan_{i:02d}.png")
print(f"wrote {len(list(out.glob('*.png')))} photos to {out}/")
