# Anomaly detection: an autoencoder trained only on digits reconstructs clothes badly.
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from wtlkit import mnist_tensors
from autoencoder import AutoEncoder

model = AutoEncoder(16)
model.load_state_dict(torch.load("autoencoder.pt")); model.eval()
_, (X_va, _), (X_digits, _) = mnist_tensors()
_, _, (X_clothes, _) = mnist_tensors(dataset="fashion")

def errors(X):
    with torch.no_grad():
        return ((model(X) - X) ** 2).mean(dim=1)       # one error score per image

e_val, e_dig, e_clo = errors(X_va), errors(X_digits), errors(X_clothes)
threshold = torch.quantile(e_val, 0.99).item()          # flag the worst 1% of normal data
print(f"median error: digits {e_dig.median():.4f}, clothes {e_clo.median():.4f}")
print(f"threshold (99th percentile of validation digits): {threshold:.4f}")
print(f"digits flagged as anomalies : {(e_dig > threshold).float().mean():.1%}")
print(f"clothes flagged as anomalies: {(e_clo > threshold).float().mean():.1%}")
json.dump({"digits": e_dig.tolist(), "clothes": e_clo.tolist(), "threshold": threshold},
          open("anomaly_scores.json", "w"))
