# A denoising autoencoder: trained to remove heavy noise it adds to its own inputs.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from wtlkit import mnist_tensors, set_seed
from autoencoder import AutoEncoder, train_autoencoder

set_seed(0)
(X_tr, _), _, (X_te, _) = mnist_tensors()
model = train_autoencoder(AutoEncoder(32), X_tr, epochs=10, noise=0.5, verbose=False)
model.eval()
torch.manual_seed(1)
noisy = (X_te + 0.5 * torch.randn_like(X_te)).clamp(0, 1)
with torch.no_grad():
    cleaned = model(noisy)
psnr = lambda a, b: 10 * torch.log10(1 / nn.functional.mse_loss(a, b)).item()
print(f"noisy images   : MSE {nn.functional.mse_loss(noisy, X_te):.4f}  PSNR {psnr(noisy, X_te):.1f} dB")
print(f"denoised images: MSE {nn.functional.mse_loss(cleaned, X_te):.4f}  PSNR {psnr(cleaned, X_te):.1f} dB")
torch.save({"noisy": noisy[:8], "clean": X_te[:8], "denoised": cleaned[:8]}, "denoise_examples.pt")
