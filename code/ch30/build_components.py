# Package three networks from earlier chapters as ONNX components for the organizer.
import sys, json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import torch
import onnx
from wtlkit import mnist_tensors, set_seed
# the networks of Chapters 16 (ResNet), 23 (SimCLR Encoder) and 19 (AutoEncoder), from the kit
from wtlkit.models import ResNet, Encoder, AutoEncoder, train_autoencoder


def export(module, path, meta):
    module.eval()
    # needs PyTorch >= 2.6 for the dynamo exporter (requirements.txt sets that floor)
    torch.onnx.export(module, (torch.zeros(1, 1, 28, 28),), path, input_names=["image"], output_names=["out"],
                      dynamic_shapes={"x": {0: torch.export.Dim("batch")}},     # forward(self, x): any batch size
                      dynamo=True, external_data=False, verbose=False)
    m = onnx.load(path)
    onnx.helper.set_model_props(m, {k: str(v) for k, v in meta.items()})
    onnx.save(m, path)
    print(f"{path:22s} {Path(path).stat().st_size / 1e6:5.2f} MB   {meta['role']}")


class ResNetEmbedder(torch.nn.Module):                   # the classifier minus its last layer
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net.features(x)


resnet = ResNet(n=3)
resnet.load_state_dict(torch.load(root / "ch16" / "fashion_resnet20.pt"))
export(ResNetEmbedder(resnet), "embed_resnet.onnx", {"role": "64-number embedding (supervised features)"})

simclr = Encoder()
simclr.load_state_dict(torch.load(root / "ch23" / "simclr_encoder.pt"))
export(simclr, "embed_simclr.onnx", {"role": "128-number embedding (self-supervised features)"})

# the Chapter 19 autoencoder was trained on digits; retrain the same architecture on clothes
set_seed(0)
(X_tr, _), (X_va, _), _ = mnist_tensors(dataset="fashion")
ae = train_autoencoder(AutoEncoder(16), X_tr, epochs=15, verbose=False).eval()

class FlatAE(torch.nn.Module):                          # accept images, return per-image error
    def __init__(self, ae):
        super().__init__()
        self.ae = ae

    def forward(self, x):
        flat = x.flatten(1)
        return ((self.ae(flat) - flat) ** 2).mean(1, keepdim=True)


export(FlatAE(ae), "outlier_ae.onnx", {"role": "reconstruction error (outlier score)"})

# reference data for the outlier detector: embeddings of 10,000 training clothes, and how
# both scores are distributed on normal validation images (to put them on a common scale)
import numpy as np
from wtlapps import OnnxModel
(X_tr, _), (X_va, _), _ = mnist_tensors(dataset="fashion", image=True)
emb, rec = OnnxModel("embed_resnet.onnx"), OnnxModel("outlier_ae.onnx")
ref = emb(X_tr[:10_000].numpy())
mean = ref.mean(0)
ref = (ref - mean) / np.linalg.norm(ref - mean, axis=1, keepdims=True)
va = X_va[:3000].numpy()
e = emb(va) - mean
nn_dist = 1 - ((e / np.linalg.norm(e, axis=1, keepdims=True)) @ ref.T).max(1)
log_ae, log_nn = np.log(rec(va)[:, 0]), np.log(nn_dist)
np.savez("outlier_reference.npz", ref=ref, mean=mean,
         ae_stats=[log_ae.mean(), log_ae.std()], nn_stats=[log_nn.mean(), log_nn.std()])

from organizer import OutlierDetector                     # threshold: 1% false alarms on other normal images
score = OutlierDetector(emb).score(X_va[3000:6000].numpy())
np.savez("outlier_reference.npz", ref=ref, mean=mean, ae_stats=[log_ae.mean(), log_ae.std()],
         nn_stats=[log_nn.mean(), log_nn.std()], threshold=np.quantile(score, 0.99))
print(f"outlier_reference.npz  10,000 reference embeddings, threshold {np.quantile(score, 0.99):.2f}")
