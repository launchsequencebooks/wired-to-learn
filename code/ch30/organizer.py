"""The Smart Photo Organizer: four networks from this book working as one application."""
import sys, json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import numpy as np
from wtlapps import OnnxModel, EmbeddingIndex, ConfidenceRouter

HERE = Path(__file__).resolve().parent


class OutlierDetector:
    """Two independent opinions on 'is this like the photos we know?', on a common scale:
    the autoencoder's reconstruction error, and the distance to the nearest known photo."""

    def __init__(self, embedder):
        self.ae = OnnxModel(HERE / "outlier_ae.onnx")
        self.embedder = embedder
        r = np.load(HERE / "outlier_reference.npz")
        self.ref, self.mean, self.ae_stats, self.nn_stats = r["ref"], r["mean"], r["ae_stats"], r["nn_stats"]
        self.threshold = float(r["threshold"]) if "threshold" in r else np.inf

    def score(self, images, embeddings=None):
        e = (self.embedder(images) if embeddings is None else embeddings) - self.mean
        nn_dist = 1 - ((e / np.linalg.norm(e, axis=1, keepdims=True)) @ self.ref.T).max(1)
        z_ae = (np.log(self.ae(images)[:, 0]) - self.ae_stats[0]) / self.ae_stats[1]
        z_nn = (np.log(nn_dist) - self.nn_stats[0]) / self.nn_stats[1]
        return z_ae + z_nn                                        # large = unlike anything we know


class Organizer:
    def __init__(self, embedder="embed_resnet.onnx", tag_threshold=None):
        self.tagger = OnnxModel(root / "ch28" / "resnet20_int8.onnx")   # what is it?
        self.embedder = OnnxModel(HERE / embedder)                     # what does it look like?
        self.outliers = OutlierDetector(OnnxModel(HERE / "embed_resnet.onnx"))   # does it belong here?
        if tag_threshold is None:                 # the 99% threshold Chapter 29 saved
            with open(root / "ch29" / "router_99.json") as f:
                tag_threshold = json.load(f)["threshold"]
        self.router = ConfidenceRouter(tag_threshold)

    def ingest(self, images, names):
        """Analyse a batch of (N, 1, 28, 28) images with pixels in [0, 1]."""
        p = self.tagger.predict_proba(images)
        self.names = list(names)
        self.tags = [self.tagger.labels[i] for i in p.argmax(1)]
        self.confidence = p.max(1)
        self.needs_review = self.router.route(self.confidence) == "human"
        self.outlier_score = self.outliers.score(images)
        self.is_outlier = self.outlier_score > self.outliers.threshold
        self.index = EmbeddingIndex(self.embedder(images[:1]).shape[1])
        self.index.add(self.embedder(images), self.names)
        return self

    def albums(self):
        """Confidently tagged photos go to their tag's album; the rest wait for a person."""
        groups = {}
        for name, tag, review, odd in zip(self.names, self.tags, self.needs_review, self.is_outlier):
            key = "Not clothing?" if odd else ("Review" if review else tag)
            groups.setdefault(key, []).append(name)
        return groups

    def similar(self, name, k=5):
        ids, sims = self.index.search(self.index.vectors[self.names.index(name)], k=k, exclude_ids=[name])
        return [(i, round(float(s), 3)) for i, s in zip(ids[0], sims[0])]
