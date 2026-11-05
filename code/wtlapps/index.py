"""Nearest-neighbour search over embeddings: the engine behind 'find similar'."""
import numpy as np


class EmbeddingIndex:
    """Stores L2-normalized vectors; similarity is the cosine (a dot product of unit vectors)."""

    def __init__(self, dim):
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.ids = []

    @staticmethod
    def _normalize(v):
        v = np.atleast_2d(np.asarray(v, dtype=np.float32))
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

    def add(self, vectors, ids):
        self.vectors = np.vstack([self.vectors, self._normalize(vectors)])
        self.ids.extend(ids)

    def search(self, queries, k=5, exclude_ids=None):
        """Top-k (ids, similarities) for each query, most similar first."""
        sims = self._normalize(queries) @ self.vectors.T          # all cosines in one matrix product
        if exclude_ids is not None:                   # e.g. the query's own id
            pos = {i: n for n, i in enumerate(self.ids)}
            for row, i in enumerate(exclude_ids):
                sims[row, pos[i]] = -np.inf
        top = np.argsort(-sims, axis=1)[:, :k]
        return [[self.ids[j] for j in row] for row in top], np.take_along_axis(sims, top, 1)

    def save(self, path):
        np.savez(path, vectors=self.vectors, ids=np.array(self.ids, dtype=object))

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        index = cls(d["vectors"].shape[1])
        index.vectors, index.ids = d["vectors"], list(d["ids"])
        return index
