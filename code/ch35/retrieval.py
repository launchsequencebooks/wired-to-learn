"""Two retrievers over the book: BM25 (word matching) and a neural encoder trained on the book itself."""
import math
import re
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F

STOP = set("a an and are as at be by can do does for from how i if in is it its of on or should so that the "
           "this to what when which why with my me we you your".split())


def tokens(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


class BM25:
    """The classic ranking function of search engines (Robertson et al., 1994)."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [Counter(tokens(d)) for d in docs]
        self.len = np.array([sum(d.values()) for d in self.docs])
        self.k1, self.b, self.avg = k1, b, self.len.mean()
        df = Counter(w for d in self.docs for w in d)
        self.idf = {w: math.log(1 + (len(docs) - n + 0.5) / (n + 0.5)) for w, n in df.items()}

    def scores(self, query):
        s = np.zeros(len(self.docs))
        for w in tokens(query):
            if w in self.idf:
                tf = np.array([d[w] for d in self.docs])          # rare words count more (idf); repeated
                s += self.idf[w] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.len / self.avg))
        return s                                                  # words count less and less (saturation)


class TextEncoder(torch.nn.Module):
    """A bag of embeddings (Chapter 31) that maps any text to a unit vector."""

    def __init__(self, vocab, d=128):
        super().__init__()
        self.vocab = {w: i + 1 for i, w in enumerate(vocab)}
        self.emb = torch.nn.EmbeddingBag(len(vocab) + 1, d, mode="mean", padding_idx=0)
        self.proj = torch.nn.Linear(d, d)

    def ids(self, texts):
        seqs = [[self.vocab.get(w, 0) for w in tokens(t)] or [0] for t in texts]
        n = max(map(len, seqs))
        return torch.tensor([s + [0] * (n - len(s)) for s in seqs])

    def forward(self, texts):
        return F.normalize(self.proj(self.emb(self.ids(texts))), dim=1)
