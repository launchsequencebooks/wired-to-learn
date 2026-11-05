"""Two neural text classifiers built from parts of earlier chapters."""
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import torch
from torch import nn
from wtlkit.models import Block                          # the Transformer block of Chapter 18


class BagOfEmbeddings(nn.Module):
    """Average the word embeddings, then an MLP: word order is ignored."""

    def __init__(self, vocab, n_classes, d=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=0)
        self.mlp = nn.Sequential(nn.Dropout(0.3), nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_classes))

    def embed(self, ids):
        mask = (ids > 0).float()[..., None]
        return (self.emb(ids) * mask).sum(1) / mask.sum(1).clamp(min=1)

    def forward(self, ids):
        return self.mlp(self.embed(ids))


class TicketTransformer(nn.Module):
    """Word embeddings + positions -> Transformer blocks -> mean over the words -> classes."""

    def __init__(self, vocab, n_classes, d=128, heads=4, layers=2, max_len=48):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d, padding_idx=0), nn.Embedding(max_len, d)
        self.blocks = nn.Sequential(*[Block(d, heads, dropout=0.2) for _ in range(layers)])
        self.ln, self.head = nn.LayerNorm(d), nn.Sequential(nn.Dropout(0.2), nn.Linear(d, n_classes))

    def embed(self, ids):
        mask = (ids > 0).float()[..., None]
        x = self.blocks(self.tok(ids) + self.pos(torch.arange(ids.shape[1])))
        return (self.ln(x) * mask).sum(1) / mask.sum(1).clamp(min=1)   # average over real words only

    def forward(self, ids):
        return self.head(self.embed(ids))
