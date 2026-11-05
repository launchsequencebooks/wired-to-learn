"""The next-event model of the log monitor (shared by detectors.py and monitor.py)."""
import numpy as np
import torch
from torch import nn
from log_data import EVENTS, ids

H, G = 10, 3                                              # history length; top-G predictions count as normal
START, END = len(EVENTS), len(EVENTS) + 1                # extra tokens: before the first event, after the last


class NextEvent(nn.Module):
    """DeepLog-style: an LSTM (Chapter 17) predicts the next event from the last H events."""

    def __init__(self, n_events, d=32, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(n_events + 2, d)
        self.lstm = nn.LSTM(d, hidden, num_layers=2, batch_first=True)
        self.out = nn.Linear(hidden, n_events + 2)         # predicts an event, or END

    def forward(self, history):
        h, _ = self.lstm(self.emb(history))
        return self.out(h[:, -1])


def windows(blocks):
    """(history, next event, block number) for every event of every block, and its end."""
    X, y, owner = [], [], []
    for n, b in enumerate(blocks):
        s = ids(b) + [END]                                 # 'the block ended here' is predicted too
        padded = [START] * H + s
        for t in range(len(s)):
            X.append(padded[t:t + H]); y.append(s[t]); owner.append(n)
    return torch.tensor(X), torch.tensor(y), np.array(owner)


def lstm_flags(model, blocks):
    """A block is anomalous if any of its events was not among the model's top-G guesses."""
    X, y, owner = windows(blocks)
    with torch.no_grad():
        top = torch.cat([model(X[i:i + 4096]).topk(G, 1).indices for i in range(0, len(X), 4096)])
    surprised = ~(top == y[:, None]).any(1).numpy() | (y.numpy() == 0)     # unseen events always count
    return np.bincount(owner, weights=surprised, minlength=len(blocks)) > 0
