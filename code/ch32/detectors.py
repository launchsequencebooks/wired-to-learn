# Two neural detectors trained only on normal blocks, compared with a simple rule.
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from log_data import TRAIN_BLOCKS, VAL_BLOCKS, TEST_BLOCKS, LABELS, EVENTS, ids, counts
from log_data import lines, BLOCK
from next_event import NextEvent, windows, lstm_flags
from wtlkit.models import train_autoencoder    # Chapter 19 trainer, kept in the kit
from wtlkit import set_seed

y_test = np.array([LABELS[b] for b in TEST_BLOCKS])

set_seed(0)
lstm = NextEvent(len(EVENTS))
X, y, _ = windows(TRAIN_BLOCKS)
opt = torch.optim.Adam(lstm.parameters(), lr=3e-3)
for epoch in range(4):
    for idx in torch.randperm(len(X)).split(256):
        loss = F.cross_entropy(lstm(X[idx]), y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
lstm.eval()
torch.save(lstm.state_dict(), "next_event_lstm.pt")
results = {"LSTM next-event (top-3)": lstm_flags(lstm, TEST_BLOCKS)}

# the Chapter 19 autoencoder idea, on bag-of-events vectors (log-scaled counts)
C_tr, C_va, C_te = (torch.tensor(np.log1p(counts(b))) for b in (TRAIN_BLOCKS, VAL_BLOCKS, TEST_BLOCKS))


class CountAE(nn.Module):
    def __init__(self, n, code=4):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n, 32), nn.ReLU(), nn.Linear(32, code))
        self.decoder = nn.Sequential(nn.Linear(code, 32), nn.ReLU(), nn.Linear(32, n))

    def forward(self, x):
        return self.decoder(self.encoder(x))


set_seed(0)
ae = train_autoencoder(CountAE(len(EVENTS)), C_tr, epochs=60, verbose=False).eval()
with torch.no_grad():
    err_va, err_te = ((ae(C_va) - C_va) ** 2).mean(1), ((ae(C_te) - C_te) ** 2).mean(1)
results["autoencoder on event counts"] = (err_te > torch.quantile(err_va, 0.99)).numpy()   # 1% false alarms on validation
results["rule: any event unseen in training"] = (counts(TEST_BLOCKS)[:, 0] > 0)

# the excerpt ends mid-operation: blocks still active in its last lines have no END
last = {b: i for i, l in enumerate(lines) for b in BLOCK.findall(l["Content"])}
tail = np.array([last[b] > len(lines) - 2000 for b in TEST_BLOCKS])
cut_off = (results["LSTM next-event (top-3)"] & tail & ~y_test).sum()

normal_seqs = {tuple(ids(b)) for b in TRAIN_BLOCKS}
normal_counts = {tuple(sorted(ids(b))) for b in TRAIN_BLOCKS}
anomalous = [b for b in TEST_BLOCKS if LABELS[b]]
same_seq = sum(tuple(ids(b)) in normal_seqs for b in anomalous)
same_counts = sum(tuple(sorted(ids(b))) in normal_counts for b in anomalous)
print(f"{len(anomalous)} anomalous test blocks; {same_seq} have exactly the events, in exactly the order, of a normal "
      f"training block\n(best possible recall {1 - same_seq / len(anomalous):.3f} from order, "
      f"{1 - same_counts / len(anomalous):.3f} from counts alone)")
for name, flags in results.items():
    p, r, f, _ = precision_recall_fscore_support(y_test, flags, average="binary", zero_division=0)
    print(f"{name:36s} precision {p:.3f}  recall {r:.3f}  F1 {f:.3f}  ({flags.sum()} blocks flagged)")
print(f"{tail.sum()} test blocks were still active in the last 2,000 lines; "
      f"{cut_off} of the LSTM's false alarms are among them")
json.dump({k: v.tolist() for k, v in results.items()}, open("flags.json", "w"))
