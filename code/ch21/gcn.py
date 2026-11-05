# A graph convolutional network (Kipf & Welling, 2017) in plain PyTorch.
import warnings
import torch
from torch import nn

warnings.filterwarnings("ignore", message="Sparse invariant")


def normalized_adjacency(edge_index, n):
    """Sparse D^-1/2 (A + I) D^-1/2 from a 2 x E list of directed edges."""
    loops = torch.arange(n)
    row = torch.cat([edge_index[0], loops]); col = torch.cat([edge_index[1], loops])
    deg = torch.zeros(n).index_add_(0, row, torch.ones(len(row)))
    vals = deg[row] ** -0.5 * deg[col] ** -0.5
    return torch.sparse_coo_tensor(torch.stack([row, col]), vals, (n, n)).coalesce()


class GCNLayer(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.lin = nn.Linear(n_in, n_out)

    def forward(self, x, A_norm):
        return torch.sparse.mm(A_norm, self.lin(x))       # transform, then average over neighbours


class GCN(nn.Module):
    def __init__(self, n_in, hidden, n_out, dropout=0.5):
        super().__init__()
        self.l1, self.l2 = GCNLayer(n_in, hidden), GCNLayer(hidden, n_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, A_norm, return_hidden=False):
        h = torch.relu(self.l1(self.drop(x), A_norm))
        out = self.l2(self.drop(h), A_norm)
        return (out, h) if return_hidden else out
