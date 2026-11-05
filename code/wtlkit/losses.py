"""Loss functions that later chapters reuse.

    nt_xent    the contrastive loss of SimCLR, Chapter 23 (ch23/ntxent.py)
"""
import torch
import torch.nn.functional as F


def nt_xent(z1, z2, temperature=0.5):
    """z1[i] and z2[i] are embeddings of two views of the same image (a positive pair).
    Every other embedding in the batch is a negative."""
    z = F.normalize(torch.cat([z1, z2]), dim=1)          # 2N unit vectors
    sim = z @ z.T / temperature                          # cosine similarities, sharpened
    n = len(z1)
    sim.fill_diagonal_(float("-inf"))                    # an image is not its own positive
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])   # index of the partner
    return F.cross_entropy(sim, targets)                 # "pick your partner out of the crowd"
