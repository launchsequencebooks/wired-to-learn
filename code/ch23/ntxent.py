# The contrastive loss of SimCLR (NT-Xent), computed by hand on a tiny batch.
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


if __name__ == "__main__":
    torch.manual_seed(0)
    anchors = torch.randn(4, 8)                          # 4 "images", 8-dim embeddings
    close = anchors + 0.1 * torch.randn(4, 8)            # views that agree with their partner
    random = torch.randn(4, 8)                           # views unrelated to their partner
    print(f"loss when partners agree     : {nt_xent(anchors, close):.3f}")
    print(f"loss when partners are random: {nt_xent(anchors, random):.3f}")
    print(f"chance level, ln(2N - 1)     : {torch.log(torch.tensor(7.0)):.3f}")
