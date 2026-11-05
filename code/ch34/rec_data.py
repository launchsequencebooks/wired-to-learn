"""MovieLens 100K for recommendation: each user's last two ratings are held out."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from wtlkit import load_movielens

ratings, ITEMS = load_movielens()
N_USERS, N_ITEMS = ratings[:, 0].max() + 1, len(ITEMS)
GENRES = sorted({g for _, gs in ITEMS for g in gs})
ITEM_GENRES = torch.zeros(N_ITEMS, len(GENRES))
for i, (_, gs) in enumerate(ITEMS):
    for g in gs:
        ITEM_GENRES[i, GENRES.index(g)] = 1

# leave-last-out: for each user, the most recent rating is the test item, the one before it the validation item
order = np.lexsort((ratings[:, 3], ratings[:, 0]))         # by user, then time
r = ratings[order]
last = np.r_[r[1:, 0] != r[:-1, 0], True]
second_last = np.r_[last[1:], False]
TEST = torch.tensor(r[last][:, :2])
VAL = torch.tensor(r[second_last][:, :2])
TRAIN = torch.tensor(r[~last & ~second_last][:, :2])      # (user, item) pairs: 'this user watched this'
SEEN = torch.zeros(N_USERS, N_ITEMS, dtype=torch.bool)
SEEN[TRAIN[:, 0], TRAIN[:, 1]] = True


def evaluate(scores, held_out, exclude=(SEEN,), k=10):
    """Hit rate and NDCG at k: rank the held-out item among ALL items the user has not seen."""
    scores = scores.clone()
    for mask in exclude:
        scores[mask] = -torch.inf                         # never recommend what they already watched
    target = scores[held_out[:, 0], held_out[:, 1]]
    rank = (scores[held_out[:, 0]] > target[:, None]).sum(1)          # 0 = top of the list
    hit = rank < k
    ndcg = torch.where(hit, 1 / torch.log2(rank.float() + 2), torch.zeros(()))
    return hit.float().mean().item(), ndcg.mean().item()


if __name__ == "__main__":
    print(f"{len(ratings):,} ratings by {N_USERS} users of {N_ITEMS:,} movies "
          f"({len(ratings) / (N_USERS * N_ITEMS):.1%} of all user-movie pairs)")
    print(f"train {len(TRAIN):,} interactions; one validation and one test movie per user")
    print("a movie:", ITEMS[49])
