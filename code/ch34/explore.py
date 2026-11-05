# Look inside the trained recommender: similar movies, one user's list, and a brand-new movie.
import torch
import torch.nn.functional as F
from rec_data import ITEMS, GENRES, SEEN, TRAIN, N_ITEMS
from recommenders import MFWithGenres

model = MFWithGenres()
model.load_state_dict(torch.load("mf_genres.pt")); model.eval()
title = [t for t, _ in ITEMS]
with torch.no_grad():
    V = F.normalize(model.item_vectors(torch.arange(N_ITEMS)), dim=1)    # movie embeddings, unit length
    popular = torch.bincount(TRAIN[:, 1], minlength=N_ITEMS) >= 20         # only well-known movies, for clarity


def neighbours(v, k=4, skip=None):
    sims = (V @ v).masked_fill(~popular, -1)
    if skip is not None:
        sims[skip] = -1
    return [title[i] for i in sims.topk(k).indices]


for name in ["Star Wars (1977)", "Fargo (1996)", "Sleepless in Seattle (1993)"]:
    i = title.index(name)
    print(f"like {name}: {', '.join(neighbours(V[i], skip=i))}")

user = 2
with torch.no_grad():
    scores = model.all_scores(torch.tensor([user]))[0].masked_fill(SEEN[user], -torch.inf)
watched = [title[i] for i in TRAIN[TRAIN[:, 0] == user, 1][:4]]
print(f"\nuser {user} has watched {int(SEEN[user].sum())} movies, e.g. {', '.join(watched)}")
print("recommended:", ", ".join(title[i] for i in scores.topk(5).indices))

genres = torch.zeros(len(GENRES))                          # a new animated family comedy: no ratings yet
for g in ["Animation", "Children's", "Comedy"]:
    genres[GENRES.index(g)] = 1
with torch.no_grad():
    new = F.normalize(model.genre(genres), dim=0)          # its vector comes from its genres alone
print("\na brand-new animated children's comedy sits next to:", ", ".join(neighbours(new)))
