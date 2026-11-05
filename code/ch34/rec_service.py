# The recommender as a service: recommendations for a user, and 'more like this' for a movie.
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from rec_data import ITEMS, SEEN, N_USERS, N_ITEMS
from recommenders import MFWithGenres

app = FastAPI(title="Movie recommender")
model = MFWithGenres()
model.load_state_dict(torch.load("mf_genres.pt")); model.eval()
with torch.no_grad():                                      # precompute once: serving is a lookup and a product
    USER_SCORES = model.all_scores(torch.arange(N_USERS)).masked_fill(SEEN, -torch.inf)
    V = F.normalize(model.item_vectors(torch.arange(N_ITEMS)), dim=1)


@app.get("/users/{user}/recommendations")
def recommend(user: int, k: int = 5):
    if not 0 <= user < N_USERS:
        raise HTTPException(404, "unknown user")
    top = USER_SCORES[user].topk(k)
    return [{"movie": ITEMS[i][0], "genres": ITEMS[i][1]} for i in top.indices.tolist()]


@app.get("/movies/{movie}/similar")
def similar(movie: int, k: int = 5):
    if not 0 <= movie < N_ITEMS:
        raise HTTPException(404, "unknown movie")
    sims = V @ V[movie]
    sims[movie] = -1
    return [{"movie": ITEMS[i][0], "similarity": round(sims[i].item(), 3)} for i in sims.topk(k).indices.tolist()]
