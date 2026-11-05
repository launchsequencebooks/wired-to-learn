# Train the recommenders and compare them with 'recommend the most popular movies'.
import torch
from rec_data import TRAIN, VAL, TEST, SEEN, N_USERS, N_ITEMS, evaluate
from recommenders import MatrixFactorization, MFWithGenres, bpr_loss
from wtlkit import set_seed

val_seen = SEEN.clone(); val_seen[VAL[:, 0], VAL[:, 1]] = True       # at test time the validation movie is known too
users = torch.arange(N_USERS)
popularity = torch.bincount(TRAIN[:, 1], minlength=N_ITEMS).float()
hr, ndcg = evaluate(popularity.expand(N_USERS, -1), TEST, exclude=(val_seen,))
print(f"{'most popular':24s} HR@10 {hr:.3f}  NDCG@10 {ndcg:.3f}")

for name, Model in [("matrix factorization", MatrixFactorization), ("MF + genres", MFWithGenres)]:
    set_seed(0)
    model = Model()
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)
    best, best_state = 0, None
    for epoch in range(30):
        model.train()
        for idx in torch.randperm(len(TRAIN)).split(1024):
            u, pos = TRAIN[idx, 0], TRAIN[idx, 1]
            neg = torch.randint(0, N_ITEMS, (len(idx),))   # a random movie stands in for 'not chosen'
            loss = bpr_loss(model, u, pos, neg)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val_hr, _ = evaluate(model.all_scores(users), VAL)
        if val_hr > best:
            best, best_state = val_hr, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    with torch.no_grad():
        hr, ndcg = evaluate(model.all_scores(users), TEST, exclude=(val_seen,))
    print(f"{name:24s} HR@10 {hr:.3f}  NDCG@10 {ndcg:.3f}   (best validation HR@10 {best:.3f})")
    torch.save(model.state_dict(), f"{'mf_genres' if 'genres' in name else 'mf'}.pt")
