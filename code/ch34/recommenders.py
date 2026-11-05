"""Recommenders as embedding models: users and movies live in the same vector space."""
import torch
from torch import nn
from rec_data import N_USERS, N_ITEMS, ITEM_GENRES


class MatrixFactorization(nn.Module):
    """score(user, movie) = user vector . movie vector + movie bias."""

    def __init__(self, d=64):
        super().__init__()
        self.user, self.item = nn.Embedding(N_USERS, d), nn.Embedding(N_ITEMS, d)
        self.bias = nn.Embedding(N_ITEMS, 1)
        for e in (self.user, self.item):
            nn.init.normal_(e.weight, std=0.1)
        nn.init.zeros_(self.bias.weight)

    def item_vectors(self, items):
        return self.item(items)

    def score(self, users, items):
        return (self.user(users) * self.item_vectors(items)).sum(-1) + self.bias(items).squeeze(-1)

    def all_scores(self, users):                            # every movie for each user: one matrix product
        return self.user(users) @ self.item_vectors(torch.arange(N_ITEMS)).T + self.bias.weight.T


class MFWithGenres(MatrixFactorization):
    """A movie's vector also gets a contribution from its genres, so even a brand-new movie has one."""

    def __init__(self, d=64):
        super().__init__(d)
        self.genre = nn.Linear(ITEM_GENRES.shape[1], d, bias=False)

    def item_vectors(self, items):
        return self.item(items) + self.genre(ITEM_GENRES[items])


def bpr_loss(model, users, pos, neg):
    """Bayesian personalized ranking: a watched movie should score higher than a random unwatched one."""
    return -torch.nn.functional.logsigmoid(model.score(users, pos) - model.score(users, neg)).mean()
