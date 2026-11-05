# Chapter 34 · Project: A Recommender System

*Part VII: Building Intelligent Applications* · [All chapters](../../README.md#chapters)

Run every command from this folder, inside your activated environment.

## Before you start

Listings 34.4 (`explore.py`) and 34.5 (`rec_service.py`) load one trained file:

| File | Trained by | Training time |
|---|---|---|
| `ch34/mf_genres.pt` (and `ch34/mf.pt`) | Listing 34.3, `train_recommenders.py` | under a minute on a 2-thread laptop CPU |

Either run the listings in order, or fetch the files with
`python scripts/fetch_artifacts.py ch34` from the `code` folder. MovieLens 100K is
downloaded from GroupLens on first use by `rec_data.py` (research license,
non-commercial; cite Harper and Konstan, 2015).

| Listing | File | What it does | Expected output |
|---|---|---|---|
| 34.1 | [`rec_data.py`](rec_data.py) | MovieLens for recommendation. | [`rec_data.log`](rec_data.log) |
| 34.2 | [`recommenders.py`](recommenders.py) | Recommenders as embedding models. | printed in the book |
| 34.3 | [`train_recommenders.py`](train_recommenders.py) | Training the recommenders and comparing with popularity. | [`train_recommenders.log`](train_recommenders.log) |
| 34.4 | [`explore.py`](explore.py) | Exploring the learned embeddings. | [`explore.log`](explore.log) |
| 34.5 | [`rec_service.py`](rec_service.py) | The recommender as a service. | printed in the book |
| 34.6 | [`try_rec_service.py`](try_rec_service.py) | Calling the recommender service. | [`try_rec_service.log`](try_rec_service.log) |

Found a problem? [Report it](../../../../issues/new/choose) and include the listing number.
