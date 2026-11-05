# Chapter 31 · Project: A Ticket Triage Assistant

*Part VII: Building Intelligent Applications* · [All chapters](../../README.md#chapters)

Run every command from this folder, inside your activated environment.

## Before you start

Listings 31.5-31.8 load the models that Listings 31.4 and 31.6 train and save:

| File | Trained by | Training time |
|---|---|---|
| `ch31/bag.pt`, `ch31/transformer.pt` | Listing 31.4, `train_triage.py` | about 4 min on a 2-thread laptop CPU (the Transformer takes almost all of it) |
| `ch31/triage_parts.pkl`, `ch31/ticket_index.npz` | Listing 31.6, `build_triage.py` | a few minutes |

Either run the listings in order, or fetch the files with
`python scripts/fetch_artifacts.py ch31` from the `code` folder. Banking77 is downloaded
on first use by `triage_data.py`. The Transformer block comes from `wtlkit/models.py`.

| Listing | File | What it does | Expected output |
|---|---|---|---|
| 31.1 | [`triage_data.py`](triage_data.py) | The data: teams, tokenizer and splits. | printed in the book |
| 31.2 | [`baseline.py`](baseline.py) | The classical baseline. | [`baseline.log`](baseline.log) |
| 31.3 | [`text_models.py`](text_models.py) | Two neural text classifiers. | printed in the book |
| 31.4 | [`train_triage.py`](train_triage.py) | Training and comparing the neural classifiers. | [`train_triage.log`](train_triage.log) |
| 31.5 | [`triage.py`](triage.py) | The triage component. | printed in the book |
| 31.6 | [`build_triage.py`](build_triage.py) | Fitting the parts and evaluating the assistant. | [`build_triage.log`](build_triage.log) |
| 31.7 | [`triage_service.py`](triage_service.py) | The triage assistant as a web service. | printed in the book |
| 31.8 | [`try_service.py`](try_service.py) | Trying the service with new messages. | [`try_service.log`](try_service.log) |

Found a problem? [Report it](../../../../issues/new/choose) and include the listing number.
