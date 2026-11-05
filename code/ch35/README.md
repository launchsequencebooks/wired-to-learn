# Chapter 35 · Project: Question Answering over Documents

*Part VII: Building Intelligent Applications* · [All chapters](../../README.md#chapters)

Run every command from this folder, inside your activated environment.

## Before you start

Listings 35.4 (`qa.py`) to 35.7 (`generate.py`) load two files that this chapter's own
training script writes:

| File | Trained by | Training time |
|---|---|---|
| `ch35/text_encoder.pt`, `ch35/passage_vectors.npy` | Listing 35.3, `train_retrievers.py` | under a minute on a 2-thread laptop CPU |

Either run the listings in order, or fetch the files with
`python scripts/fetch_artifacts.py ch35` from the `code` folder. Listing 35.7 also
downloads a language model of about 1 GB on first use (needs `transformers`).

| Listing | File | What it does | Expected output |
|---|---|---|---|
| 35.1 | [`corpus.py`](corpus.py) | The book as a corpus of citable passages. | [`corpus.log`](corpus.log) |
| 35.2 | [`retrieval.py`](retrieval.py) | BM25 and a neural text encoder. | printed in the book |
| 35.3 | [`train_retrievers.py`](train_retrievers.py) | Training the neural retriever and comparing all three (plus a pretrained MiniLM if `sentence-transformers` is installed). | [`train_retrievers.log`](train_retrievers.log) |
| 35.4 | [`qa.py`](qa.py) | The question-answering component. | printed in the book |
| 35.5 | [`ask.py`](ask.py) | Asking the book. | [`ask.log`](ask.log) |
| 35.6 | [`prompt.py`](prompt.py) | Building a grounded prompt from the retrieved passages. | [`prompt.log`](prompt.log) |
| 35.7 | [`generate.py`](generate.py) | Retrieval-augmented generation with a small open instruct model (needs `transformers`; about 1 GB download). | [`generate.log`](generate.log) |
| 35.8 | [`qa_app.py`](qa_app.py) | A chat interface for the book. | printed in the book |

**Supporting files:** [`questions.py`](questions.py), [`data/passages.json`](data/passages.json)

**The corpus.** The documents are the book's LaTeX sources, which are not part of this
repository. The repository therefore ships the result of Listing 35.1, `data/passages.json`
(derived from the book's text for use with the companion code; not for redistribution outside
this repository). `passages()` in `corpus.py` loads it whenever the `chapters/` folder is
absent, so every later listing runs from it. Run `python train_retrievers.py` once first: it
writes `text_encoder.pt` and `passage_vectors.npy`, which `qa.py`, `ask.py`, `prompt.py`,
`generate.py` and `qa_app.py` load.

Found a problem? [Report it](../../../../issues/new/choose) and include the listing number.
