"""Answer questions about the book: retrieve, pick the best sentences, cite, or admit ignorance."""
import re
import numpy as np
import torch
from corpus import passages
from retrieval import BM25, TextEncoder, tokens


def readable(sentence):
    """Skip sentences mangled by removing their formulas (many one-letter leftovers)."""
    words = sentence.split()
    return sum(len(w.strip(".,;:()")) == 1 for w in words) / len(words) < 0.15


class BookQA:
    def __init__(self, min_match=1.2):
        self.P = passages()
        self.bm25 = BM25([p["text"] for p in self.P])
        saved = torch.load("text_encoder.pt")
        self.enc = TextEncoder(saved["vocab"]); self.enc.load_state_dict(saved["state"]); self.enc.eval()
        self.vectors = np.load("passage_vectors.npy")
        self.min_match = min_match

    def retrieve(self, question, k=3):
        with torch.no_grad():
            sim = self.enc([question]).numpy()[0] @ self.vectors.T
        r1 = np.argsort(np.argsort(-self.bm25.scores(question))) + 1
        r2 = np.argsort(np.argsort(-sim)) + 1
        top = np.argsort(-(1 / (60 + r1) + 1 / (60 + r2)))[:k]   # hybrid ranking (rank fusion)
        match = self.bm25.scores(question).max() / max(1, len(tokens(question)))   # word evidence per query word
        return top, match

    def answer(self, question, n_sentences=2):
        top, confidence = self.retrieve(question)
        if confidence < self.min_match:                       # too few of the question's words are in the book
            return {"answer": "I could not find this in the book.", "sources": [], "confidence": round(float(confidence), 2)}
        sentences = [(s, i, rank) for rank, i in enumerate(top)
                     for s in re.split(r"(?<=[.!?])\s+", self.P[i]["text"]) if 6 <= len(s.split()) <= 60
                     and readable(s)]
        bm = BM25([s for s, _, _ in sentences])
        with torch.no_grad():
            sim = (self.enc([question]) @ self.enc([s for s, _, _ in sentences]).T).numpy()[0]
        prior = np.array([0.5 / (1 + rank) for _, _, rank in sentences])     # prefer the best passage
        score = bm.scores(question) / (bm.scores(question).max() + 1e-9) + sim + prior   # words, meaning, rank
        best = sorted(np.argsort(-score)[:n_sentences])                          # keep the book's order
        cite = lambda i: f"Chapter {self.P[i]['chapter']}, \"{self.P[i]['section']}\""
        return {"answer": " ".join(sentences[j][0] for j in best),
                "sources": list(dict.fromkeys(cite(sentences[j][1]) for j in best)),
                "confidence": round(float(confidence), 2)}
