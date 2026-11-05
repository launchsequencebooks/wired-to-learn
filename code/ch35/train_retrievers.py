# Train a neural retriever on the book with contrastive learning, and compare it with BM25.
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from corpus import passages
from questions import QUESTIONS
from retrieval import BM25, TextEncoder, tokens
from wtlkit.losses import nt_xent              # the contrastive loss of Chapter 23
from wtlkit import set_seed

P = passages()
texts = [p["text"] for p in P]
bm25 = BM25(texts)
vocab = sorted({w for t in texts for w in tokens(t)})


def two_views(p):
    """Two different random spans of the same passage: they should get similar vectors."""
    w = p["text"].split()
    a, b = random.randrange(0, len(w) // 2), random.randrange(0, len(w) // 2)
    return " ".join(w[a:a + 25]), " ".join(w[b:b + 25])


set_seed(0); random.seed(0)
enc = TextEncoder(vocab)
opt = torch.optim.Adam(enc.parameters(), lr=3e-3)
for step in range(3000):
    batch = random.sample(P, 64)
    views = [two_views(p) for p in batch]
    a, b = zip(*views)
    if step % 2:                                           # every other step: section title vs passage text
        a = [f"{p['chapter_title']} {p['section']}" for p in batch]
    loss = nt_xent(enc(list(a)), enc(list(b)), temperature=0.1)
    opt.zero_grad(); loss.backward(); opt.step()
enc.eval()
torch.save({"state": enc.state_dict(), "vocab": vocab}, "text_encoder.pt")
with torch.no_grad():
    doc_vecs = enc(texts).numpy()
np.save("passage_vectors.npy", doc_vecs)


def ranks(scores_per_query):
    """1-based rank of the first passage from the correct section, for every question."""
    out = []
    for s, (_, ch, sec) in zip(scores_per_query, QUESTIONS):
        order = np.argsort(-s)
        out.append(next(r for r, i in enumerate(order, 1) if P[i]["chapter"] == ch and P[i]["section"] == sec))
    return np.array(out)


with torch.no_grad():
    q_vecs = enc([q for q, _, _ in QUESTIONS]).numpy()
results = {"BM25": ranks([bm25.scores(q) for q, _, _ in QUESTIONS]),
           "neural encoder": ranks(q_vecs @ doc_vecs.T)}
rrf = lambda r: 1 / (60 + r)                               # reciprocal rank fusion of the two rankings
fused = []
for q, qv in zip(QUESTIONS, q_vecs):
    r1 = np.argsort(np.argsort(-bm25.scores(q[0]))) + 1
    r2 = np.argsort(np.argsort(-(qv @ doc_vecs.T))) + 1
    fused.append(rrf(r1) + rrf(r2))
results["hybrid (rank fusion)"] = ranks(fused)
print(f"{len(P)} passages, {len(QUESTIONS)} test questions (right section must be found)")
report = lambda name, r: print(f"{name:22s} top-1 {np.mean(r <= 1):5.1%}   top-5 {np.mean(r <= 5):5.1%}"
                               f"   MRR {np.mean(1 / r):.3f}")
for name, r in results.items():
    report(name, r)

# Optional: the kind of retriever you would deploy, pretrained on billions of sentence pairs.
try:
    from sentence_transformers import SentenceTransformer       # pip install sentence-transformers
except ImportError:
    SentenceTransformer = None
if SentenceTransformer is not None:
    st = SentenceTransformer("all-MiniLM-L6-v2")                 # 22M parameters, about 90 MB
    d_vecs = st.encode(texts, normalize_embeddings=True)
    q_vecs2 = st.encode([q for q, _, _ in QUESTIONS], normalize_embeddings=True)
    sims = q_vecs2 @ d_vecs.T
    report("pretrained MiniLM", ranks(sims))
    fused2 = [rrf(np.argsort(np.argsort(-bm25.scores(q[0]))) + 1) + rrf(np.argsort(np.argsort(-s)) + 1)
              for q, s in zip(QUESTIONS, sims)]
    report("MiniLM + BM25 (fusion)", ranks(fused2))
