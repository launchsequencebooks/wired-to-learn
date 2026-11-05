# Fit the word-count model, choose the routing threshold, index past tickets, and evaluate.
import json, pickle
from collections import Counter
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from triage_data import split, encode, VOCAB, INTENTS, INTENT_TO_TEAM
from text_models import BagOfEmbeddings
from wtlapps import EmbeddingIndex, ConfidenceRouter

(t_tr, y_tr), (t_va, y_va), (t_te, y_te) = split("train"), split("val"), split("test")
vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
logreg = LogisticRegression(C=20, max_iter=3000).fit(vec.fit_transform(t_tr), y_tr.numpy())
parts = {"vectorizer": vec, "logreg": logreg, "threshold": 0.0, "past_tickets": t_tr}
pickle.dump(parts, open("triage_parts.pkl", "wb"))

net = BagOfEmbeddings(len(VOCAB), 77)
net.load_state_dict(torch.load("bag.pt")); net.eval()
index = EmbeddingIndex(128)
with torch.no_grad():
    index.add(net.embed(encode(t_tr)).numpy(), list(range(len(t_tr))))   # every past ticket, by meaning
index.save("ticket_index.npz")

from triage import Triage                                 # now the component can load all its parts
tri = Triage()


def evaluate(texts, y):
    p_int = tri.intent_probs(texts)
    p_team = tri.team_probs(p_int)
    return p_int.argmax(1), p_team.max(1), p_team.argmax(1) == INTENT_TO_TEAM[y.numpy()]


pred, _, team_ok = evaluate(t_te, y_te)
print(f"ensemble on {len(t_te):,} test tickets: intent accuracy {np.mean(pred == y_te.numpy()):.2%}, "
      f"team accuracy {team_ok.mean():.2%}")
json.dump({"acc": float(np.mean(pred == y_te.numpy())), "team": float(team_ok.mean())}, open("ensemble_results.json", "w"))
_, conf_va, ok_va = evaluate(t_va, y_va)
parts["threshold"] = ConfidenceRouter().fit(conf_va, ok_va, target_accuracy=0.99).threshold
pickle.dump(parts, open("triage_parts.pkl", "wb"))
_, conf_te, ok_te = evaluate(t_te, y_te)
auto = conf_te >= parts["threshold"]
print(f"routing for 99% team accuracy: threshold {parts['threshold']:.3f}, {auto.mean():.1%} routed automatically "
      f"at {ok_te[auto].mean():.2%}; {(~auto).sum()} tickets to a dispatcher")
pairs = Counter((INTENTS[t], INTENTS[p]) for t, p in zip(y_te.numpy(), pred) if t != p)
print("most frequent intent mix-ups (true -> predicted):")
for (t, p), n in pairs.most_common(4):
    print(f"  {n:2d} x  {t} -> {p}")
