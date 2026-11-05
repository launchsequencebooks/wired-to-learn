"""The triage component: two models vote, a router decides, an index finds similar tickets."""
import pickle
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
import torch
from triage_data import encode, VOCAB, INTENTS, TEAM_NAMES, INTENT_TO_TEAM
from text_models import BagOfEmbeddings
from wtlapps import EmbeddingIndex, ConfidenceRouter


class Triage:
    def __init__(self):
        self.net = BagOfEmbeddings(len(VOCAB), 77)
        self.net.load_state_dict(torch.load(HERE / "bag.pt")); self.net.eval()
        with open(HERE / "triage_parts.pkl", "rb") as f:
            parts = pickle.load(f)
        self.vectorizer, self.logreg = parts["vectorizer"], parts["logreg"]
        self.router = ConfidenceRouter(parts["threshold"])
        self.index = EmbeddingIndex.load(HERE / "ticket_index.npz")
        self.past = parts["past_tickets"]

    @torch.no_grad()
    def intent_probs(self, texts):
        p_net = self.net(encode(texts)).softmax(1).numpy()
        p_words = self.logreg.predict_proba(self.vectorizer.transform(texts))
        return (p_net + p_words) / 2                        # an ensemble of a network and a classical model

    def team_probs(self, p_intent):
        p_team = np.zeros((len(p_intent), len(TEAM_NAMES)))
        np.add.at(p_team.T, INTENT_TO_TEAM, p_intent.T)    # a team's probability = sum over its intents
        return p_team

    @torch.no_grad()
    def __call__(self, texts, k=3):
        p_intent = self.intent_probs(texts)
        p_team = self.team_probs(p_intent)
        ids, sims = self.index.search(self.net.embed(encode(texts)).numpy(), k=k)
        return [{"team": TEAM_NAMES[pt.argmax()], "team_confidence": round(float(pt.max()), 3),
                 "intent": INTENTS[pi.argmax()], "route": self.router.route(pt.max()).item(),
                 "similar_past_tickets": [self.past[j] for j in row]}
                for pt, pi, row in zip(p_team, p_intent, ids)]
