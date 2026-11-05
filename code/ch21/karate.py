# Zachary's karate club: a GCN predicts which faction each member joins,
# given the label of only one member of each faction.
import json
import networkx as nx
import torch
from torch import nn
from gcn import GCN, normalized_adjacency

G = nx.karate_club_graph()
n = G.number_of_nodes()
y = torch.tensor([0 if G.nodes[i]["club"] == "Mr. Hi" else 1 for i in range(n)])
edges = torch.tensor(list(G.edges())).T
edge_index = torch.cat([edges, edges.flip(0)], 1)         # both directions
A_norm = normalized_adjacency(edge_index, n)
X = torch.eye(n)                                          # no features: each node is just "itself"
labeled = torch.tensor([0, 33])                           # the instructor and the administrator
print(f"{n} members, {edges.shape[1]} friendships, labels known for members {labeled.tolist()}")


def run(seed):
    """Train from one random initialization; return the prediction and its accuracy."""
    torch.manual_seed(seed)
    model = GCN(n, 16, 2, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for epoch in range(1, 201):
        out = model(X, A_norm)
        loss = nn.functional.cross_entropy(out[labeled], y[labeled])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = model(X, A_norm).argmax(1)
    return pred, (pred == y).float().mean().item()


pred, acc = run(0)                                  # seed 0 is the run in the figure
print(f"accuracy on all {n} members: {acc:.0%} (from 2 labels)")
print("misclassified members:", (pred != y).nonzero().squeeze(1).tolist())
accs = sorted(run(s)[1] for s in range(10))         # how much does the seed matter?
median = (accs[4] + accs[5]) / 2
print(f"accuracy over 10 seeds: min {min(accs):.0%}, median {median:.0%}, "
      f"max {max(accs):.0%}")
json.dump({"pos": {i: list(p) for i, p in nx.spring_layout(G, seed=3).items()},
           "edges": edges.T.tolist(), "y": y.tolist(), "pred": pred.tolist()},
          open("karate.json", "w"))
