# Train both neural classifiers on 9,003 tickets and compare them with the baseline.
import json, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from triage_data import split, encode, VOCAB, INTENTS, INTENT_TO_TEAM
from text_models import BagOfEmbeddings, TicketTransformer
from wtlkit import set_seed, count_params

(t_tr, y_tr), (t_va, y_va), (t_te, y_te) = split("train"), split("val"), split("test")
X_tr, X_va, X_te = encode(t_tr), encode(t_va), encode(t_te)


def word_dropout(X, p=0.1):                              # augmentation: hide random words
    return X.masked_fill((torch.rand(X.shape) < p) & (X > 0), 1)


def train(model, epochs=30, lr=2e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * (len(X_tr) // 64 + 1))
    best, best_state = 0, None
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(X_tr)).split(64):
            loss = F.cross_entropy(model(word_dropout(X_tr[idx])), y_tr[idx], label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            acc = (model(X_va).argmax(1) == y_va).float().mean().item()
        if acc > best:                                     # keep the best epoch (validation data)
            best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.eval()


results = {}
for name, make in [("bag of embeddings", lambda: BagOfEmbeddings(len(VOCAB), 77)),
                   ("Transformer", lambda: TicketTransformer(len(VOCAB), 77))]:
    set_seed(0)
    t0 = time.perf_counter()
    model = train(make())
    with torch.no_grad():
        pred = model(X_te).argmax(1).numpy()
    y = y_te.numpy()
    results[name] = {"acc": float(np.mean(pred == y)), "team": float(np.mean(INTENT_TO_TEAM[pred] == INTENT_TO_TEAM[y]))}
    print(f"{name:18s} {count_params(model):9,} params   intent accuracy {results[name]['acc']:.2%}, "
          f"macro-F1 {f1_score(y, pred, average='macro'):.3f}, team accuracy {results[name]['team']:.2%}  "
          f"({time.perf_counter() - t0:.0f}s)")
    torch.save(model.state_dict(), f"{name.split()[0].lower()}.pt")
json.dump(results, open("triage_results.json", "w"), indent=1)
