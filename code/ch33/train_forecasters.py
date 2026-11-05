# Train both forecasters and compare them with the seasonal-naive baseline on Oct-Dec 2012.
import json, time
import numpy as np
import torch
from demand_data import windows, TRAIN, VAL, TEST, TIME, MISSING_MASK
from forecasters import MLPForecaster, GRUForecaster, pinball_loss
from wtlkit import set_seed

P_tr, F_tr, Y_tr, _ = windows(*TRAIN)                      # a forecast made at every hour
P_va, F_va, Y_va, _ = windows(*VAL, step=24)               # forecasts made at midnight, for the next day
P_te, F_te, Y_te, origins = windows(*TEST, step=24)
rentals = lambda log_counts: torch.expm1(log_counts).clamp(min=0)
# hours absent from the records are not scored: their 0 is a gap, not a measurement
keep = ~torch.tensor(MISSING_MASK)[origins[:, None] + np.arange(24)]


def report(name, q):                                       # q: (days, 24, 3) quantiles, log scale
    pred, lo, hi, actual = rentals(q[..., 1]), rentals(q[..., 0]), rentals(q[..., 2]), rentals(Y_te)
    mae = (pred - actual).abs()[keep].mean().item()
    inside = ((actual >= lo) & (actual <= hi))[keep].float().mean().item()
    coverage = f"{inside:6.1%}" if (hi > lo).any() else "    --"           # the baseline gives no interval
    print(f"{name:28s} MAE {mae:5.1f} rentals/hour   80% interval contains the truth {coverage}")
    return mae


naive = P_te[:, :24, 0]                                    # same hours one week earlier (the 'seasonal naive')
results = {"seasonal naive (last week)": report("seasonal naive (last week)", torch.stack([naive] * 3, -1))}
forecasts = {}
for name, Model in [("MLP", MLPForecaster), ("GRU", GRUForecaster)]:
    set_seed(0)
    model, t0 = Model(), time.perf_counter()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state = float("inf"), None
    for epoch in range(25):
        model.train()
        for idx in torch.randperm(len(P_tr)).split(128):
            loss = pinball_loss(model(P_tr[idx], F_tr[idx]), Y_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val = pinball_loss(model(P_va, F_va), Y_va).item()
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    with torch.no_grad():
        q = model(P_te, F_te)
    results[name] = report(f"{name} (quantiles, {time.perf_counter() - t0:.0f}s)", q)
    forecasts[name] = q.numpy().tolist()
    torch.save(model.state_dict(), f"{name.lower()}_forecaster.pt")
    day_err = ((rentals(q[..., 1]) - rentals(Y_te)).abs() * keep).sum(1) / keep.sum(1).clamp(min=1)
worst = day_err.topk(3).indices                            # the GRU's worst test days
print("worst GRU days:", ", ".join(str(TIME[origins[i]].date()) for i in worst))
json.dump({"mae": results, "origins": [int(o) for o in origins], "forecasts": forecasts}, open("forecasts.json", "w"))
