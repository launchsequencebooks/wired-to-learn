# Forecasting tomorrow's minimum temperature in Melbourne from the past 30 days.
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from wtlkit import load_temperatures, set_seed

dates, temps = load_temperatures()
years = np.array([int(d[:4]) for d in dates])
WINDOW = 30
mean, std = temps[years <= 1988].mean(), temps[years <= 1988].std()   # training statistics
z = (temps - mean) / std

X = np.stack([z[i:i + WINDOW] for i in range(len(z) - WINDOW)])        # 30 past days
y = z[WINDOW:]                                                         # the next day
year_of_target = years[WINDOW:]
split = {"train": year_of_target <= 1988, "val": year_of_target == 1989, "test": year_of_target == 1990}
data = {k: (torch.tensor(X[m]).unsqueeze(-1), torch.tensor(y[m])) for k, m in split.items()}
print({k: len(v[1]) for k, v in data.items()}, "windows of", WINDOW, "days")


class Recurrent(nn.Module):
    def __init__(self, cell, hidden=32):
        super().__init__()
        self.rnn = cell(1, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):                    # x: (batch, 30, 1)
        h, _ = self.rnn(x)
        return self.out(h[:, -1]).squeeze(-1)   # predict from the final hidden state


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(WINDOW, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def mae_celsius(pred, target):
    return ((pred - target).abs().mean() * std).item()


def train(model, epochs=40):
    set_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loader = DataLoader(TensorDataset(*data["train"]), batch_size=64, shuffle=True)
    best, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            nn.functional.mse_loss(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = mae_celsius(model(data["val"][0]), data["val"][1])
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    with torch.no_grad():
        return mae_celsius(model(data["test"][0]), data["test"][1]), model


Xt, yt = data["test"]
results = {"persistence (tomorrow = today)": mae_celsius(Xt[:, -1, 0], yt),
           "7-day average": mae_celsius(Xt[:, -7:, 0].mean(1), yt)}
models = {}
for name, make in [("MLP on the window", MLP), ("RNN", lambda: Recurrent(nn.RNN)),
                   ("LSTM", lambda: Recurrent(nn.LSTM)), ("GRU", lambda: Recurrent(nn.GRU))]:
    set_seed(0)
    results[name], models[name] = train(make())
print(f"{'model':32s}{'test MAE (deg C)':>17s}")
for name, mae in results.items():
    print(f"{name:32s}{mae:>17.3f}")
with torch.no_grad():
    pred = models["LSTM"](Xt).numpy() * std + mean
json.dump({"results": results, "pred": pred.tolist(), "true": (yt.numpy() * std + mean).tolist()},
          open("forecast.json", "w"))
