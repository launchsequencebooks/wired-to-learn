"""Two neural forecasters that predict three quantiles for each of the next 24 hours."""
import torch
from torch import nn
from demand_data import PAST, AHEAD

QUANTILES = torch.tensor([0.1, 0.5, 0.9])                 # an 80% interval and the median


def pinball_loss(pred, target):
    """The quantile loss: under-predicting costs q per unit, over-predicting costs (1 - q)."""
    err = target[..., None] - pred                         # (batch, 24, 3)
    return torch.maximum(QUANTILES * err, (QUANTILES - 1) * err).mean()


class MLPForecaster(nn.Module):
    """Flatten the week of history and tomorrow's known features into one vector."""

    def __init__(self, n_feat=9, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(PAST + AHEAD * n_feat, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, AHEAD * 3))

    def forward(self, past, future):
        x = torch.cat([past[:, :, 0], future.flatten(1)], 1)       # counts only, plus the known future
        return self.net(x).view(-1, AHEAD, 3).sort(-1).values      # sorted: quantiles must not cross


class GRUForecaster(nn.Module):
    """A GRU (Chapter 17) reads the week hour by hour; an MLP combines its summary with the known future."""

    def __init__(self, n_feat=9, hidden=64):
        super().__init__()
        self.gru = nn.GRU(1 + n_feat, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + AHEAD * n_feat + PAST // 7, 256), nn.ReLU(),
                                  nn.Linear(256, AHEAD * 3))

    def forward(self, past, future):
        _, h = self.gru(past)
        same_day_last_week = past[:, :24, 0]                       # a direct shortcut to the weekly pattern
        x = torch.cat([h[-1], future.flatten(1), same_day_last_week], 1)
        return self.head(x).view(-1, AHEAD, 3).sort(-1).values
