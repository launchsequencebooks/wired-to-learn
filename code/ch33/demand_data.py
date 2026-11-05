"""Hourly bike rentals as forecasting windows: 7 days of history -> the next 24 hours."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from wtlkit import load_bike

PAST, AHEAD = 168, 24                                     # one week of history, one day ahead
raw = load_bike()
raw["time"] = raw["dteday"] + pd.to_timedelta(raw["hr"], unit="h")
df = raw.set_index("time").reindex(pd.date_range(raw["time"].min(), raw["time"].max(), freq="h"))
MISSING_MASK = df["cnt"].isna().values                    # hours absent from the records
MISSING = int(MISSING_MASK.sum())
df["cnt"] = df["cnt"].fillna(0)                           # an unrecorded hour had no rentals
df[["temp", "hum", "windspeed", "weathersit"]] = df[["temp", "hum", "windspeed", "weathersit"]].ffill()
df["holiday"] = df["holiday"].ffill()
df["workingday"] = ((df.index.dayofweek < 5) & (df["holiday"] == 0)).astype(float)
TIME = df.index


def calendar_and_weather(d):
    """Known-in-advance features for each hour: time of day and week, working day, weather (forecast)."""
    h, w = d.index.hour.values, d.index.dayofweek.values
    return np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                     np.sin(2 * np.pi * w / 7), np.cos(2 * np.pi * w / 7), d["workingday"].values,
                     d["temp"].values, d["hum"].values, d["windspeed"].values,
                     (d["weathersit"].values >= 3).astype(float)], axis=1).astype(np.float32)   # rain or snow


LOG_CNT = np.log1p(df["cnt"].values).astype(np.float32)   # log scale: errors become relative errors
FEATS = calendar_and_weather(df)


def windows(start, end, step=1):
    """Forecast origins t in [start, end): inputs from the week before t, targets the 24 hours after."""
    t0, t1 = TIME.searchsorted(pd.Timestamp(start)), TIME.searchsorted(pd.Timestamp(end))
    origins = np.arange(max(t0, PAST), t1 - AHEAD + 1, step)
    past = np.stack([np.c_[LOG_CNT[t - PAST:t], FEATS[t - PAST:t]] for t in origins])
    future = np.stack([FEATS[t:t + AHEAD] for t in origins])
    target = np.stack([LOG_CNT[t:t + AHEAD] for t in origins])
    return torch.tensor(past), torch.tensor(future), torch.tensor(target), origins


TRAIN = ("2011-01-01", "2012-07-01")
VAL = ("2012-07-01", "2012-10-01")
TEST = ("2012-10-01", "2013-01-01")

if __name__ == "__main__":
    print(f"{len(df):,} hours from {TIME[0].date()} to {TIME[-1].date()} "
          f"({MISSING} unrecorded hours, counted as 0 rentals)")
    # do the gaps look like 'no rentals'? (night hours, and a few whole days)
    night = int((TIME[MISSING_MASK].hour <= 5).sum())
    per_day = pd.Series(MISSING_MASK, index=TIME).resample("D").sum()
    quiet_days = int((per_day >= 12).sum())
    print(f"{night} of the {MISSING} unrecorded hours are between midnight and 5 a.m.; "
          f"{quiet_days} days are missing 12 hours or more")
    print(f"average rentals per hour: 2011 {df.loc['2011', 'cnt'].mean():.0f}, 2012 {df.loc['2012', 'cnt'].mean():.0f}")
    for name, (a, b) in [("train", TRAIN), ("validation", VAL), ("test", TEST)]:
        print(f"{name:10s} forecasts made at midnight: {len(windows(a, b, step=24)[3]):4d}  "
              f"(and {len(windows(a, b)[3]):,} when made at every hour)")
