# Chapter 33 · Project: A Demand-Forecasting Dashboard

*Part VII: Building Intelligent Applications* · [All chapters](../../README.md#chapters)

Run every command from this folder, inside your activated environment.

## Before you start

Listing 33.4 (`dashboard.py`, run with `streamlit run dashboard.py`) loads two trained files:

| File | Trained by | Training time |
|---|---|---|
| `ch33/mlp_forecaster.pt` | Listing 33.3, `train_forecasters.py` | seconds on a 2-thread laptop CPU |
| `ch33/gru_forecaster.pt` | Listing 33.3, `train_forecasters.py` | about 2 minutes on a 2-thread laptop CPU |

Either run the listings in order, or fetch both files with
`python scripts/fetch_artifacts.py ch33` from the `code` folder. The hourly rentals
data is downloaded from the UCI Machine Learning Repository on first use by
`demand_data.py`.

| Listing | File | What it does | Expected output |
|---|---|---|---|
| 33.1 | [`demand_data.py`](demand_data.py) | Forecasting windows from hourly rentals. | [`demand_data.log`](demand_data.log) |
| 33.2 | [`forecasters.py`](forecasters.py) | Two quantile forecasters. | printed in the book |
| 33.3 | [`train_forecasters.py`](train_forecasters.py) | Training and comparing the forecasters. | [`train_forecasters.log`](train_forecasters.log) |
| 33.4 | [`dashboard.py`](dashboard.py) | The forecasting dashboard. | printed in the book |

Found a problem? [Report it](../../../../issues/new/choose) and include the listing number.
