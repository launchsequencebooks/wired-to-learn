# Chapter 32 · Project: A Log Anomaly Detector

*Part VII: Building Intelligent Applications* · [All chapters](../../README.md#chapters)

Run every command from this folder, inside your activated environment.

## Before you start

Listing 32.4 (`monitor.py`) loads one trained file:

| File | Trained by | Training time |
|---|---|---|
| `ch32/next_event_lstm.pt` | Listing 32.3, `detectors.py` | under a minute on a 2-thread laptop CPU |

Either run the listings in order, or fetch the file with
`python scripts/fetch_artifacts.py ch32` from the `code` folder. The HDFS log excerpt
is downloaded on first use by `log_data.py`.

| Listing | File | What it does | Expected output |
|---|---|---|---|
| 32.1 | [`log_data.py`](log_data.py) | HDFS logs as event sequences. | [`log_data.log`](log_data.log) |
| 32.2 | [`next_event.py`](next_event.py) | The next-event model. | printed in the book |
| 32.3 | [`detectors.py`](detectors.py) | Training and comparing the detectors. | [`detectors.log`](detectors.log) |
| 32.4 | [`monitor.py`](monitor.py) | A streaming log monitor. | [`monitor.log`](monitor.log) |

Found a problem? [Report it](../../../../issues/new/choose) and include the listing number.
