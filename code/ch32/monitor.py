# A streaming log monitor: check every log line the moment it arrives.
import time
from collections import defaultdict, deque
import numpy as np
import torch
from log_data import lines, LABELS, BLOCK, EVENT_ID, TEST_BLOCKS
from log_data import EVENTS
from next_event import NextEvent, START, H, G


class LogMonitor:
    def __init__(self, model):
        self.model = model
        self.history = defaultdict(lambda: deque([START] * H, maxlen=H))   # last H events of each block
        self.alerted = set()

    @torch.no_grad()
    def on_line(self, line):
        """Returns an alert (a dict) if this line is surprising for its block, else None."""
        event = EVENT_ID.get(line["EventId"], 0)
        alert = None
        for block in dict.fromkeys(BLOCK.findall(line["Content"])):
            h = self.history[block]
            top = self.model(torch.tensor([list(h)])).topk(G).indices[0].tolist()
            if (event == 0 or event not in top) and block not in self.alerted:
                self.alerted.add(block)                    # one alert per block, not one per line
                alert = {"block": block, "event": line["EventId"], "time": line["Time"],
                         "expected": [EVENTS[i] for i in top if i < len(EVENTS)], "text": line["Content"][:60]}
            h.append(event)
        return alert


model = NextEvent(len(EVENTS))
model.load_state_dict(torch.load("next_event_lstm.pt")); model.eval()
monitor = LogMonitor(model)
test = set(TEST_BLOCKS)
first_seen = {}
stream = [l for l in lines if any(b in test for b in BLOCK.findall(l["Content"]))]   # the later half of the log
alerts, events_before_alert = [], []
seen = defaultdict(int)
t0 = time.perf_counter()
for line in stream:
    for b in dict.fromkeys(BLOCK.findall(line["Content"])):
        seen[b] += 1
    a = monitor.on_line(line)
    if a:
        alerts.append(a); events_before_alert.append(seen[a["block"]])
rate = len(stream) / (time.perf_counter() - t0)
true = [LABELS[a["block"]] for a in alerts]
n_anomalous = sum(LABELS[b] for b in test)
print(f"replayed {len(stream):,} log lines ({rate:,.0f} lines/s); {len(alerts)} alerts, "
      f"{sum(true)} on anomalous blocks ({sum(true)}/{n_anomalous} anomalous blocks caught)")
print(f"median position of the alerting line within its block: event {int(np.median(events_before_alert))}")
for a in alerts[:2]:
    print(f"ALERT {a['time']} {a['block']}: {a['event']} '{a['text']}...' (expected one of {a['expected']})")
