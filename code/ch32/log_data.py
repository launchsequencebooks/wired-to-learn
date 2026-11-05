"""HDFS logs as event sequences: one sequence of event types per storage block."""
import re
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from wtlkit import load_hdfs

lines, LABELS = load_hdfs()
BLOCK = re.compile(r"blk_-?\d+")


def sequences():
    """{block_id: [event ids in order]}, blocks in order of first appearance."""
    seqs = defaultdict(list)
    for line in lines:
        for b in dict.fromkeys(BLOCK.findall(line["Content"])):
            seqs[b].append(line["EventId"])
    return dict(seqs)


SEQS = sequences()
BLOCKS = list(SEQS)                                       # chronological: by first log line
half = len(BLOCKS) // 2
TRAIN_BLOCKS = [b for b in BLOCKS[:half] if not LABELS[b]]   # learn what normal looks like
cut = len(TRAIN_BLOCKS) * 4 // 5
TRAIN_BLOCKS, VAL_BLOCKS = TRAIN_BLOCKS[:cut], TRAIN_BLOCKS[cut:]
TEST_BLOCKS = BLOCKS[half:]                               # later blocks, normal and anomalous
EVENTS = ["<unseen>"] + sorted({e for b in TRAIN_BLOCKS for e in SEQS[b]})
EVENT_ID = {e: i for i, e in enumerate(EVENTS)}


def ids(block):
    return [EVENT_ID.get(e, 0) for e in SEQS[block]]     # events never seen in training -> 0


def counts(blocks):
    """Bag-of-events vectors: how often each event type occurs in each block."""
    m = np.zeros((len(blocks), len(EVENTS)), dtype=np.float32)
    for i, b in enumerate(blocks):
        np.add.at(m[i], ids(b), 1)
    return m


if __name__ == "__main__":
    n_anom = sum(LABELS[b] for b in TEST_BLOCKS)
    print(f"{len(lines):,} log lines, {len(BLOCKS):,} blocks, {sum(LABELS[b] for b in BLOCKS)} labeled anomalous")
    print(f"training on {len(TRAIN_BLOCKS):,} normal blocks ({len(VAL_BLOCKS)} for validation); "
          f"testing on {len(TEST_BLOCKS):,} later blocks, {n_anom} anomalous")
    print(f"{len(EVENTS) - 1} event types seen in training; a typical normal block, as event ids:",
          ids(next(b for b in TRAIN_BLOCKS if len(SEQS[b]) == 13)))
    print("one log line:", lines[0]["Content"][:70], "->", lines[0]["EventId"])
