"""Download the trained files that later chapters load (P3-BOOK-001).

Chapters from Part III onward reuse networks trained earlier in the book. If you work
through the book in order those files are already on disk. If you jump in later, run
this once from the code folder:

    python scripts/fetch_artifacts.py          # every trained file (about 40 MB)
    python scripts/fetch_artifacts.py ch30     # only what Chapter 30 needs
    python scripts/fetch_artifacts.py --list   # show what would be fetched

Files that already exist are skipped. Each file is downloaded from the GitHub release
RELEASE of the companion repository; the table below says which chapter trains it and
roughly how long that takes on the laptop CPU (2 threads) used for the book's results, so you
can always choose to train instead of download.
"""
import argparse
import sys
import urllib.request
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
REPO = "https://github.com/launchsequencebooks/wired-to-learn"
RELEASE = "v1.0-artifacts"
BASE = f"{REPO}/releases/download/{RELEASE}/"

# (path under code/, chapter that trains it, script, training time)
ARTIFACTS = [
    ("ch09/digit_mlp.npz",          "ch09", "mlp_digits.py",          "about 1 min"),
    ("ch16/fashion_resnet20.pt",    "ch16", "fashion_resnet.py",      "about 10 min"),
    ("ch18/mini_gpt.pt",            "ch18", "mini_gpt.py",            "about 8 min"),
    ("ch19/autoencoder.pt",         "ch19", "autoencoder.py",         "about 1 min"),
    ("ch23/simclr_encoder.pt",      "ch23", "simclr.py",              "about 13 min"),
    ("ch30/outlier_reference.npz",  "ch30", "build_components.py",    "under a minute"),
    ("ch31/bag.pt",                 "ch31", "train_triage.py",        "about 4 min"),
    ("ch31/transformer.pt",         "ch31", "train_triage.py",        "about 4 min"),
    ("ch31/triage_parts.pkl",       "ch31", "build_triage.py",        "under a minute"),
    ("ch31/ticket_index.npz",       "ch31", "build_triage.py",        "under a minute"),
    ("ch32/next_event_lstm.pt",     "ch32", "detectors.py",           "under a minute"),
    ("ch33/mlp_forecaster.pt",      "ch33", "train_forecasters.py",   "about 2 min"),
    ("ch33/gru_forecaster.pt",      "ch33", "train_forecasters.py",   "about 2 min"),
    ("ch34/mf.pt",                  "ch34", "train_recommenders.py",  "under a minute"),
    ("ch34/mf_genres.pt",           "ch34", "train_recommenders.py",  "under a minute"),
    ("ch35/text_encoder.pt",        "ch35", "train_retrievers.py",    "under a minute"),
    ("ch35/passage_vectors.npy",    "ch35", "train_retrievers.py",    "under a minute"),
]
# Training times as in the chapter READMEs and .log files (laptop CPU, 2 threads).

# Which chapters load which files (from the torch.load / np.load calls in code/).
# A chapter's own artifacts are included so that `fetch_artifacts.py ch31` also lets
# you skip training ch31's models.
NEEDS = {
    "ch10": ["ch09/digit_mlp.npz"],
    "ch25": ["ch18/mini_gpt.pt"],
    "ch27": ["ch16/fashion_resnet20.pt"],
    "ch28": ["ch16/fashion_resnet20.pt"],
    "ch30": ["ch16/fashion_resnet20.pt", "ch23/simclr_encoder.pt",
             "ch30/outlier_reference.npz"],   # ch30 retrains its own autoencoder on clothes
    "ch31": ["ch31/bag.pt", "ch31/transformer.pt", "ch31/triage_parts.pkl",
             "ch31/ticket_index.npz"],
    "ch32": ["ch32/next_event_lstm.pt"],
    "ch33": ["ch33/mlp_forecaster.pt", "ch33/gru_forecaster.pt"],
    "ch34": ["ch34/mf.pt", "ch34/mf_genres.pt"],
    "ch35": ["ch35/text_encoder.pt", "ch35/passage_vectors.npy"],
    "ch36": ["ch16/fashion_resnet20.pt", "ch31/bag.pt", "ch31/triage_parts.pkl",
             "ch31/ticket_index.npz"],
}


def wanted(chapters):
    if not chapters:
        return [a for a, *_ in ARTIFACTS]
    out = []
    for ch in chapters:
        if ch not in NEEDS:
            sys.exit(f"{ch}: no trained files are needed (or the chapter is unknown)")
        out += [f for f in NEEDS[ch] if f not in out]
    return out


def fetch(rel: str, info: dict, dry: bool) -> None:
    dest = CODE / rel
    _, trained_by, script, minutes = info[rel]
    if dest.exists():
        print(f"have   {rel}  ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    # release assets are flat: ch16__fashion_resnet20.pt
    url = BASE + rel.replace("/", "__")
    if dry:
        print(f"would  {rel}  <- {url}")
        print(f"       (or train: {trained_by}/{script}, {minutes})")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetch  {rel}  ...", end="", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
            f.write(r.read())
    except Exception as e:  # noqa: BLE001 - report and keep going
        print(f" FAILED ({e})")
        print(f"       train it instead: cd {trained_by} && python {script}"
              f"  ({minutes})")
        if dest.exists():
            dest.unlink()
        return
    size = dest.stat().st_size / 1e6
    print(f" {size:.1f} MB  (trained by {trained_by}/{script}, {minutes})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("chapters", nargs="*",
                    help="chapter folders, e.g. ch30 ch36 (default: all)")
    ap.add_argument("--list", action="store_true",
                    help="show what would be fetched, download nothing")
    args = ap.parse_args(argv)
    info = {a[0]: a for a in ARTIFACTS}
    for rel in wanted(args.chapters):
        fetch(rel, info, args.list)


if __name__ == "__main__":
    main()
