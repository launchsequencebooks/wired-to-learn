"""Download-on-first-use loaders for the datasets of Parts IV-VII.
Files are stored in code/data/. Sources are public GitHub repositories, except MovieLens
(GroupLens) and the bike-sharing data (UCI), which come from their official hosts."""
import gzip
import pickle
import sys
import urllib.request
import zipfile
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"
RAW = "https://raw.githubusercontent.com/"
SOURCES = {
    "fashion": RAW + "zalandoresearch/fashion-mnist/master/data/fashion/",
    "shakespeare.txt": RAW + "karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "daily-min-temperatures.csv": RAW + "jbrownlee/Datasets/master/daily-min-temperatures.csv",
    "cora": RAW + "kimiyoung/planetoid/master/data/",
    "alice.txt": RAW + "GITenberg/Alice-s-Adventures-in-Wonderland_11/master/11.txt",
    "banking77": RAW + "PolyAI-LDN/task-specific-datasets/master/banking_data/",
    "hdfs": RAW + "logpai/loglizer/master/data/HDFS/",
    "movielens": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
    "bike": "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
}
FASHION_CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def _fetch(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {path.name} ...", file=sys.stderr)
        urllib.request.urlretrieve(url, path)
    return path


def _unzip(url, archive, members):
    """Download a zip archive once and extract the named members beside it."""
    if not all((archive.parent / Path(m).name).exists() for m in members):
        with zipfile.ZipFile(_fetch(url, archive)) as z:
            for m in members:
                (archive.parent / Path(m).name).write_bytes(z.read(m))
    return [archive.parent / Path(m).name for m in members]


def load_fashion(val_size=10_000):
    """Fashion-MNIST in the same format as load_mnist(): 0..1 floats, flattened."""
    def read(name, offset):
        p = _fetch(SOURCES["fashion"] + name, DATA / "fashion" / name)
        with gzip.open(p) as f:
            return np.frombuffer(f.read(), np.uint8, offset=offset)
    X_tr = read("train-images-idx3-ubyte.gz", 16).reshape(-1, 784).astype(np.float32) / 255
    y_tr = read("train-labels-idx1-ubyte.gz", 8).astype(np.int64)
    X_te = read("t10k-images-idx3-ubyte.gz", 16).reshape(-1, 784).astype(np.float32) / 255
    y_te = read("t10k-labels-idx1-ubyte.gz", 8).astype(np.int64)
    return ((X_tr[:-val_size], y_tr[:-val_size]), (X_tr[-val_size:], y_tr[-val_size:]),
            (X_te, y_te))


def load_text(name="shakespeare.txt"):
    text = _fetch(SOURCES[name], DATA / name).read_text(encoding="utf-8-sig")
    if "*** START OF" in text:                       # strip Project Gutenberg header/footer
        text = text.split("*** START OF", 1)[1].split("\n", 1)[1]
        text = text.split("*** END OF", 1)[0]
        text = text.replace("\r\n", "\n").strip()
    return text


def load_temperatures():
    """Daily minimum temperatures in Melbourne, 1981-1990 (3,650 days)."""
    p = _fetch(SOURCES["daily-min-temperatures.csv"], DATA / "daily-min-temperatures.csv")
    rows = [l.split(",") for l in p.read_text().strip().splitlines()[1:]]
    dates = [r[0].strip('"') for r in rows]
    temps = np.array([float(r[1].strip('"?')) for r in rows], dtype=np.float32)
    return dates, temps


def load_cora():
    """The Cora citation graph (Planetoid split): features, labels, edges, splits."""
    parts = {}
    for s in ["x", "y", "tx", "ty", "allx", "ally", "graph", "test.index"]:
        p = _fetch(SOURCES["cora"] + f"ind.cora.{s}", DATA / "cora" / f"ind.cora.{s}")
        if s == "test.index":
            parts[s] = np.array([int(l) for l in p.read_text().split()])
        else:
            with open(p, "rb") as f:
                parts[s] = pickle.load(f, encoding="latin1")
    test_idx = parts["test.index"]
    feats = np.vstack([parts["allx"].toarray(), parts["tx"].toarray()]).astype(np.float32)
    labels = np.vstack([parts["ally"], parts["ty"]]).argmax(1)
    order = np.sort(test_idx)
    feats[test_idx] = feats[order]; labels[test_idx] = labels[order]   # undo test shuffling
    edges = {(i, j) for i, nbrs in parts["graph"].items() for j in nbrs if i != j}
    edges |= {(j, i) for i, j in edges}
    edge_index = np.array(sorted(edges)).T
    n_train = parts["y"].shape[0]
    return {"x": feats, "y": labels, "edge_index": edge_index,
            "train": np.arange(n_train), "val": np.arange(n_train, n_train + 500),
            "test": order}


def load_banking77():
    """Banking77: 13,083 real customer-support messages in 77 intents (Casanueva et al., 2020).
    Returns (train_texts, train_labels), (test_texts, test_labels), intent_names."""
    import csv
    out = []
    for split in ["train", "test"]:
        p = _fetch(SOURCES["banking77"] + f"{split}.csv", DATA / "banking77" / f"{split}.csv")
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        out.append(([r["text"] for r in rows], [r["category"] for r in rows]))
    names = sorted(set(out[0][1]))
    return ((out[0][0], np.array([names.index(c) for c in out[0][1]])),
            (out[1][0], np.array([names.index(c) for c in out[1][1]])), names)


def load_hdfs():
    """HDFS logs (Xu et al., 2009; Loghub): 100,000 parsed log lines and per-block anomaly labels.
    Returns the parsed lines as a list of dicts and a {block_id: is_anomaly} dict."""
    import csv
    p = _fetch(SOURCES["hdfs"] + "HDFS_100k.log_structured.csv", DATA / "hdfs" / "HDFS_100k.log_structured.csv")
    q = _fetch(SOURCES["hdfs"] + "anomaly_label.csv", DATA / "hdfs" / "anomaly_label.csv")
    with open(p, newline="") as f:
        lines = list(csv.DictReader(f))
    with open(q, newline="") as f:
        labels = {r["BlockId"]: r["Label"] == "Anomaly" for r in csv.DictReader(f)}
    return lines, labels


def load_movielens():
    """MovieLens 100K (Harper and Konstan, 2015): 100,000 ratings of 1,682 movies by 943 users.
    Downloaded from GroupLens on first use. License: GroupLens research license,
    non-commercial use only; cite Harper and Konstan (2015).
    Returns ratings as an (N, 4) int array [user, item, rating, timestamp] (0-based ids),
    and a list of (title, genres) per item."""
    p, q = _unzip(SOURCES["movielens"], DATA / "movielens" / "ml-100k.zip",
                  ["ml-100k/u.data", "ml-100k/u.item"])
    r = np.loadtxt(p, dtype=np.int64)
    r[:, :2] -= 1
    genres = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary",
              "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller",
              "War", "Western"]
    items = []
    for line in q.read_text(encoding="latin-1").strip().splitlines():
        f = line.split("|")
        items.append((f[1], [g for g, flag in zip(genres, f[5:]) if flag == "1"]))
    return r, items


def load_bike():
    """Capital Bikeshare hourly rentals, Washington D.C., 2011-2012 (Fanaee-T and Gama, 2013).
    Downloaded from the UCI Machine Learning Repository on first use. License: CC BY 4.0.
    Returns a pandas DataFrame with one row per hour."""
    import pandas as pd
    p, = _unzip(SOURCES["bike"], DATA / "bike" / "bike-sharing-dataset.zip", ["hour.csv"])
    return pd.read_csv(p, parse_dates=["dteday"])
