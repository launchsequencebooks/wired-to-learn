# Test the organizer on a realistic mixed collection: 2,000 clothes and 100 stray digits.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from wtlkit import mnist_tensors, FASHION_CLASSES
from organizer import Organizer

_, _, (X, y) = mnist_tensors(dataset="fashion", image=True)
_, _, (D, _) = mnist_tensors(dataset="mnist", image=True)
images = np.concatenate([X[:2000].numpy(), D[:100].numpy()])       # photos that do not belong, mixed in
names = [f"img{i:04d}" for i in range(2000)] + [f"digit{i:02d}" for i in range(100)]
truth = y[:2000].numpy()

org = Organizer().ingest(images, names)
albums = org.albums()
print("albums:", ", ".join(f"{k} {len(v)}" for k, v in sorted(albums.items(), key=lambda kv: -len(kv[1]))))
clothes_auto = ~org.needs_review[:2000] & ~org.is_outlier[:2000]
correct = np.array([org.tags[i] == FASHION_CLASSES[truth[i]] for i in range(2000)])
print(f"clothes filed automatically {clothes_auto.mean():.1%}, of which correctly tagged {correct[clothes_auto].mean():.2%}")
print(f"digits caught as 'Not clothing?' {org.is_outlier[2000:].sum()}/100, "
      f"clothes wrongly flagged {org.is_outlier[:2000].sum()}/2000")
thr = org.router.threshold                        # the 99% threshold from Chapter 29
print(f"what the classifier alone says about the digits: {np.mean(org.confidence[2000:] > thr):.0%} "
      f"tagged with over {thr:.0%} confidence, e.g. as {org.tags[2000]!r}, {org.tags[2001]!r}, {org.tags[2002]!r}")

for embedder in ["embed_resnet.onnx", "embed_simclr.onnx"]:          # which embedding finds similar items?
    o = Organizer(embedder).ingest(images[:2000], names[:2000])
    # a photo is not its own neighbour: exclude each query by its id
    ids, _ = o.index.search(o.index.vectors, k=5, exclude_ids=names[:2000])
    hits = [np.mean([truth[int(j[3:])] == truth[i] for j in row]) for i, row in enumerate(ids)]
    print(f"{embedder:18s} 'find similar': {np.mean(hits):.1%} of the top 5 are the same kind of garment")

near = np.roll(images[:2000], 1, axis=3) * 0.9      # a copy, shifted one pixel and darkened
sim_copy = (o.index.vectors * o.index._normalize(o.embedder(near))).sum(1)
sims = o.index.vectors @ o.index.vectors.T; np.fill_diagonal(sims, -1)
print(f"duplicates: a shifted, darkened copy has cosine {np.median(sim_copy):.3f} "
      f"to its original; {np.mean(sims.max(1) > np.median(sim_copy)):.0%} of photos "
      f"have a *different* photo at least that close")
