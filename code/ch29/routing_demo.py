# How much can the clothing classifier handle on its own, at 99% accuracy?
import sys, json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import numpy as np
from wtlkit import mnist_tensors
from wtlapps import OnnxModel, ConfidenceRouter

model = OnnxModel(root / "ch28" / "resnet20_int8.onnx")   # the component built in Chapter 28
_, (X_va, y_va), (X_te, y_te) = mnist_tensors(dataset="fashion", image=True)


def confidence_and_correct(X, y):
    p = model.predict_proba(X.numpy())
    return p.max(1), p.argmax(1) == y.numpy()


conf_va, ok_va = confidence_and_correct(X_va, y_va)
conf_te, ok_te = confidence_and_correct(X_te, y_te)
print(f"model alone: test accuracy {ok_te.mean():.2%} on 100% of items")
curves = {}
for target in [0.98, 0.99, 0.995]:
    router = ConfidenceRouter().fit(conf_va, ok_va, target)    # threshold chosen on validation data
    auto = router.route(conf_te) == "auto"                     # ... and checked on unseen test data
    print(f"target {target:.1%}: threshold {router.threshold:.3f} -> automated {auto.mean():.1%} of test items "
          f"at {ok_te[auto].mean():.2%} accuracy; {(~auto).sum():,} sent to a person")
    if target == 0.99:                                  # Chapter 30 reads this threshold
        with open("router_99.json", "w") as f:
            json.dump({"threshold": router.threshold}, f)
conf, acc, cov = ConfidenceRouter.curve(conf_te, ok_te)
step = len(cov) // 200
json.dump({"coverage": cov[::step].tolist(), "accuracy": acc[::step].tolist()}, open("coverage_curve.json", "w"))
