# The baseline every text model must beat: word counts (TF-IDF) + logistic regression.
import json, time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from triage_data import split, INTENT_TO_TEAM

(X_tr, y_tr), (X_te, y_te) = split("train"), split("test")
vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
clf = LogisticRegression(C=20, max_iter=3000)
t0 = time.perf_counter()
clf.fit(vec.fit_transform(X_tr), y_tr.numpy())
pred = clf.predict(vec.transform(X_te))
y = y_te.numpy()
print(f"TF-IDF + logistic regression: intent accuracy {np.mean(pred == y):.2%}, macro-F1 {f1_score(y, pred, average='macro'):.3f}, "
      f"team accuracy {np.mean(INTENT_TO_TEAM[pred] == INTENT_TO_TEAM[y]):.2%}  ({time.perf_counter() - t0:.0f}s)")
json.dump({"acc": float(np.mean(pred == y)), "team": float(np.mean(INTENT_TO_TEAM[pred] == INTENT_TO_TEAM[y]))}, open("baseline_results.json", "w"))
