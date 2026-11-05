"""Decide when to trust the model and when to ask a person."""
import numpy as np


class ConfidenceRouter:
    """Automates predictions whose confidence is at least a threshold and sends the rest
    to human review. The threshold is chosen on validation data to reach a target accuracy
    on the automated part."""

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    @staticmethod
    def curve(confidence, correct):
        """Accuracy of the automated part and coverage, for every possible threshold."""
        order = np.argsort(-confidence)
        conf, ok = confidence[order], correct[order].astype(float)
        n = np.arange(1, len(ok) + 1)
        return conf, np.cumsum(ok) / n, n / len(ok)          # threshold, accuracy, coverage

    def fit(self, confidence, correct, target_accuracy):
        conf, acc, cov = self.curve(confidence, correct)
        ok = np.nonzero(acc >= target_accuracy)[0]
        self.threshold = float(conf[ok.max()]) if len(ok) else 1.01   # the largest coverage that meets the target
        return self

    def route(self, confidence):
        return np.where(confidence >= self.threshold, "auto", "human")
