"""Pair-wise (1 - Pearson correlation) / 2 disagreement."""
from __future__ import annotations
import numpy as np
from drift_detectors.model_based.disagreement_metrics.base import DisagreementMetric


def _pearson(x, y):
    if x.size < 2:
        return 0.0
    xc = x - x.mean()
    yc = y - y.mean()
    den = np.sqrt((xc * xc).sum() * (yc * yc).sum())
    if den <= 0 or not np.isfinite(den):
        return 0.0
    return float(np.clip((xc * yc).sum() / den, -1.0, 1.0))


class PearsonDisagreement(DisagreementMetric):
    name = "pearson"
    kind = "correlation"

    def pair(self, y_i, y_j, *, scale=1.0):
        return float(np.clip(1.0 - _pearson(y_i, y_j), 0.0, 2.0)) / 2.0
