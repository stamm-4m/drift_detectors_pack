"""Pair-wise (1 - Pearson correlation) / 2 disagreement."""
from __future__ import annotations

import numpy as np

from drift_detectors.model_based.disagreement_metrics.base import DisagreementMetric


def _pearson(x, y):
    """Pearson correlation, or NaN when it is not defined (fewer than 2
    points, or a constant series). Until 0.5.1 this returned 0.0, which the
    metric reported as a disagreement of 0.5 (looked like a real value)."""
    if x.size < 2:
        return float("nan")
    xc = x - x.mean()
    yc = y - y.mean()
    den = np.sqrt((xc * xc).sum() * (yc * yc).sum())
    if den <= 0 or not np.isfinite(den):
        return float("nan")
    return float(np.clip((xc * yc).sum() / den, -1.0, 1.0))


class PearsonDisagreement(DisagreementMetric):
    name = "pearson"
    kind = "correlation"

    def pair(self, y_i, y_j, *, scale=1.0):
        """(1 - r) / 2 in [0, 1]; NaN when r is not defined (constant series)."""
        r = _pearson(y_i, y_j)
        if not np.isfinite(r):
            return float("nan")
        return float(np.clip(1.0 - r, 0.0, 2.0)) / 2.0
