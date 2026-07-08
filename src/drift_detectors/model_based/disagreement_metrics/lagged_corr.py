"""Lagged cross-correlation disagreement (time-series-aware)."""
from __future__ import annotations

import numpy as np

from drift_detectors.model_based.disagreement_metrics.base import DisagreementMetric
from drift_detectors.model_based.disagreement_metrics.pearson import _pearson


class LaggedCorrelationDisagreement(DisagreementMetric):
    """Best absolute cross-correlation over a range of time lags.

    Two soft-sensor series that track the same signal with a small delay are
    strongly correlated at some lag; ordinary Pearson (lag 0) would miss it.
    Disagreement is ``1 - max_lag |corr|``, so a shifted-but-matching pair
    scores low and an unrelated pair scores high. Family: ``temporal``.
    """

    name = "lagged_corr"
    kind = "temporal"

    def __init__(self, max_lag: int = 10):
        self.max_lag = int(max_lag)

    def pair(self, y_i, y_j, *, scale=1.0):
        a = np.asarray(y_i, dtype=float).ravel()
        b = np.asarray(y_j, dtype=float).ravel()
        n = min(len(a), len(b))
        if n < 3:
            return 0.0
        a, b = a[:n], b[:n]
        lim = min(self.max_lag, n - 2)
        best = 0.0
        for lag in range(-lim, lim + 1):
            if lag >= 0:
                x, y = a[lag:], b[:n - lag]
            else:
                x, y = a[:n + lag], b[-lag:]
            if len(x) >= 2:
                best = max(best, abs(_pearson(x, y)))
        return float(np.clip(1.0 - best, 0.0, 1.0))
