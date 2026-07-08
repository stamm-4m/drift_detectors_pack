"""Autocorrelation-structure divergence (time-series-aware)."""
from __future__ import annotations

import numpy as np

from drift_detectors.model_based.disagreement_metrics.base import DisagreementMetric


def _acf(x: np.ndarray, n_lags: int) -> np.ndarray:
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return np.zeros(n_lags)
    n = len(x)
    return np.array([np.dot(x[: n - k], x[k:]) / denom for k in range(1, n_lags + 1)])


class ACFDivergence(DisagreementMetric):
    """Distance between the autocorrelation functions of two prediction series.

    The default MSE/Pearson/Spearman metrics assume independent samples and so
    ignore the strong autocorrelation of bioprocess time series. This metric
    compares the two series' autocorrelation functions directly: two sensors
    with different temporal dependence structure score high, regardless of
    magnitude (the ACF is scale-free). Family: ``temporal``.
    """

    name = "acf"
    kind = "temporal"

    def __init__(self, max_lag: int = 10):
        self.max_lag = int(max_lag)

    def pair(self, y_i, y_j, *, scale=1.0):
        a = np.asarray(y_i, dtype=float).ravel()
        b = np.asarray(y_j, dtype=float).ravel()
        n_lags = min(self.max_lag, len(a) // 2, len(b) // 2)
        if n_lags < 1:
            return 0.0
        d = np.sqrt(np.mean((_acf(a, n_lags) - _acf(b, n_lags)) ** 2))
        return float(np.clip(d, 0.0, 1.0))
