"""Dynamic Time Warping disagreement (time-series-aware)."""
from __future__ import annotations

import numpy as np

from drift_detectors.model_based.disagreement_metrics.base import DisagreementMetric


def _downsample(x: np.ndarray, max_len: int) -> np.ndarray:
    if len(x) <= max_len:
        return x
    idx = np.linspace(0, len(x) - 1, max_len).round().astype(int)
    return x[idx]


def _dtw(a: np.ndarray, b: np.ndarray, window: int | None) -> float:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0
    w = None if window is None else max(int(window), abs(n - m))
    prev = np.full(m + 1, np.inf)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = np.full(m + 1, np.inf)
        jlo, jhi = (1, m) if w is None else (max(1, i - w), min(m, i + w))
        for j in range(jlo, jhi + 1):
            cost = abs(a[i - 1] - b[j - 1])
            cur[j] = cost + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return float(prev[m])


class DTWDisagreement(DisagreementMetric):
    """Dynamic Time Warping distance between two prediction series.

    Unlike MSE/Pearson, DTW aligns the series in time, so it measures how far
    apart two co-deployed soft sensors are *as trajectories*, tolerating small
    time shifts. Scale-normalised to [0, 1]. Family: ``temporal``.
    """

    name = "dtw"
    kind = "temporal"

    def __init__(self, window: int | None = None, max_len: int = 1000):
        self.window = window
        self.max_len = int(max_len)

    def pair(self, y_i, y_j, *, scale=1.0):
        a = _downsample(np.asarray(y_i, dtype=float).ravel(), self.max_len)
        b = _downsample(np.asarray(y_j, dtype=float).ravel(), self.max_len)
        if scale <= 0 or not np.isfinite(scale):
            scale = 1.0
        steps = max(len(a), len(b), 1)
        return float(min((_dtw(a, b, self.window) / steps) / scale, 1.0))
