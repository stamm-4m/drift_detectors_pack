"""Linear-time MMD drift detector (Gretton et al., 2012, linear estimate)."""
from __future__ import annotations

import numpy as np

from drift_detectors.drift_detector import DriftDetector
from drift_detectors.utility.drift_detection_output import ScoreDriftResult


class MMDLinear(DriftDetector):
    """Unbiased linear-time estimate of MMD^2 with an RBF kernel -- an O(n)
    alternative to the quadratic MMD for large batches. Multivariate, batch.
    ``gamma`` may be a float or ``'median'`` (median-distance heuristic)."""

    def __init__(self, reference_data: np.ndarray = None, gamma="median",
                 threshold: float = 0.03, online: bool = False):
        super().__init__(reference_data=reference_data, online=online)
        self.gamma = gamma
        self.threshold = float(threshold)
        self._test_buffer = [] if online else None

    @staticmethod
    def _k(a, b, g):
        return np.exp(-g * np.sum((a - b) ** 2, axis=-1))

    def calculate(self, test_data, reference_data=None, gamma=None, threshold=None):
        ref = reference_data if reference_data is not None else self._reference_data
        if ref is None:
            raise ValueError("Reference data must be provided or set before calculation.")
        ref = np.atleast_2d(np.asarray(ref, dtype=float))
        test = np.atleast_2d(np.asarray(test_data, dtype=float))
        if self.is_online():
            self._test_buffer.extend(test)
            test = np.asarray(self._test_buffer, dtype=float)
        g = self.gamma if gamma is None else gamma
        thr = self.threshold if threshold is None else float(threshold)
        m = min(len(ref), len(test))
        m -= m % 2
        if m < 2:
            return ScoreDriftResult(score=None, drift=False, details={"status": "not_ready"})
        x, y = ref[:m], test[:m]
        if isinstance(g, str) and g == "median":
            xy = np.vstack([x, y])
            diff = xy[:, None, :] - xy[None, :, :]
            d2 = np.sum(diff * diff, axis=-1)
            med = float(np.median(d2[np.triu_indices_from(d2, 1)])) or 1.0
            g = 1.0 / (2.0 * max(med, 1e-12))
        x1, x2 = x[0::2], x[1::2]
        y1, y2 = y[0::2], y[1::2]
        h = (self._k(x1, x2, g) + self._k(y1, y2, g)
             - self._k(x1, y2, g) - self._k(x2, y1, g))
        score = float(np.mean(h))
        return ScoreDriftResult(score=score, drift=bool(score >= thr),
            details={"gamma": float(g), "threshold": thr,
                     "mode": "online" if self.is_online() else "offline",
                     "reference_size": len(ref), "test_size": len(test)})
