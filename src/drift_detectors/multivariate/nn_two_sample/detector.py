"""Nearest-neighbour two-sample test (Schilling, 1986; Henze, 1988)."""
from __future__ import annotations

import math

import numpy as np

from drift_detectors.drift_detector import DriftDetector
from drift_detectors.utility._stats import knn_indices
from drift_detectors.utility.drift_detection_output import ScoreDriftResult


class NNTwoSample(DriftDetector):
    """Pool the reference and test samples (balanced), and for each point count
    how many of its k nearest neighbours share its label. Under the null the
    same-label fraction is at chance; a significantly higher fraction indicates
    the two samples separate. One-sided normal approximation. Multivariate."""

    def __init__(self, reference_data: np.ndarray = None, k_neighbors: int = 5,
                 alpha: float = 0.05, max_samples: int = 1000, online: bool = False,
                 seed: int = 0):
        super().__init__(reference_data=reference_data, online=online)
        self.k_neighbors = int(k_neighbors)
        self.alpha = alpha
        self.max_samples = int(max_samples)
        self.seed = int(seed)
        self._test_buffer = [] if online else None

    def calculate(self, test_data, reference_data=None, alpha=None):
        ref = reference_data if reference_data is not None else self._reference_data
        if ref is None:
            raise ValueError("Reference data must be provided or set before calculation.")
        ref = np.atleast_2d(np.asarray(ref, dtype=float))
        test = np.atleast_2d(np.asarray(test_data, dtype=float))
        if self.is_online():
            self._test_buffer.extend(test)
            test = np.asarray(self._test_buffer, dtype=float)
        alpha = self.alpha if alpha is None else float(alpha)
        rng = np.random.default_rng(self.seed)
        n = min(len(ref), len(test), self.max_samples)
        if n < max(2 * self.k_neighbors, 10):
            return ScoreDriftResult(score=None, drift=False, details={"status": "not_ready"})
        rs = ref[rng.choice(len(ref), n, replace=False)]
        ts = test[rng.choice(len(test), n, replace=False)]
        X = np.vstack([rs, ts])
        y = np.concatenate([np.zeros(n), np.ones(n)])
        # k+1 neighbours, drop the first (the point itself)
        idx = knn_indices(X, X, self.k_neighbors + 1)[:, 1:]
        same = float(np.mean(y[idx] == y[:, None]))
        # under H0 (balanced) same-label fraction ~ (n-1)/(2n-1); normal approx
        p0 = (n - 1) / (2 * n - 1)
        se = math.sqrt(p0 * (1 - p0) / (2 * n * self.k_neighbors))
        z = (same - p0) / se if se > 0 else 0.0
        p_value = 0.5 * math.erfc(z / math.sqrt(2.0))
        return ScoreDriftResult(score=float(same), drift=bool(p_value < alpha),
            details={"p_value": float(p_value), "same_label_fraction": float(same),
                     "chance_level": float(p0), "k_neighbors": self.k_neighbors,
                     "mode": "online" if self.is_online() else "offline",
                     "reference_size": len(ref), "test_size": len(test)})
