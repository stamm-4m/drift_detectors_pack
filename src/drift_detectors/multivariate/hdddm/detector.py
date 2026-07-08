"""HDDDM -- Hellinger Distance Drift Detection Method (Ditzler & Polikar, 2011)."""
from __future__ import annotations

import numpy as np

from drift_detectors.drift_detector import DriftDetector
from drift_detectors.utility._stats import hist_pq
from drift_detectors.utility.drift_detection_output import ScoreDriftResult


class HDDDM(DriftDetector):
    """Per-feature histogram Hellinger distance, averaged over features. Bounded
    [0, 1]; 0 = identical marginals, 1 = disjoint. Multivariate, batch."""

    def __init__(self, reference_data: np.ndarray = None, threshold: float = 0.2,
                 bins: int = 20, online: bool = False):
        super().__init__(reference_data=reference_data, online=online)
        self.threshold = float(threshold)
        self.bins = int(bins)
        self._test_buffer = [] if online else None

    def calculate(self, test_data, reference_data=None, threshold=None):
        ref = reference_data if reference_data is not None else self._reference_data
        if ref is None:
            raise ValueError("Reference data must be provided or set before calculation.")
        ref = np.atleast_2d(np.asarray(ref, dtype=float))
        test = np.atleast_2d(np.asarray(test_data, dtype=float))
        if self.is_online():
            self._test_buffer.extend(test)
            test = np.asarray(self._test_buffer, dtype=float)
        thr = self.threshold if threshold is None else float(threshold)
        dists = []
        for d in range(ref.shape[1]):
            p, q = hist_pq(ref[:, d], test[:, d], self.bins)
            dists.append(np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2.0))
        score = float(np.mean(dists))
        return ScoreDriftResult(score=score, drift=bool(score >= thr),
            details={"threshold": thr, "bins": self.bins, "n_features": ref.shape[1],
                     "mode": "online" if self.is_online() else "offline",
                     "reference_size": len(ref), "test_size": len(test)})
