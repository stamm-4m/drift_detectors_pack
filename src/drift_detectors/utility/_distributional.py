"""Shared base for histogram/CDF-based univariate distributional detectors."""
from __future__ import annotations

import numpy as np

from drift_detectors.drift_detector import DriftDetector
from drift_detectors.utility._stats import hist_pq
from drift_detectors.utility.drift_detection_output import ScoreDriftResult


class HistDistanceDetector(DriftDetector):
    """Base class: subclasses implement ``_score(ref, test) -> float`` and set
    ``name`` and ``default_threshold``. Handles ref/test resolution, the online
    buffer, thresholding and the result object uniformly."""

    name: str = "abstract"
    default_threshold: float = 0.1

    def __init__(self, reference_data: np.ndarray = None, threshold: float = None,
                 bins: int = 20, epsilon: float = 1e-8, online: bool = False):
        super().__init__(reference_data=reference_data, online=online)
        self._threshold = self.default_threshold if threshold is None else float(threshold)
        self._bins = int(bins)
        self._epsilon = float(epsilon)
        self._test_buffer = [] if online else None

    def _hist(self, ref, test):
        return hist_pq(ref, test, self._bins, self._epsilon)

    def _score(self, ref: np.ndarray, test: np.ndarray) -> float:
        raise NotImplementedError

    def calculate(self, test_data: np.ndarray, reference_data: np.ndarray = None,
                  threshold: float = None) -> ScoreDriftResult:
        ref = reference_data if reference_data is not None else self._reference_data
        if ref is None:
            raise ValueError("Reference data must be provided or initialized.")
        ref = np.asarray(ref, dtype=float).ravel()
        new = np.asarray(test_data, dtype=float).ravel()
        if self.is_online():
            self._test_buffer.extend(new.tolist())
            test = np.asarray(self._test_buffer, dtype=float)
        else:
            test = new
        if ref.size == 0 or test.size == 0:
            raise ValueError("Reference and test data must be non-empty.")
        thr = self._threshold if threshold is None else float(threshold)
        score = float(self._score(ref, test))
        return ScoreDriftResult(
            score=score, drift=bool(score >= thr),
            details={"threshold": thr, "bins": self._bins, "metric": self.name,
                     "mode": "online" if self.is_online() else "offline",
                     "reference_size": ref.size, "test_size": test.size})
