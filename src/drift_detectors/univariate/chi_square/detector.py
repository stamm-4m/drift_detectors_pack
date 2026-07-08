import numpy as np

from drift_detectors.utility._distributional import HistDistanceDetector


class ChiSquare(HistDistanceDetector):
    """Chi-square (triangular) divergence drift detector (univariate, distributional, batch)."""

    name = "chi_square"
    default_threshold = 0.1

    def _score(self, ref, test):
        p, q = self._hist(ref, test)
        return float(0.5 * np.sum((p - q) ** 2 / (p + q)))
