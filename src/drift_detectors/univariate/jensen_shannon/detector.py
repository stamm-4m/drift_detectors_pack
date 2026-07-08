import numpy as np

from drift_detectors.utility._distributional import HistDistanceDetector


class JensenShannon(HistDistanceDetector):
    """Jensen-Shannon divergence drift detector (univariate, distributional, batch)."""

    name = "jensen_shannon"
    default_threshold = 0.1

    def _score(self, ref, test):
        p, q = self._hist(ref, test)
        m = 0.5 * (p + q)
        return float(0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m)))
