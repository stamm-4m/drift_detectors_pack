import numpy as np

from drift_detectors.utility._distributional import HistDistanceDetector


class HellingerDistance(HistDistanceDetector):
    """Hellinger distance drift detector (univariate, distributional, batch)."""

    name = "hellinger"
    default_threshold = 0.1

    def _score(self, ref, test):
        p, q = self._hist(ref, test)
        return float(np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2.0))
