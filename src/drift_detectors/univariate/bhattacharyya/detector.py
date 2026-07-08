import numpy as np

from drift_detectors.utility._distributional import HistDistanceDetector


class Bhattacharyya(HistDistanceDetector):
    """Bhattacharyya distance drift detector (univariate, distributional, batch)."""

    name = "bhattacharyya"
    default_threshold = 0.05

    def _score(self, ref, test):
        p, q = self._hist(ref, test)
        bc = float(np.sum(np.sqrt(p * q)))
        return float(-np.log(max(bc, 1e-12)))
