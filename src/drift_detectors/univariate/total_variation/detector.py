import numpy as np

from drift_detectors.utility._distributional import HistDistanceDetector


class TotalVariation(HistDistanceDetector):
    """Total variation distance drift detector (univariate, distributional, batch)."""

    name = "total_variation"
    default_threshold = 0.1

    def _score(self, ref, test):
        p, q = self._hist(ref, test)
        return float(0.5 * np.sum(np.abs(p - q)))
