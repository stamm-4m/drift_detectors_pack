import numpy as np

from drift_detectors.utility._distributional import HistDistanceDetector


class CramerVonMises(HistDistanceDetector):
    """Cramer-von Mises (2-sample) drift detector (univariate, distributional, batch)."""

    name = "cramer_von_mises"
    default_threshold = 0.05

    def _score(self, ref, test):
        allv = np.sort(np.concatenate([ref, test]))
        fa = np.searchsorted(np.sort(ref), allv, side="right") / ref.size
        fb = np.searchsorted(np.sort(test), allv, side="right") / test.size
        return float(np.mean((fa - fb) ** 2))
