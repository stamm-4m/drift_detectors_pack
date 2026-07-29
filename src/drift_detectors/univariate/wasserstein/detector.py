import numpy as np

from drift_detectors.utility._distributional import HistDistanceDetector
from drift_detectors.utility._stats import wasserstein_1d


class Wasserstein(HistDistanceDetector):
    """Wasserstein-1 (earth-mover) distance drift detector (univariate, distributional, batch)."""

    name = "wasserstein"
    default_threshold = 0.2

    def _score(self, ref, test):
        scale = float(np.std(np.concatenate([ref, test]))) or 1.0
        return float(wasserstein_1d(ref, test) / max(scale, 1e-12))
