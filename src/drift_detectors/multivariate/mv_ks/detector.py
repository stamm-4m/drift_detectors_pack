"""Multivariate KS via random projections."""
from __future__ import annotations

import numpy as np

from drift_detectors.drift_detector import DriftDetector
from drift_detectors.utility._stats import ks_2samp
from drift_detectors.utility.drift_detection_output import ScoreDriftResult


class MultivariateKS(DriftDetector):
    """Project the reference and test samples onto ``n_projections`` random unit
    directions and run a univariate two-sample KS test on each; the score is the
    fraction of projections that show significant drift. Multivariate, batch."""

    def __init__(self, reference_data: np.ndarray = None, n_projections: int = 50,
                 alpha: float = 0.01, score_threshold: float = 0.1,
                 seed: int = 0, online: bool = False):
        super().__init__(reference_data=reference_data, online=online)
        self.n_projections = int(n_projections)
        self.alpha = float(alpha)
        self.score_threshold = float(score_threshold)
        self.seed = int(seed)
        self._test_buffer: list = []

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
        d = ref.shape[1]
        dirs = rng.normal(size=(self.n_projections, d))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        sig = 0
        for w in dirs:
            _, p = ks_2samp(ref @ w, test @ w)
            sig += p < alpha
        score = float(sig / self.n_projections)
        return ScoreDriftResult(score=score, drift=bool(score >= self.score_threshold),
            details={"alpha": alpha, "score_threshold": self.score_threshold,
                     "n_projections": self.n_projections,
                     "mode": "online" if self.is_online() else "offline",
                     "reference_size": len(ref), "test_size": len(test)})
