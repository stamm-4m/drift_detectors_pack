"""PCA reconstruction-error drift detector."""
from __future__ import annotations

import numpy as np

from drift_detectors.drift_detector import DriftDetector
from drift_detectors.utility._stats import ks_2samp
from drift_detectors.utility.drift_detection_output import ScoreDriftResult


class PCAReconstruction(DriftDetector):
    """Fit PCA on one half of the reference and compare the reconstruction-error
    distribution of the *other* (held-out) reference half against the test set
    with a KS test. Using a held-out reference half removes the in-sample bias
    that would otherwise inflate false positives. Sensitive to drift outside the
    reference's principal subspace. Multivariate, batch."""

    def __init__(self, reference_data: np.ndarray = None, n_components: int = 2,
                 alpha: float = 0.05, seed: int = 0, online: bool = False):
        super().__init__(reference_data=reference_data, online=online)
        self.n_components = int(n_components)
        self.alpha = float(alpha)
        self.seed = int(seed)
        self._test_buffer = [] if online else None

    def calculate(self, test_data, reference_data=None, alpha=None):
        ref = reference_data if reference_data is not None else self._reference_data
        if ref is None:
            raise ValueError("Reference data must be provided or set before calculation.")
        ref = np.atleast_2d(np.asarray(ref, dtype=float))
        test = np.atleast_2d(np.asarray(test_data, dtype=float))
        if self.is_online():
            self._test_buffer.extend(test)
            test = np.asarray(self._test_buffer, dtype=float)
        half = len(ref) // 2
        if half <= self.n_components or len(test) < 2:
            return ScoreDriftResult(score=None, drift=False, details={"status": "not_ready"})
        alpha = self.alpha if alpha is None else float(alpha)

        perm = np.random.default_rng(self.seed).permutation(len(ref))
        fit, cal = ref[perm[:half]], ref[perm[half:]]
        mean = fit.mean(axis=0)
        _, _, vt = np.linalg.svd(fit - mean, full_matrices=False)
        comps = vt[: self.n_components]

        def err(x):
            xc = x - mean
            return np.sqrt(np.sum((xc - (xc @ comps.T) @ comps) ** 2, axis=1))

        e_cal, e_test = err(cal), err(test)
        d, p = ks_2samp(e_cal, e_test)
        return ScoreDriftResult(score=float(d), drift=bool(p < alpha),
            details={"p_value": float(p), "alpha": alpha, "n_components": self.n_components,
                     "mean_ref_error": float(e_cal.mean()), "mean_test_error": float(e_test.mean()),
                     "mode": "online" if self.is_online() else "offline",
                     "reference_size": len(ref), "test_size": len(test)})
