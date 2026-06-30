import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from drift_detectors.drift_detector import DriftDetector
from drift_detectors.utility.drift_detection_output import ScoreDriftResult


class MMDDetector(DriftDetector):
    """Maximum Mean Discrepancy (MMD) Drift Detector.

    Compares distributions using kernel embedding of samples.
    """

    def __init__(self,
                 reference_data: np.ndarray = None,
                 gamma=1.0,
                 threshold: float = 0.05,
                 online: bool = False):
        """gamma may be a float or the string ``"median"`` for the
        median-distance heuristic (Gretton et al., 2012). The latter is
        strongly recommended for real-world data with mixed-scale features.
        """
        super().__init__(reference_data=reference_data, online=online)
        self.gamma = gamma
        self.threshold = threshold
        self._test_buffer = [] if online else None

    @staticmethod
    def _median_heuristic_gamma(X: np.ndarray, Y: np.ndarray, max_samples: int = 500) -> float:
        rng = np.random.default_rng(0)
        XY = np.vstack([np.atleast_2d(X), np.atleast_2d(Y)])
        if len(XY) > max_samples:
            idx = rng.choice(len(XY), size=max_samples, replace=False)
            XY = XY[idx]
        diff = XY[:, None, :] - XY[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)
        iu = np.triu_indices_from(d2, k=1)
        med = float(np.median(d2[iu])) if iu[0].size else 1.0
        return 1.0 / (2.0 * max(med, 1e-12))

    def calculate(self,
                  test_data: np.ndarray,
                  reference_data: np.ndarray = None,
                  gamma=None,
                  threshold: float = None) -> ScoreDriftResult:
        ref = reference_data if reference_data is not None else self._reference_data
        if ref is None:
            raise ValueError("Reference data must be provided or set before calculation.")

        test_data = np.atleast_2d(test_data)
        if self.is_online():
            self._test_buffer.extend(test_data)
            test = np.array(self._test_buffer)
        else:
            test = test_data

        if len(ref) == 0 or len(test) == 0:
            return ScoreDriftResult(score=None, drift=False, details={"status": "not_ready"})

        gamma = gamma if gamma is not None else self.gamma
        threshold = threshold if threshold is not None else self.threshold

        if isinstance(gamma, str) and gamma == "median":
            gamma = self._median_heuristic_gamma(np.atleast_2d(ref), np.atleast_2d(test))

        XX = rbf_kernel(ref, ref, gamma=gamma)
        YY = rbf_kernel(test, test, gamma=gamma)
        XY = rbf_kernel(ref, test, gamma=gamma)

        score = float(XX.mean() + YY.mean() - 2 * XY.mean())
        drift = score >= threshold

        return ScoreDriftResult(
            score=score,
            drift=drift,
            details={
                "gamma": float(gamma),
                "threshold": threshold,
                "reference_size": len(ref),
                "test_size": len(test),
                "mode": "online" if self.is_online() else "offline",
            },
        )
