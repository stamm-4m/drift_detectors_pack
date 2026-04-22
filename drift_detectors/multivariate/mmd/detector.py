import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from drift_detectors.drift_detector import DriftDetector
from drift_detectors.utility.drift_detection_output import ScoreDriftResult


class MMDDetector(DriftDetector):
    """
    Maximum Mean Discrepancy (MMD) Drift Detector.

    Compares distributions using kernel embedding of samples.
    """

    def __init__(self,
                 reference_data: np.ndarray = None,
                 gamma: float = 1.0,
                 threshold: float = 0.05,
                 online: bool = False):
        """
        Parameters:
            reference_data (np.ndarray, optional): Reference dataset.
            gamma (float): Kernel width parameter for RBF kernel.
            threshold (float): MMD score threshold to flag drift.
            online (bool): Whether to accumulate test data incrementally.
        """
        super().__init__(reference_data=reference_data, online=online)
        self.gamma = gamma
        self.threshold = threshold
        self._test_buffer = [] if online else None

    def calculate(self,
                  test_data: np.ndarray,
                  reference_data: np.ndarray = None,
                  gamma: float = None,
                  threshold: float = None) -> ScoreDriftResult:
        """
        Compute MMD score between reference and test data.

        Parameters:
            test_data (np.ndarray): New test data.
            reference_data (np.ndarray, optional): Optional override for reference.
            gamma (float, optional): Kernel width (overrides default).
            threshold (float, optional): Drift threshold (overrides default).

        Returns:
            ScoreDriftResult
        """
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

        XX = rbf_kernel(ref, ref, gamma=gamma)
        YY = rbf_kernel(test, test, gamma=gamma)
        XY = rbf_kernel(ref, test, gamma=gamma)

        score = XX.mean() + YY.mean() - 2 * XY.mean()
        drift = score >= threshold

        return ScoreDriftResult(
            score=score,
            drift=drift,
            details={
                "gamma": gamma,
                "threshold": threshold,
                "reference_size": len(ref),
                "test_size": len(test),
                "mode": "online" if self.is_online() else "offline"
            }
        )
