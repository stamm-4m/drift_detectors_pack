import numpy as np
from scipy.stats import ks_2samp

from drift_detectors.drift_detector import DriftDetector
from drift_detectors.utility.drift_detection_output import ScoreDriftResult


class KSDetector(DriftDetector):
    """
    Kolmogorov–Smirnov Drift Detector (univariate, batch or online).

    Uses the two-sample KS test to determine whether two univariate distributions
    differ significantly. Outputs the KS statistic and p-value, and flags drift
    based on a significance threshold (alpha).
    """

    def __init__(self,
                 reference_data: np.ndarray = None,
                 alpha: float = 0.05,
                 online: bool = False):
        """
        Initialize the KS detector.

        Parameters:
            reference_data (np.ndarray, optional): Optional reference distribution.
            alpha (float): Significance threshold for p-value.
            online (bool): Whether to accumulate test data internally.
        """
        super().__init__(reference_data=reference_data, online=online)
        self._alpha = alpha
        if self.is_online():
            self._test_buffer = []

    def calculate(self,
                  test_data: np.ndarray,
                  reference_data: np.ndarray = None,
                  alpha: float = None) -> ScoreDriftResult:
        """
        Compare reference and test data using the two-sample KS test.

        Parameters:
            test_data (np.ndarray): Incoming test data (batch or update).
            reference_data (np.ndarray, optional): Override for stored reference data.
            alpha (float, optional): Significance level.

        Returns:
            ScoreDriftResult: KS statistic, p-value, and drift decision.
        """
        ref = reference_data if reference_data is not None else self._reference_data
        if ref is None:
            raise ValueError("Reference data must be provided or initialized.")

        ref = np.asarray(ref).ravel()
        new = np.asarray(test_data).ravel()
        alpha = alpha if alpha is not None else self._alpha

        if self.is_online():
            self._test_buffer.extend(new.tolist())
            test = np.array(self._test_buffer)
        else:
            test = new

        stat, p_value = ks_2samp(ref, test)
        drift_detected = p_value < alpha

        return ScoreDriftResult(
            score=stat,
            drift=drift_detected,
            details={
                "alpha": alpha,
                "p_value": p_value,
                "mode": "online" if self.is_online() else "offline",
                "reference_size": ref.size,
                "test_size": test.size,
            }
        )
