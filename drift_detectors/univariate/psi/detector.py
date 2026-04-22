import numpy as np
from drift_detectors.drift_detector import DriftDetector
from drift_detectors.utility.drift_detection_output import ScoreDriftResult


class PSI(DriftDetector):
    """
    Population Stability Index (PSI) Drift Detector.
    """

    def __init__(self,
                 reference_data: np.ndarray = None,
                 online: bool = False,
                 threshold: float = 0.1):
        """
        Initialize the PSI detector.

        Parameters:
            reference_data (np.ndarray, optional): Reference data distribution.
            online (bool): Enable stateful online mode.
            threshold (float): Threshold for detecting drift.
        """
        super().__init__(reference_data=reference_data, online=online)
        self._threshold = threshold
        if online:
            self._test_buffer = []

    def calculate(self,
                  test_data: np.ndarray,
                  reference_data: np.ndarray = None,
                  bins: int = 10,
                  epsilon: float = 1e-8) -> ScoreDriftResult:
        """
        Compute PSI between reference and test data.

        Parameters:
            test_data (np.ndarray): New test data or update to test buffer.
            reference_data (np.ndarray, optional): Override reference for this call. If not provided, internal reference is used.
            bins (int): Number of bins for histogram comparison.
            epsilon (float): Smoothing constant for stability.

        Returns:
            ScoreDriftResult: PSI score, drift flag, and details.
        """
        ref = reference_data if reference_data is not None else self._reference_data
        if ref is None:
            raise ValueError("Reference data must be provided or initialized.")

        ref = np.asarray(ref).ravel()
        new_data = np.asarray(test_data).ravel()

        if self.is_online():
            self._test_buffer.extend(new_data.tolist())
            test = np.array(self._test_buffer)
        else:
            test = new_data

        if ref.size == 0 or test.size == 0:
            raise ValueError("Reference and test data must be non-empty.")

        bin_edges = np.linspace(
            min(ref.min(), test.min()),
            max(ref.max(), test.max()),
            bins + 1,
        )

        ref_counts, _ = np.histogram(ref, bins=bin_edges)
        test_counts, _ = np.histogram(test, bins=bin_edges)

        ref_probs = np.clip(ref_counts / ref_counts.sum(), epsilon, 1.0)
        test_probs = np.clip(test_counts / test_counts.sum(), epsilon, 1.0)

        psi_score = np.sum((test_probs - ref_probs) * np.log(test_probs / ref_probs))
        drift_flag = psi_score >= self._threshold

        return ScoreDriftResult(
            score=psi_score,
            drift=drift_flag,
            details={
                "bins": bins,
                "epsilon": epsilon,
                "mode": "online" if self.is_online() else "offline",
                "reference_size": ref.size,
                "test_size": test.size
            },
        )
