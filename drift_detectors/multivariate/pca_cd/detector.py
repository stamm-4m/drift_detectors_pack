import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import entropy

from drift_detectors.drift_detector import DriftDetector
from drift_detectors.utility.drift_detection_output import ScoreDriftResult


class PCA_CD(DriftDetector):
    """
    PCA-CD: Detects multivariate drift based on changes in PCA component
    distributions — using both mean (CSD) and variance (KL) divergence.
    """

    def __init__(self, 
                 reference_data: np.ndarray = None,
                 n_components: int = 2, 
                 csd_threshold: float = 0.1, 
                 kl_threshold: float = 0.05,
                 online: bool = False):
        """
        Initialize the PCA-CD detector.

        Parameters:
            reference_data (np.ndarray, optional): Reference multivariate data.
            n_components (int): Number of PCA components to use.
            csd_threshold (float): Threshold for mean shift detection.
            kl_threshold (float): Threshold for variance shift detection.
            online (bool): Whether to accumulate test data across calls.
        """
        super().__init__(reference_data=reference_data, online=online)
        self._n_components = n_components
        self._csd_threshold = csd_threshold
        self._kl_threshold = kl_threshold
        self._test_buffer = [] if online else None

    def calculate(self,
                  test_data: np.ndarray,
                  reference_data: np.ndarray = None,
                  csd_threshold: float = None,
                  kl_threshold: float = None) -> ScoreDriftResult:
        """
        Perform PCA-based drift detection on multivariate data.

        Parameters:
            test_data (np.ndarray): New multivariate samples.
            reference_data (np.ndarray, optional): Overrides stored reference data.
            csd_threshold (float, optional): Threshold for mean shift (CSD).
            kl_threshold (float, optional): Threshold for variance shift (KL divergence).

        Returns:
            ScoreDriftResult: Drift score, drift flag, and internal statistics.
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

        if len(ref) < self._n_components or len(test) < self._n_components:
            return ScoreDriftResult(
                score=None,
                drift=False,
                details={"status": "not_ready"}
            )

        # Fit PCA on reference, transform both
        pca = PCA(n_components=self._n_components)
        ref_proj = pca.fit_transform(ref)
        test_proj = pca.transform(test)

        csd_thresh = csd_threshold or self._csd_threshold
        kl_thresh = kl_threshold or self._kl_threshold

        score, csd, kl = self._compute_drift_score(ref_proj, test_proj, csd_thresh, kl_thresh)
        drift_flag = score >= 1.0

        return ScoreDriftResult(
            score=score,
            drift=drift_flag,
            details={
                "csd": csd,
                "kl": kl,
                "csd_threshold": csd_thresh,
                "kl_threshold": kl_thresh,
                "mode": "online" if self.is_online() else "offline",
                "reference_size": len(ref),
                "test_size": len(test)
            }
        )

    def _symmetric_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        p = np.clip(p, 1e-8, None)
        q = np.clip(q, 1e-8, None)
        return 0.5 * (entropy(p, q) + entropy(q, p))

    def _compute_drift_score(self, ref: np.ndarray, test: np.ndarray,
                             csd_thresh: float, kl_thresh: float) -> tuple[float, float, float]:
        csd = np.sum((np.mean(test, axis=0) - np.mean(ref, axis=0)) ** 2)
        kl = self._symmetric_kl_divergence(np.var(ref, axis=0), np.var(test, axis=0))
        score = max(csd / csd_thresh, kl / kl_thresh)
        return score, csd, kl
