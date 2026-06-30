import numpy as np
from scipy.spatial import KDTree
from scipy.stats import combine_pvalues, ks_2samp

from drift_detectors.drift_detector import DriftDetector
from drift_detectors.utility.drift_detection_output import ScoreDriftResult


class KDQTree(DriftDetector):
    """KDQTree Drift Detector (multivariate).

    Compares k-nearest neighbourhoods in test vs reference data using
    KS-tests across all features. Drift score is the proportion of
    neighborhoods with significant statistical change.
    """

    def __init__(self,
                 reference_data: np.ndarray = None,
                 k_neighbors: int = 25,
                 ks_method: str = "asymp",
                 alpha: float = 0.05,
                 score_threshold: float = 0.25,
                 use_fisher: bool = True,
                 online: bool = False):
        super().__init__(reference_data=reference_data, online=online)
        self.k_neighbors = k_neighbors
        self.ks_method = ks_method
        self.alpha = alpha
        self.score_threshold = score_threshold
        self.use_fisher = use_fisher
        self._test_buffer = [] if online else None

    def calculate(self,
                  test_data: np.ndarray,
                  reference_data: np.ndarray = None,
                  alpha: float = None) -> ScoreDriftResult:
        ref = reference_data if reference_data is not None else self._reference_data
        if ref is None:
            raise ValueError("Reference data must be provided or set before calculation.")

        test_data = np.atleast_2d(test_data)
        if self.is_online():
            self._test_buffer.extend(test_data)
            test = np.array(self._test_buffer)
        else:
            test = test_data

        if len(test) < self.k_neighbors or len(ref) < self.k_neighbors:
            return ScoreDriftResult(score=None, drift=False, details={"status": "not_ready"})

        # KD-trees: query the reference tree with the reference points to define
        # stable neighbourhoods, then query the test tree with the same
        # reference anchors so each pair of patches is spatially aligned.
        tree_ref = KDTree(ref)
        tree_test = KDTree(test)
        ref_neighbors_idx = tree_ref.query(ref, k=self.k_neighbors)[1]
        test_neighbors_idx = tree_test.query(ref, k=self.k_neighbors)[1]

        p_values = []
        for ref_idx, test_idx in zip(ref_neighbors_idx, test_neighbors_idx, strict=False):
            ref_patch = ref[ref_idx]
            test_patch = test[test_idx]
            dim_p = [
                ks_2samp(ref_patch[:, d], test_patch[:, d], method=self.ks_method)[1]
                for d in range(ref.shape[1])
            ]
            if self.use_fisher:
                _, combined_p = combine_pvalues(dim_p)
                p_values.append(combined_p)
            else:
                p_values.append(np.mean(dim_p))

        alpha = alpha if alpha is not None else self.alpha
        score = float(np.mean(np.array(p_values) < alpha))
        drift_flag = score >= self.score_threshold

        return ScoreDriftResult(
            score=score,
            drift=drift_flag,
            details={
                "alpha": alpha,
                "score_threshold": self.score_threshold,
                "k_neighbors": self.k_neighbors,
                "ks_method": self.ks_method,
                "use_fisher": self.use_fisher,
                "n_patches": len(p_values),
                "mode": "online" if self.is_online() else "offline",
                "reference_size": len(ref),
                "test_size": len(test),
            },
        )
