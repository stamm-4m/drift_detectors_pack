import math

import numpy as np

from drift_detectors.drift_detector import DriftDetector
from drift_detectors.utility._stats import knn_indices
from drift_detectors.utility.drift_detection_output import ScoreDriftResult


class C2ST(DriftDetector):
    """Classifier Two-Sample Test (multivariate, NumPy-only).

    Trains a k-nearest-neighbour classifier to distinguish reference samples
    (label 0) from test samples (label 1). If the two distributions coincide,
    held-out accuracy is at chance (0.5 for balanced classes); the more the
    distributions differ, the higher the accuracy. Drift is flagged when the
    accuracy is significantly above chance (one-sided binomial test, normal
    approximation). Classes are balanced by subsampling, so chance = 0.5.

    Reference: Lopez-Paz & Oquab, "Revisiting Classifier Two-Sample Tests",
    ICLR 2017 (kNN classifier variant).
    """

    def __init__(self,
                 reference_data: np.ndarray = None,
                 k_neighbors: int = 5,
                 test_size: float = 0.3,
                 alpha: float = 0.05,
                 max_samples: int = 1000,
                 online: bool = False,
                 seed: int = 0):
        super().__init__(reference_data=reference_data, online=online)
        self.k_neighbors = int(k_neighbors)
        self.test_size = float(test_size)
        self.alpha = alpha
        self.max_samples = int(max_samples)
        self.seed = int(seed)
        self._test_buffer = [] if online else None

    @staticmethod
    def _normal_sf(z: float) -> float:
        return 0.5 * math.erfc(z / math.sqrt(2.0))

    def calculate(self,
                  test_data: np.ndarray,
                  reference_data: np.ndarray = None,
                  alpha: float = None) -> ScoreDriftResult:
        ref = reference_data if reference_data is not None else self._reference_data
        if ref is None:
            raise ValueError("Reference data must be provided or set before calculation.")
        ref = np.atleast_2d(np.asarray(ref, dtype=float))

        test_data = np.atleast_2d(np.asarray(test_data, dtype=float))
        if self.is_online():
            self._test_buffer.extend(test_data)
            test = np.asarray(self._test_buffer, dtype=float)
        else:
            test = test_data

        alpha = alpha if alpha is not None else self.alpha
        rng = np.random.default_rng(self.seed)

        # balance the classes so the chance level is exactly 0.5
        n = min(len(ref), len(test), self.max_samples)
        if n < max(2 * self.k_neighbors, 10):
            return ScoreDriftResult(score=None, drift=False, details={"status": "not_ready"})
        ref_s = ref[rng.choice(len(ref), n, replace=False)]
        test_s = test[rng.choice(len(test), n, replace=False)]

        # per-class train/test split (keeps the test set balanced)
        n_te = max(1, int(round(self.test_size * n)))
        n_tr = n - n_te
        if n_tr < self.k_neighbors:
            return ScoreDriftResult(score=None, drift=False, details={"status": "not_ready"})
        ref_perm = rng.permutation(n)
        test_perm = rng.permutation(n)

        X_tr = np.vstack([ref_s[ref_perm[:n_tr]], test_s[test_perm[:n_tr]]])
        y_tr = np.concatenate([np.zeros(n_tr), np.ones(n_tr)])
        X_te = np.vstack([ref_s[ref_perm[n_tr:]], test_s[test_perm[n_tr:]]])
        y_te = np.concatenate([np.zeros(n_te), np.ones(n_te)])

        # kNN vote from the training set
        idx = knn_indices(X_tr, X_te, self.k_neighbors)
        preds = (y_tr[idx].mean(axis=1) >= 0.5).astype(float)
        n_test = len(y_te)
        correct = int(np.sum(preds == y_te))
        accuracy = correct / n_test

        # one-sided binomial test vs chance 0.5 (normal approx, continuity corr.)
        z = (correct - 0.5 - 0.5 * n_test) / math.sqrt(0.25 * n_test)
        p_value = float(min(max(self._normal_sf(z), 0.0), 1.0))

        return ScoreDriftResult(
            score=float(accuracy),
            drift=bool(p_value < alpha),
            details={
                "p_value": p_value,
                "accuracy": float(accuracy),
                "chance_level": 0.5,
                "k_neighbors": self.k_neighbors,
                "n_test": n_test,
                "alpha": alpha,
                "reference_size": len(ref),
                "test_size": len(test),
                "mode": "online" if self.is_online() else "offline",
            },
        )
