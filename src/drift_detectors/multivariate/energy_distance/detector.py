import numpy as np

from drift_detectors.drift_detector import DriftDetector
from drift_detectors.utility.drift_detection_output import ScoreDriftResult


def _pairwise_euclidean(A: np.ndarray) -> np.ndarray:
    """Full N x N matrix of Euclidean distances between rows of A."""
    sq = np.einsum("ij,ij->i", A, A)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (A @ A.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


class EnergyDistance(DriftDetector):
    """Energy-distance two-sample drift detector (multivariate).

    The energy distance (Szekely & Rizzo, 2013) between the reference and test
    samples is ``E = 2*mean||x-y|| - mean||x-x'|| - mean||y-y'||`` (Euclidean).
    It is zero iff the two distributions coincide and grows with any
    distributional difference (location, scale or shape). Significance is
    assessed by a label-permutation test, so no distributional assumptions are
    made. Pure NumPy.
    """

    def __init__(self,
                 reference_data: np.ndarray = None,
                 alpha: float = 0.05,
                 n_permutations: int = 199,
                 max_samples: int = 500,
                 online: bool = False,
                 seed: int = 0):
        super().__init__(reference_data=reference_data, online=online)
        self.alpha = alpha
        self.n_permutations = int(n_permutations)
        self.max_samples = int(max_samples)
        self.seed = int(seed)
        self._test_buffer = [] if online else None

    @staticmethod
    def _energy_from_idx(D, idx_x, idx_y):
        dxy = D[np.ix_(idx_x, idx_y)].mean()
        dxx = D[np.ix_(idx_x, idx_x)].mean()
        dyy = D[np.ix_(idx_y, idx_y)].mean()
        return float(2.0 * dxy - dxx - dyy)

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

        # cap sample sizes so the permutation test stays tractable
        def _cap(A):
            if len(A) > self.max_samples:
                return A[rng.choice(len(A), self.max_samples, replace=False)]
            return A
        ref_s, test_s = _cap(ref), _cap(test)
        if len(ref_s) < 2 or len(test_s) < 2:
            return ScoreDriftResult(score=None, drift=False, details={"status": "not_ready"})

        combined = np.vstack([ref_s, test_s])
        n = len(ref_s)
        total = len(combined)
        D = _pairwise_euclidean(combined)

        observed = self._energy_from_idx(D, np.arange(n), np.arange(n, total))
        count = 0
        for _ in range(self.n_permutations):
            perm = rng.permutation(total)
            if self._energy_from_idx(D, perm[:n], perm[n:]) >= observed:
                count += 1
        p_value = (count + 1) / (self.n_permutations + 1)

        return ScoreDriftResult(
            score=observed,
            drift=bool(p_value < alpha),
            details={
                "p_value": p_value,
                "n_permutations": self.n_permutations,
                "alpha": alpha,
                "reference_size": len(ref),
                "test_size": len(test),
                "sampled_reference": len(ref_s),
                "sampled_test": len(test_s),
                "mode": "online" if self.is_online() else "offline",
            },
        )
