"""
HDDM-A — Hoeffding's Drift Detection Method (A-test variant).

HDDM-A (Frías-Blanco et al., 2015) tests whether the mean of the recent stream
window differs from the mean of the historical stream by more than what is
permitted by Hoeffding's inequality. It maintains two empirical means — one
"long" (everything seen so far) and one "short" (sliding test window) — and
flags drift when the gap exceeds the Hoeffding bound for a chosen confidence
level. This implementation is pure-numpy.
"""

from __future__ import annotations

import numpy as np

from drift_detectors.drift_detector import DriftDetector
from drift_detectors.utility.drift_detection_output import StreamingDriftResult


class HDDM_A(DriftDetector):
    """
    HDDM-A streaming drift detector for univariate data.

    The detector keeps two empirical estimators (a long-term reference and a
    short-term test) and signals drift / warning whenever the absolute
    difference between their means exceeds the Hoeffding bound at the
    configured confidence levels.

    Parameters
    ----------
    drift_confidence : float, default=0.001
        Confidence level for drift detection (smaller = stricter).
    warning_confidence : float, default=0.005
        Confidence level for raising a warning before drift.
    two_sided : bool, default=True
        If True, both increases and decreases of the mean are flagged.
    online : bool, default=False
        Persist internal state across calls when True.

    References
    ----------
    Frías-Blanco, I., del Campo-Ávila, J., Ramos-Jiménez, G., Morales-Bueno,
    R., Ortiz-Díaz, A., & Caballero-Mota, Y. (2015). Online and non-parametric
    drift detection methods based on Hoeffding's bounds.
    IEEE Transactions on Knowledge and Data Engineering, 27(3), 810-823.
    """

    def __init__(
        self,
        drift_confidence: float = 0.001,
        warning_confidence: float = 0.005,
        two_sided: bool = True,
        online: bool = False,
    ) -> None:
        super().__init__(reference_data=None, online=online)
        self._drift_conf = float(drift_confidence)
        self._warn_conf = float(warning_confidence)
        self._two_sided = bool(two_sided)
        self._reset_state()

    # ------------------------------------------------------------------ helpers
    def _reset_state(self) -> None:
        # cumulative ("long") estimator
        self._n_total = 0
        self._sum_total = 0.0
        # short / cut-point estimator (best candidate for change)
        self._n_cut = 0
        self._sum_cut = 0.0
        # bookkeeping for the running minimum-mean cut point
        self._best_cut_mean = float("inf")

    def reset(self) -> None:
        """Reset the detector to its initial state."""
        self._reset_state()

    @staticmethod
    def _hoeffding_bound(n: int, conf: float) -> float:
        """Hoeffding bound for an estimator with [0, 1] -range observations."""
        if n <= 0:
            return float("inf")
        return float(np.sqrt(np.log(1.0 / conf) / (2.0 * n)))

    # -------------------------------------------------------------------- main
    def calculate(
        self,
        test_data: np.ndarray,
        drift_confidence: float | None = None,
        warning_confidence: float | None = None,
    ) -> StreamingDriftResult:
        """
        Update the detector with a batch of observations and return the
        drift status of the most recent batch.

        Note
        ----
        Observations are internally rescaled to [0, 1] using a min-max
        transform that is updated online. This makes the Hoeffding bound
        well-defined even for unbounded sensor signals.
        """
        drift_conf = self._drift_conf if drift_confidence is None else float(drift_confidence)
        warn_conf = self._warn_conf if warning_confidence is None else float(warning_confidence)

        x = np.asarray(test_data, dtype=float).ravel()
        if x.size == 0:
            raise ValueError("test_data must contain at least one observation.")

        if not self.is_online():
            self._reset_state()

        # Rescale to [0, 1] using observed min/max for this call (and history).
        # This is a pragmatic approximation of the original HDDM-A which assumes
        # bounded inputs; it preserves the relative behaviour of the test.
        if self._n_total == 0:
            ref_min = float(x.min())
            ref_max = float(x.max())
        else:
            ref_min = float(min(self._sum_total / max(self._n_total, 1), x.min()))
            ref_max = float(max(self._sum_total / max(self._n_total, 1), x.max()))
        span = max(ref_max - ref_min, 1e-12)
        x_scaled = np.clip((x - ref_min) / span, 0.0, 1.0)

        drift = False
        warning = False
        last_index = -1

        for i, value in enumerate(x_scaled):
            self._n_total += 1
            self._sum_total += float(value)
            mean_total = self._sum_total / self._n_total

            # Update the running cut-point estimator: track the minimum mean
            # observed so far as the "reference" point against which the
            # current mean is compared.
            if mean_total < self._best_cut_mean:
                self._best_cut_mean = mean_total
                self._n_cut = self._n_total
                self._sum_cut = self._sum_total

            if self._n_cut < self._n_total:
                n_recent = self._n_total - self._n_cut
                sum_recent = self._sum_total - self._sum_cut
                mean_recent = sum_recent / n_recent

                eps_drift = self._hoeffding_bound(self._n_cut, drift_conf) + \
                            self._hoeffding_bound(n_recent, drift_conf)
                eps_warn = self._hoeffding_bound(self._n_cut, warn_conf) + \
                           self._hoeffding_bound(n_recent, warn_conf)

                gap = abs(mean_recent - self._best_cut_mean) if self._two_sided \
                    else max(0.0, mean_recent - self._best_cut_mean)

                if gap > eps_drift:
                    drift = True
                    last_index = i
                    self._reset_state()
                elif gap > eps_warn:
                    warning = True

        return StreamingDriftResult(
            last_index=last_index,
            drift=drift,
            details={
                "warning": warning,
                "drift_confidence": drift_conf,
                "warning_confidence": warn_conf,
                "two_sided": self._two_sided,
                "mode": "online" if self.is_online() else "offline",
                "n_observations": int(x.size),
            },
        )
