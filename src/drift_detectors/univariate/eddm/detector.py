"""
EDDM — Early Drift Detection Method.

EDDM (Baena-García et al., 2006) was originally introduced for binary
classification streams: it monitors the *distance between consecutive
errors* and flags drift when the distribution of these distances narrows
significantly compared to its historical maximum. In its more general form
implemented here, EDDM operates on a binary error stream (0 = correct,
1 = error). For non-classification streams, ``EDDM`` can be combined with
a thresholding step that converts continuous residuals into a binary
error signal.
"""

from __future__ import annotations

import numpy as np

from drift_detectors.drift_detector import DriftDetector
from drift_detectors.utility.drift_detection_output import StreamingDriftResult


class EDDM(DriftDetector):
    """
    Early Drift Detection Method for binary error streams.

    Parameters
    ----------
    warning_level : float, default=0.95
        Ratio of (mu + 2*sigma) to its historical maximum below which a
        warning is raised.
    drift_level : float, default=0.90
        Ratio of (mu + 2*sigma) to its historical maximum below which drift
        is signalled.
    min_n_errors : int, default=30
        Minimum number of accumulated errors before drift can be raised.
    online : bool, default=False
        Persist internal state across calls when True.

    References
    ----------
    Baena-García, M., del Campo-Ávila, J., Fidalgo, R., Bifet, A.,
    Gavaldà, R., & Morales-Bueno, R. (2006). Early drift detection method.
    Proc. ECML PKDD Workshop on Knowledge Discovery from Data Streams.
    """

    def __init__(
        self,
        warning_level: float = 0.95,
        drift_level: float = 0.90,
        min_n_errors: int = 30,
        min_consecutive_drift_signals: int = 3,
        online: bool = False,
    ) -> None:
        super().__init__(reference_data=None, online=online)
        self._warning_level = float(warning_level)
        self._drift_level = float(drift_level)
        self._min_n_errors = int(min_n_errors)
        self._min_consecutive = int(min_consecutive_drift_signals)
        self._reset_state()

    # ------------------------------------------------------------------ helpers
    def _reset_state(self) -> None:
        self._n_errors: int = 0
        self._n_seen: int = 0
        self._last_error_position: int = 0
        # mean and m2 (Welford) of distances between consecutive errors
        self._distance_mean: float = 0.0
        self._distance_m2: float = 0.0
        # historical maximum of (mu + 2*sigma)
        self._max_score: float = 0.0
        # state machine: drift may only fire from the warning state, as in
        # the original EDDM paper. This dramatically reduces false positives
        # on streams with random / IID error patterns.
        self._in_warning: bool = False
        # number of consecutive sub-drift-level errors observed since the
        # detector entered the warning state — must accumulate before firing.
        self._consecutive_sub_drift: int = 0

    def reset(self) -> None:
        """Reset the detector to its initial state."""
        self._reset_state()

    @staticmethod
    def _to_binary_error(values: np.ndarray) -> np.ndarray:
        """
        Convert input to a binary 0/1 error vector. Values strictly greater
        than 0.5 are treated as errors. This makes EDDM usable with
        either binary inputs (0/1) or thresholded continuous residuals.
        """
        return (np.asarray(values, dtype=float).ravel() > 0.5).astype(int)

    # -------------------------------------------------------------------- main
    def calculate(
        self,
        test_data: np.ndarray,
        warning_level: float | None = None,
        drift_level: float | None = None,
    ) -> StreamingDriftResult:
        """
        Update the detector with a batch of error indicators and return
        the drift status of the most recent batch.
        """
        warn = self._warning_level if warning_level is None else float(warning_level)
        drift_lvl = self._drift_level if drift_level is None else float(drift_level)

        errors = self._to_binary_error(test_data)
        if errors.size == 0:
            raise ValueError("test_data must contain at least one observation.")

        if not self.is_online():
            self._reset_state()

        drift = False
        warning = False
        last_index = -1

        for i, is_err in enumerate(errors):
            self._n_seen += 1
            if not is_err:
                continue

            self._n_errors += 1
            distance = self._n_seen - self._last_error_position
            self._last_error_position = self._n_seen

            # Welford's online mean / variance update for the distance.
            delta = distance - self._distance_mean
            self._distance_mean += delta / self._n_errors
            delta2 = distance - self._distance_mean
            self._distance_m2 += delta * delta2

            if self._n_errors < 2:
                continue

            variance = self._distance_m2 / (self._n_errors - 1)
            sigma = float(np.sqrt(max(variance, 0.0)))
            score = self._distance_mean + 2.0 * sigma

            # Phase 1: stabilisation. While we have not yet seen
            # ``min_n_errors``, do not engage the max-score tracker at
            # all — early statistics are too noisy to be a useful
            # reference. As soon as we cross the threshold, seed the
            # max-score with the *current* (now reasonable) score.
            if self._n_errors < self._min_n_errors:
                continue
            if self._max_score <= 0.0:
                self._max_score = score
                continue

            # Phase 2: post-stabilisation. Update the max-score whenever
            # the current score genuinely exceeds it.
            if score > self._max_score:
                self._max_score = score
                continue

            if self._max_score <= 0.0:
                continue

            ratio = score / self._max_score
            if ratio < warn:
                # Enter / remain in warning state.
                self._in_warning = True
                warning = True
                # Drift requires:
                #  (a) we are already in warning,
                #  (b) the ratio crosses the stricter drift_level, AND
                #  (c) we have observed enough consecutive sub-drift signals
                #      to rule out random fluctuation in the inter-error gap
                #      distribution. This is a robustification of the
                #      original Baena-García et al. (2006) state machine
                #      that prevents false positives on IID error streams.
                if ratio < drift_lvl:
                    self._consecutive_sub_drift += 1
                    if self._consecutive_sub_drift >= self._min_consecutive:
                        drift = True
                        last_index = i
                        self._reset_state()
                else:
                    # Sub-warning but not sub-drift: warning persists, but
                    # the consecutive drift counter resets.
                    self._consecutive_sub_drift = 0
            else:
                # Recovered above the warning threshold — clear state.
                self._in_warning = False
                self._consecutive_sub_drift = 0

        return StreamingDriftResult(
            last_index=last_index,
            drift=drift,
            details={
                "warning": warning,
                "warning_level": warn,
                "drift_level": drift_lvl,
                "n_errors": int(self._n_errors),
                "n_seen": int(self._n_seen),
                "mode": "online" if self.is_online() else "offline",
            },
        )
