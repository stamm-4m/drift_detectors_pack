"""
Page-Hinkley change-point detector.

The Page-Hinkley test (Page, 1954) monitors a univariate stream by accumulating
the deviation between each new observation and the running mean. When the
cumulative deviation exceeds a threshold ``lambda_``, a drift is signalled.
This implementation is purely numpy-based — no external dependencies — and
follows the conventional formulation used by river / scikit-multiflow.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from drift_detectors.drift_detector import DriftDetector
from drift_detectors.utility.drift_detection_output import StreamingDriftResult


class PageHinkley(DriftDetector):
    """
    Page-Hinkley univariate streaming drift detector.

    Parameters
    ----------
    delta : float, default=0.005
        Magnitude of changes that are allowed without flagging drift
        (acts as a tolerance / minimum-amplitude parameter).
    lambda_ : float, default=50.0
        Detection threshold. Lower values increase sensitivity.
    alpha : float, default=1 - 1e-4
        Forgetting factor for the running mean (closer to 1 = longer memory).
    online : bool, default=False
        When True, internal state (mean, cumulative sum, sample count) persists
        across successive ``calculate`` calls.

    References
    ----------
    Page, E. S. (1954). Continuous inspection schemes. Biometrika 41(1/2), 100-115.
    """

    def __init__(
        self,
        delta: float = 0.005,
        lambda_: float = 50.0,
        alpha: float = 1 - 1e-4,
        online: bool = False,
    ) -> None:
        super().__init__(reference_data=None, online=online)
        self._delta = float(delta)
        self._lambda = float(lambda_)
        self._alpha = float(alpha)
        self._reset_state()

    # ------------------------------------------------------------------ helpers
    def _reset_state(self) -> None:
        self._n: int = 0
        self._mean: float = 0.0
        self._sum: float = 0.0  # cumulative deviation tracker

    def reset(self) -> None:
        """Reset the detector to its initial state."""
        self._reset_state()

    # -------------------------------------------------------------------- main
    def calculate(
        self,
        test_data: np.ndarray,
        delta: Optional[float] = None,
        lambda_: Optional[float] = None,
    ) -> StreamingDriftResult:
        """
        Feed observations into the Page-Hinkley test and return the
        drift status of the *most recent* batch.

        Parameters
        ----------
        test_data : np.ndarray
            New observation(s); a scalar, list, or 1-D array.
        delta : float, optional
            Override of the tolerance parameter for this call.
        lambda_ : float, optional
            Override of the detection threshold for this call.

        Returns
        -------
        StreamingDriftResult
            ``last_index`` is the offset within ``test_data`` where the most
            recent drift was raised (-1 if none).
        """
        delta = self._delta if delta is None else float(delta)
        lam = self._lambda if lambda_ is None else float(lambda_)

        x = np.asarray(test_data, dtype=float).ravel()
        if x.size == 0:
            raise ValueError("test_data must contain at least one observation.")

        if not self.is_online():
            self._reset_state()

        drift = False
        last_index = -1

        for i, value in enumerate(x):
            self._n += 1
            # incremental running mean
            self._mean = self._mean + (value - self._mean) / self._n
            self._sum = max(0.0, self._alpha * self._sum + (value - self._mean - delta))

            if self._sum > lam:
                drift = True
                last_index = i
                # Reset following the standard scheme so successive drifts
                # can be detected as the stream evolves.
                self._reset_state()

        return StreamingDriftResult(
            last_index=last_index,
            drift=drift,
            details={
                "delta": delta,
                "lambda": lam,
                "alpha": self._alpha,
                "mode": "online" if self.is_online() else "offline",
                "n_observations": int(x.size),
            },
        )
