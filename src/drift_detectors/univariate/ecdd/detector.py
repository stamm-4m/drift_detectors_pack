"""ECDD -- EWMA control chart for concept drift (Ross et al., 2012)."""
from __future__ import annotations

from drift_detectors.utility._streaming import StreamingDetector


class ECDD(StreamingDetector):
    """Exponentially Weighted Moving Average control chart. Tracks an EWMA ``z``
    of the stream and its time-varying control limit ``L * sigma_z``; drift
    fires when ``|z - mean| > L * sigma_z``. Operates on any numeric value
    stream (e.g., a 0/1 error stream or residuals)."""

    name = "ecdd"

    def __init__(self, lambda_: float = 0.2, warning_level: float = 3.0,
                 control_level: float = 4.5, min_instances: int = 30,
                 online: bool = False):
        self.lambda_ = float(lambda_)
        self.warning_level = float(warning_level)
        self.control_level = float(control_level)
        self.min_instances = int(min_instances)
        super().__init__(online=online)

    def _reset_state(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.z = None
        self._warning = False

    def _update(self, x):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)
        self.z = x if self.z is None else self.lambda_ * x + (1.0 - self.lambda_) * self.z
        if self.n < self.min_instances:
            return False
        std = (self.m2 / (self.n - 1)) ** 0.5 if self.n > 1 else 0.0
        lam = self.lambda_
        sigma_z = std * (lam / (2.0 - lam) * (1.0 - (1.0 - lam) ** (2 * self.n))) ** 0.5
        if sigma_z <= 0:
            return False
        dev = abs(self.z - self.mean)
        self._warning = dev > self.warning_level * sigma_z
        if dev > self.control_level * sigma_z:
            self._reset_state()
            return True
        return False
