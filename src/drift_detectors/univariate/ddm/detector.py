"""DDM -- Drift Detection Method (Gama et al., 2004), binary error stream."""
from __future__ import annotations

from drift_detectors.utility._streaming import StreamingDetector


class DDM(StreamingDetector):
    """Monitors the online error rate p and its std s. It records (p_min, s_min)
    at the minimum of p+s, warns when ``p+s >= p_min + 2*s_min`` and signals
    drift when ``p+s >= p_min + 3*s_min``. Input is a binary error stream
    (values > 0.5 are treated as errors)."""

    name = "ddm"

    def __init__(self, min_instances: int = 30, warning_level: float = 2.0,
                 drift_level: float = 4.0, online: bool = False):
        self.min_instances = int(min_instances)
        self.warning_level = float(warning_level)
        self.drift_level = float(drift_level)
        super().__init__(online=online)

    def _reset_state(self):
        self.n = 0
        self.p = 1.0
        self.s = 0.0
        self.p_min = float("inf")
        self.s_min = 0.0
        self._warning = False

    def _update(self, x):
        e = 1.0 if x > 0.5 else 0.0
        self.n += 1
        self.p += (e - self.p) / self.n
        self.s = (self.p * (1.0 - self.p) / self.n) ** 0.5
        if self.n < self.min_instances:
            return False
        if self.p + self.s < self.p_min + self.s_min:
            self.p_min, self.s_min = self.p, self.s
        self._warning = (self.p + self.s) >= (self.p_min + self.warning_level * self.s_min)
        if (self.p + self.s) >= (self.p_min + self.drift_level * self.s_min):
            self._reset_state()
            return True
        return False

    def _extra(self):
        return {"error_rate": self.p, "n_seen": self.n}
