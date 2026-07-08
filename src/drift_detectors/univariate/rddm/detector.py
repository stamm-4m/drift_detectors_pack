"""RDDM -- Reactive Drift Detection Method (Barros et al., 2017)."""
from __future__ import annotations

from drift_detectors.utility._streaming import StreamingDetector


class RDDM(StreamingDetector):
    """A reactive variant of DDM with more sensitive thresholds and a
    ``max_concept_size``: if a concept stays stable for too long, the running
    (p_min, s_min) baseline is recomputed so the detector stays responsive.
    Input is a binary error stream (values > 0.5 = error)."""

    name = "rddm"

    def __init__(self, min_instances: int = 129, warning_level: float = 3.0,
                 drift_level: float = 4.0, max_concept_size: int = 40000,
                 online: bool = False):
        self.min_instances = int(min_instances)
        self.warning_level = float(warning_level)
        self.drift_level = float(drift_level)
        self.max_concept_size = int(max_concept_size)
        super().__init__(online=online)

    def _reset_state(self):
        self.n = 0
        self.p = 1.0
        self.s = 0.0
        self.p_min = float("inf")
        self.s_min = 0.0
        self.stable = 0
        self._warning = False

    def _update(self, x):
        e = 1.0 if x > 0.5 else 0.0
        self.n += 1
        self.stable += 1
        self.p += (e - self.p) / self.n
        self.s = (self.p * (1.0 - self.p) / self.n) ** 0.5
        if self.stable > self.max_concept_size:   # reactive re-baseline
            self.p_min, self.s_min = float("inf"), 0.0
            self.stable = 0
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
