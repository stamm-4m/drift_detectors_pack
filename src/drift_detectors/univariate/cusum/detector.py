"""CUSUM -- two-sided standardized cumulative-sum change detector."""
from __future__ import annotations

import numpy as np

from drift_detectors.utility._streaming import StreamingDetector


class CUSUM(StreamingDetector):
    """Two-sided CUSUM. A baseline mean/std is estimated from the first
    ``min_instances`` observations and then frozen; subsequent standardized
    deviations accumulate beyond a slack ``k`` into positive/negative sums and
    drift fires when either exceeds the threshold ``h``. Any numeric value
    stream (e.g., residuals)."""

    name = "cusum"

    def __init__(self, k: float = 0.5, h: float = 12.0, min_instances: int = 30,
                 online: bool = False):
        self.k = float(k)
        self.h = float(h)
        self.min_instances = int(min_instances)
        super().__init__(online=online)

    def _reset_state(self):
        self._warm = []
        self.mu = None
        self.sd = 1.0
        self.g_pos = 0.0
        self.g_neg = 0.0
        self._warning = False

    def _update(self, x):
        if self.mu is None:
            self._warm.append(x)
            if len(self._warm) >= self.min_instances:
                arr = np.asarray(self._warm, dtype=float)
                self.mu = float(arr.mean())
                self.sd = float(arr.std()) or 1.0
            return False
        z = (x - self.mu) / self.sd
        self.g_pos = max(0.0, self.g_pos + z - self.k)
        self.g_neg = max(0.0, self.g_neg - z - self.k)
        self._warning = (self.g_pos > 0.5 * self.h) or (self.g_neg > 0.5 * self.h)
        if self.g_pos > self.h or self.g_neg > self.h:
            self._reset_state()
            return True
        return False
