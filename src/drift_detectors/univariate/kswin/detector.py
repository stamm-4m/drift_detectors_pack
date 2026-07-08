"""KSWIN -- Kolmogorov-Smirnov Windowing drift detector (Raab et al., 2020)."""
from __future__ import annotations

import numpy as np

from drift_detectors.utility._stats import ks_2samp
from drift_detectors.utility._streaming import StreamingDetector


class KSWIN(StreamingDetector):
    """Keeps a sliding window of the most recent values and periodically runs a
    two-sample KS test between the newest ``stat_size`` samples and a random
    sample of the older part of the window; drift fires when the KS p-value
    falls below ``alpha``. Operates on any numeric value stream."""

    name = "kswin"

    def __init__(self, alpha: float = 0.00002, window_size: int = 100,
                 stat_size: int = 30, seed: int = 0, online: bool = False):
        self.alpha = float(alpha)
        self.window_size = int(window_size)
        self.stat_size = int(stat_size)
        self.seed = int(seed)
        super().__init__(online=online)

    def _reset_state(self):
        self.window = []
        self._rng = np.random.default_rng(self.seed)
        self._warning = False

    def _update(self, x):
        self.window.append(x)
        if len(self.window) > self.window_size:
            self.window.pop(0)
        if len(self.window) < self.window_size:
            return False
        recent = np.asarray(self.window[-self.stat_size:], dtype=float)
        older = np.asarray(self.window[:-self.stat_size], dtype=float)
        sample = self._rng.choice(older, size=self.stat_size, replace=True)
        _, p = ks_2samp(sample, recent)
        if p < self.alpha:
            self.window = self.window[-self.stat_size:]   # keep recent, drop stale
            return True
        return False
