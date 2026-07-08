"""STEPD -- Statistical Test of Equal Proportions drift detector
(Nishida & Yamauchi, 2007), binary error/accuracy stream."""
from __future__ import annotations

import math

from drift_detectors.utility._streaming import StreamingDetector


class STEPD(StreamingDetector):
    """Compares the accuracy over a recent window to the accuracy over all older
    observations with a two-proportion z-test; warning/drift fire when the
    p-value drops below ``alpha_w`` / ``alpha_d``. Input is a binary error
    stream (values > 0.5 = error; accuracy = 1 - error)."""

    name = "stepd"

    def __init__(self, window_size: int = 30, alpha_warning: float = 0.01,
                 alpha_drift: float = 0.00001, online: bool = False):
        self.window_size = int(window_size)
        self.alpha_warning = float(alpha_warning)
        self.alpha_drift = float(alpha_drift)
        super().__init__(online=online)

    def _reset_state(self):
        self.recent = []          # recent correct/incorrect (1=correct)
        self.n_old = 0
        self.correct_old = 0
        self._warning = False

    @staticmethod
    def _sf(z):
        return 0.5 * math.erfc(z / math.sqrt(2.0))

    def _update(self, x):
        correct = 0 if x > 0.5 else 1
        self.recent.append(correct)
        if len(self.recent) > self.window_size:
            self.n_old += 1
            self.correct_old += self.recent.pop(0)
        if len(self.recent) < self.window_size or self.n_old < self.window_size:
            return False
        n_r = len(self.recent)
        r_r = sum(self.recent)
        n_o, r_o = self.n_old, self.correct_old
        p_hat = (r_r + r_o) / (n_r + n_o)
        denom = p_hat * (1.0 - p_hat) * (1.0 / n_r + 1.0 / n_o)
        if denom <= 0:
            return False
        z = abs(r_r / n_r - r_o / n_o) / math.sqrt(denom)
        pval = 2.0 * self._sf(z)
        self._warning = pval < self.alpha_warning
        if pval < self.alpha_drift:
            self._reset_state()
            return True
        return False
