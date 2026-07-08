"""Shared base for stateful streaming drift detectors.

Subclasses implement ``_reset_state`` and ``_update(value) -> bool`` (True when
drift fires on that value). The base handles the online/offline reset policy,
the per-batch loop, and the StreamingDriftResult. A ``self._warning`` flag may
be set by ``_update`` and is surfaced in the result details.
"""
from __future__ import annotations

import numpy as np

from drift_detectors.drift_detector import DriftDetector
from drift_detectors.utility.drift_detection_output import StreamingDriftResult


class StreamingDetector(DriftDetector):
    name: str = "abstract"

    def __init__(self, online: bool = False):
        super().__init__(reference_data=None, online=online)
        self._warning = False
        self._reset_state()

    def _reset_state(self) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        self._reset_state()

    def _update(self, value: float) -> bool:
        raise NotImplementedError

    def _extra(self) -> dict:
        return {}

    def calculate(self, test_data: np.ndarray, **kwargs) -> StreamingDriftResult:
        vals = np.asarray(test_data, dtype=float).ravel()
        if vals.size == 0:
            raise ValueError("test_data must contain at least one observation.")
        if not self.is_online():
            self._reset_state()
        drift = False
        last_index = -1
        for i, v in enumerate(vals):
            if self._update(float(v)):
                drift = True
                last_index = i
        return StreamingDriftResult(
            last_index=last_index, drift=drift,
            details={"warning": self._warning, "metric": self.name,
                     "mode": "online" if self.is_online() else "offline", **self._extra()})
