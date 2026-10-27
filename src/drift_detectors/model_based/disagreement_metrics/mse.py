"""Output-magnitude disagreement, scale-normalised."""
from __future__ import annotations

import numpy as np

from drift_detectors.model_based.disagreement_metrics.base import DisagreementMetric


class MSEDisagreement(DisagreementMetric):
    """Root-MSE between two prediction vectors, normalised by a scale factor.

    The scale is set by the orchestrator (ModelDisagreementMetric
    ``scale``): by default the standard deviation of the prediction stack
    of the window, so the metric reads as "fraction of the typical
    prediction spread"; a fixed value or the spread of reference
    predictions can be used instead (0.5.2).
    """
    name = "mse"
    kind = "error"

    def pair(self, y_i, y_j, *, scale=1.0):
        rmse = float(np.sqrt(np.mean((y_i - y_j) ** 2)))
        if scale <= 0 or not np.isfinite(scale):
            scale = 1.0
        return float(min(rmse / scale, 1.0))
