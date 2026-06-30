"""Base class for pluggable model-disagreement metrics."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class DisagreementMetric(ABC):
    """Abstract base class for a single pair-wise disagreement metric.

    A disagreement metric maps two prediction arrays of equal length to a
    scalar in [0, 1], where 0 means "the two models agree" and 1 means
    "the two models maximally disagree" within the metric's chosen scale.

    Each concrete subclass must declare:

    - name: a short identifier (used as the dictionary key in the
      detector's output, e.g. "mse").
    - kind: the family the metric belongs to. Two families are
      currently in use:

        * "error"        -- metrics that compare prediction magnitudes
                            (e.g. MSE-, MAE-, RMSE-based disagreements).
        * "correlation"  -- metrics that compare prediction shape or
                            ordering (Pearson, Spearman, Kendall).

      The orchestrator (ModelDisagreementMetric) aggregates metrics
      within a family but never across families, because magnitude errors
      and correlation-based disagreements are dimensionally and
      conceptually inhomogeneous: a model pair can be highly correlated
      yet far apart in magnitude (one a scaled copy of the other), so the
      mean of an MSE-style number and a 1-|rho| number hides more than it
      summarises.
    """

    name: str = "abstract"
    kind: str = "abstract"  # must be "error" or "correlation"

    @abstractmethod
    def pair(self, y_i: np.ndarray, y_j: np.ndarray, *, scale: float = 1.0) -> float:
        """Compute the pair-wise disagreement between two prediction vectors."""
        raise NotImplementedError

    def __call__(self, y_i, y_j, *, scale=1.0):
        return self.pair(np.asarray(y_i, dtype=float).ravel(),
                         np.asarray(y_j, dtype=float).ravel(),
                         scale=float(scale))
