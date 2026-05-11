"""Base class for pluggable model-disagreement metrics."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class DisagreementMetric(ABC):
    """Abstract base class for a single pair-wise disagreement metric.

    A disagreement metric maps two prediction arrays of equal length to a
    scalar in [0, 1], where 0 means "the two models agree" and 1 means
    "the two models maximally disagree" within the metric's chosen scale.
    """

    name: str = "abstract"

    @abstractmethod
    def pair(self, y_i: np.ndarray, y_j: np.ndarray, *, scale: float = 1.0) -> float:
        """Compute the pair-wise disagreement between two prediction vectors."""
        raise NotImplementedError

    def __call__(self, y_i, y_j, *, scale=1.0):
        return self.pair(np.asarray(y_i, dtype=float).ravel(),
                         np.asarray(y_j, dtype=float).ravel(),
                         scale=float(scale))
