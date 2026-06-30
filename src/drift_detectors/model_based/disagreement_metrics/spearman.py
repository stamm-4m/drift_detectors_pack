"""Pair-wise (1 - Spearman rank correlation) / 2 disagreement."""
from __future__ import annotations

import numpy as np

from drift_detectors.model_based.disagreement_metrics.base import DisagreementMetric
from drift_detectors.model_based.disagreement_metrics.pearson import _pearson


def _rankdata(a):
    a = np.asarray(a, dtype=float).ravel()
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros_like(counts, dtype=float)
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    return ranks


class SpearmanDisagreement(DisagreementMetric):
    name = "spearman"
    kind = "correlation"

    def pair(self, y_i, y_j, *, scale=1.0):
        return float(np.clip(1.0 - _pearson(_rankdata(y_i), _rankdata(y_j)), 0.0, 2.0)) / 2.0
