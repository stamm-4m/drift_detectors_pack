"""Pluggable disagreement metrics used by ModelDisagreementMetric.

The package exposes one ``DisagreementMetric`` subclass per metric.
Instantiate the metrics you want and pass them to
``ModelDisagreementMetric(metrics=[...])`` -- you can register your own
metric by subclassing ``DisagreementMetric`` and implementing ``pair``.
"""
from drift_detectors.model_based.disagreement_metrics.acf import ACFDivergence
from drift_detectors.model_based.disagreement_metrics.base import DisagreementMetric
from drift_detectors.model_based.disagreement_metrics.dtw import DTWDisagreement
from drift_detectors.model_based.disagreement_metrics.lagged_corr import (
    LaggedCorrelationDisagreement,
)
from drift_detectors.model_based.disagreement_metrics.mse import MSEDisagreement
from drift_detectors.model_based.disagreement_metrics.pearson import PearsonDisagreement
from drift_detectors.model_based.disagreement_metrics.spearman import SpearmanDisagreement

__all__ = [
    "DisagreementMetric",
    "MSEDisagreement",
    "PearsonDisagreement",
    "SpearmanDisagreement",
    "DTWDisagreement",
    "LaggedCorrelationDisagreement",
    "ACFDivergence",
]
