"""drift_detectors -- a unified, lightweight, open-source toolkit for drift detection.

Quick start
-----------
>>> import numpy as np
>>> from drift_detectors import PSI
>>> ref  = np.random.normal(0.0, 1.0, 1000)
>>> test = np.random.normal(0.3, 1.0, 1000)
>>> result = PSI().calculate(test, ref)
>>> result.drift, round(float(result.score), 3)
(True, ...)
"""
from drift_detectors.drift_detector import DriftDetector, get_metadata
from drift_detectors.utility.drift_detection_output import (
    PointwiseDriftResult, ScoreDriftResult, StreamingDriftResult,
)

# Univariate
from drift_detectors.univariate.adwin.detector import Adwin
from drift_detectors.univariate.eddm.detector import EDDM
from drift_detectors.univariate.hddm_a.detector import HDDM_A
from drift_detectors.univariate.ks.detector import KSDetector
from drift_detectors.univariate.page_hinkley.detector import PageHinkley
from drift_detectors.univariate.psi.detector import PSI

# Multivariate
from drift_detectors.multivariate.kdq_tree.detector import KDQTree
from drift_detectors.multivariate.mmd.detector import MMDDetector
from drift_detectors.multivariate.pca_cd.detector import PCA_CD

# Model-based (v0.4.0+: pluggable disagreement metrics, predictions API)
from drift_detectors.model_based.disagreement_metrics import (
    DisagreementMetric, MSEDisagreement, PearsonDisagreement, SpearmanDisagreement,
)
from drift_detectors.model_based.model_disagreement.detector import ModelDisagreementMetric

__all__ = [
    "DriftDetector", "get_metadata",
    "ScoreDriftResult", "StreamingDriftResult", "PointwiseDriftResult",
    "PSI", "KSDetector", "Adwin", "PageHinkley", "HDDM_A", "EDDM",
    "MMDDetector", "PCA_CD", "KDQTree",
    "ModelDisagreementMetric",
    "DisagreementMetric",
    "MSEDisagreement", "PearsonDisagreement", "SpearmanDisagreement",
]

__version__ = "0.4.0"
