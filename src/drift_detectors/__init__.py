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
from drift_detectors.model_based.disagreement_metrics import (
    ACFDivergence,
    DisagreementMetric,
    DTWDisagreement,
    LaggedCorrelationDisagreement,
    MSEDisagreement,
    PearsonDisagreement,
    SpearmanDisagreement,
)
from drift_detectors.model_based.model_disagreement.detector import ModelDisagreementMetric
from drift_detectors.multivariate.c2st.detector import C2ST
from drift_detectors.multivariate.energy_distance.detector import EnergyDistance
from drift_detectors.multivariate.hdddm.detector import HDDDM
from drift_detectors.multivariate.kdq_tree.detector import KDQTree
from drift_detectors.multivariate.mmd.detector import MMD, MMDDetector
from drift_detectors.multivariate.mmd_linear.detector import MMDLinear
from drift_detectors.multivariate.mv_ks.detector import MultivariateKS
from drift_detectors.multivariate.nn_two_sample.detector import NNTwoSample
from drift_detectors.multivariate.pca_cd.detector import PCA_CD
from drift_detectors.multivariate.pca_recon.detector import PCAReconstruction
from drift_detectors.univariate.adwin.detector import ADWIN, Adwin
from drift_detectors.univariate.bhattacharyya.detector import Bhattacharyya
from drift_detectors.univariate.chi_square.detector import ChiSquare
from drift_detectors.univariate.cramer_von_mises.detector import CramerVonMises
from drift_detectors.univariate.cusum.detector import CUSUM
from drift_detectors.univariate.ddm.detector import DDM
from drift_detectors.univariate.ecdd.detector import ECDD
from drift_detectors.univariate.eddm.detector import EDDM
from drift_detectors.univariate.hddm_a.detector import HDDM_A
from drift_detectors.univariate.hellinger.detector import HellingerDistance
from drift_detectors.univariate.jensen_shannon.detector import JensenShannon
from drift_detectors.univariate.ks.detector import KS, KSDetector
from drift_detectors.univariate.kswin.detector import KSWIN
from drift_detectors.univariate.page_hinkley.detector import PageHinkley
from drift_detectors.univariate.psi.detector import PSI
from drift_detectors.univariate.rddm.detector import RDDM
from drift_detectors.univariate.stepd.detector import STEPD
from drift_detectors.univariate.total_variation.detector import TotalVariation
from drift_detectors.univariate.wasserstein.detector import Wasserstein
from drift_detectors.utility.drift_detection_output import (
    PointwiseDriftResult,
    ScoreDriftResult,
    StreamingDriftResult,
)

__all__ = [
    "DriftDetector", "get_metadata",
    "ScoreDriftResult", "StreamingDriftResult", "PointwiseDriftResult",
    "PSI", "KS", "KSDetector", "ADWIN", "Adwin", "PageHinkley", "HDDM_A", "EDDM",
    "JensenShannon", "HellingerDistance", "TotalVariation", "ChiSquare",
    "Bhattacharyya", "Wasserstein", "CramerVonMises",
    "DDM", "RDDM", "CUSUM", "ECDD", "KSWIN", "STEPD",
    "MMD", "MMDDetector", "PCA_CD", "KDQTree", "C2ST", "EnergyDistance",
    "HDDDM", "MMDLinear", "MultivariateKS", "NNTwoSample", "PCAReconstruction",
    "ModelDisagreementMetric",
    "DisagreementMetric",
    "MSEDisagreement", "PearsonDisagreement", "SpearmanDisagreement",
    "DTWDisagreement", "LaggedCorrelationDisagreement", "ACFDivergence",
]

__version__ = "0.5.0"
