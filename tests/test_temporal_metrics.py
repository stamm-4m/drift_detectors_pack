import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(".."))
from drift_detectors import (
    ACFDivergence,
    DTWDisagreement,
    LaggedCorrelationDisagreement,
    ModelDisagreementMetric,
)

METRICS = [DTWDisagreement, LaggedCorrelationDisagreement, ACFDivergence]


class TestTemporalMetrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        t = np.linspace(0, 10, 300)
        base = np.sin(t)
        self.a = base + rng.normal(0, 0.02, 300)
        self.b = base + rng.normal(0, 0.02, 300)          # near-identical
        self.c = np.sin(t + 1.5) + rng.normal(0, 0.3, 300)  # temporally divergent

    def test_kind_is_temporal(self):
        for M in METRICS:
            self.assertEqual(M().kind, "temporal")

    def test_bounded_and_identical_near_zero(self):
        for M in METRICS:
            v_same = M().pair(self.a, self.b, scale=1.0)
            self.assertGreaterEqual(v_same, 0.0)
            self.assertLessEqual(v_same, 1.0)
            self.assertLess(v_same, 0.15)   # near-identical series ~ 0

    def test_divergent_scores_higher(self):
        for M in METRICS:
            same = M().pair(self.a, self.b, scale=1.0)
            div = M().pair(self.a, self.c, scale=1.0)
            self.assertGreater(div, same)

    def test_deterministic(self):
        for M in METRICS:
            self.assertEqual(M().pair(self.a, self.c, scale=1.0),
                             M().pair(self.a, self.c, scale=1.0))

    def test_mdm_temporal_family(self):
        metrics = [M() for M in METRICS]
        res = ModelDisagreementMetric(metrics=metrics, threshold=0.2).calculate(
            predictions=[self.a, self.b, self.a, self.c])
        self.assertIn("temporal", res.details["score_by_kind"])
        for name in ("dtw", "lagged_corr", "acf"):
            self.assertIn(name, res.details["metric_means"])


if __name__ == "__main__":
    unittest.main()
