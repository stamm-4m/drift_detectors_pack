import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(".."))
from drift_detectors import (
    HDDDM,
    MMDLinear,
    MultivariateKS,
    NNTwoSample,
    PCAReconstruction,
)

DETECTORS = [HDDDM, MMDLinear, MultivariateKS, NNTwoSample, PCAReconstruction]


def _ref(s):
    return np.random.default_rng(s).normal(0, 1, (400, 4))


def _same(s):
    return np.random.default_rng(s + 100).normal(0, 1, (300, 4))


def _shift(s, D):
    # PCA reconstruction detects out-of-subspace (covariance) drift, not a pure
    # in-subspace mean shift; the others detect a mean/location shift.
    r = np.random.default_rng(s + 200)
    if D is PCAReconstruction:
        x = r.normal(0, 1, (300, 4)); x[:, 2:] *= 2.5; return x
    return r.normal(0.7, 1, (300, 4))


class TestMultivariateExtra(unittest.TestCase):
    def test_requires_reference(self):
        for D in DETECTORS:
            with self.assertRaises(ValueError):
                D().calculate(test_data=_same(0))

    def test_no_drift_on_same(self):
        for D in DETECTORS:
            fp = sum(D().calculate(test_data=_same(s), reference_data=_ref(s)).drift for s in range(10))
            self.assertLessEqual(fp, 1, f"{D.__name__} fp={fp}/10")

    def test_drift_on_shift(self):
        for D in DETECTORS:
            det = sum(D().calculate(test_data=_shift(s, D), reference_data=_ref(s)).drift for s in range(10))
            self.assertGreaterEqual(det, 9, f"{D.__name__} det={det}/10")

    def test_metadata(self):
        for D in DETECTORS:
            m = D().metadata
            self.assertEqual(m.get("type"), "multivariate")
            self.assertEqual(m.get("mode"), "batch")
            self.assertIn("references", m)


if __name__ == "__main__":
    unittest.main()
