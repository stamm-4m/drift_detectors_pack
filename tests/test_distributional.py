import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(".."))
from drift_detectors import (
    Bhattacharyya,
    ChiSquare,
    CramerVonMises,
    HellingerDistance,
    JensenShannon,
    TotalVariation,
    Wasserstein,
)

DETECTORS = [JensenShannon, HellingerDistance, TotalVariation, ChiSquare,
             Bhattacharyya, Wasserstein, CramerVonMises]


class TestDistributional(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.ref = rng.normal(0, 1, 2000)
        self.same = rng.normal(0, 1, 2000)
        self.shift = rng.normal(0.8, 1, 2000)

    def test_requires_reference(self):
        for D in DETECTORS:
            with self.assertRaises(ValueError):
                D().calculate(test_data=self.same)

    def test_no_drift_on_same(self):
        for D in DETECTORS:
            self.assertFalse(D().calculate(test_data=self.same, reference_data=self.ref).drift, D.__name__)

    def test_drift_on_shift(self):
        for D in DETECTORS:
            self.assertTrue(D().calculate(test_data=self.shift, reference_data=self.ref).drift, D.__name__)

    def test_score_increases_with_shift(self):
        for D in DETECTORS:
            s0 = D().calculate(test_data=self.same, reference_data=self.ref).score
            s1 = D().calculate(test_data=self.shift, reference_data=self.ref).score
            self.assertGreater(s1, s0, D.__name__)

    def test_metadata_fields(self):
        for D in DETECTORS:
            det = D()
            self.assertEqual(det.metadata.get("type"), "univariate")
            self.assertEqual(det.metadata.get("mode"), "batch")
            self.assertIn("references", det.metadata)


if __name__ == "__main__":
    unittest.main()
