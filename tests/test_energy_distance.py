import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(".."))
from drift_detectors.multivariate.energy_distance.detector import EnergyDistance


class TestEnergyDistance(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.ref = rng.normal(0, 1, size=(400, 4))
        self.test_drift = rng.normal(0.7, 1, size=(300, 4))
        self.test_no_drift = rng.normal(0, 1, size=(300, 4))

    def test_requires_reference(self):
        with self.assertRaises(ValueError):
            EnergyDistance().calculate(test_data=self.test_no_drift)

    def test_offline_drift(self):
        res = EnergyDistance(n_permutations=99).calculate(
            test_data=self.test_drift, reference_data=self.ref)
        self.assertTrue(res.drift)
        self.assertGreater(res.score, 0.0)

    def test_no_drift(self):
        res = EnergyDistance(n_permutations=99).calculate(
            test_data=self.test_no_drift, reference_data=self.ref)
        self.assertFalse(res.drift)

    def test_not_ready_small(self):
        res = EnergyDistance().calculate(
            test_data=np.random.normal(0, 1, (1, 4)), reference_data=self.ref)
        self.assertEqual(res.details.get("status"), "not_ready")

    def test_deterministic(self):
        a = EnergyDistance(seed=7, n_permutations=99).calculate(
            test_data=self.test_drift, reference_data=self.ref)
        b = EnergyDistance(seed=7, n_permutations=99).calculate(
            test_data=self.test_drift, reference_data=self.ref)
        self.assertEqual(a.details["p_value"], b.details["p_value"])

    def test_metadata_and_output_fields(self):
        res = EnergyDistance(n_permutations=99).calculate(
            test_data=self.test_drift, reference_data=self.ref)
        for key in ("p_value", "n_permutations", "alpha", "reference_size", "test_size"):
            self.assertIn(key, res.details)


if __name__ == "__main__":
    unittest.main()
