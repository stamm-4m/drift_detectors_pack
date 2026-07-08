import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(".."))
from drift_detectors.multivariate.c2st.detector import C2ST


class TestC2ST(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.ref = rng.normal(0, 1, size=(600, 4))
        self.test_drift = rng.normal(0.8, 1, size=(600, 4))
        self.test_no_drift = rng.normal(0, 1, size=(600, 4))

    def test_requires_reference(self):
        with self.assertRaises(ValueError):
            C2ST().calculate(test_data=self.test_no_drift)

    def test_offline_drift(self):
        res = C2ST(k_neighbors=5).calculate(
            test_data=self.test_drift, reference_data=self.ref)
        self.assertTrue(res.drift)
        self.assertGreater(res.score, 0.5)

    def test_no_drift_near_chance(self):
        res = C2ST(k_neighbors=5).calculate(
            test_data=self.test_no_drift, reference_data=self.ref)
        self.assertFalse(res.drift)
        self.assertLess(res.score, 0.6)   # accuracy near chance 0.5

    def test_not_ready_small(self):
        res = C2ST().calculate(
            test_data=np.random.normal(0, 1, (3, 4)), reference_data=self.ref)
        self.assertEqual(res.details.get("status"), "not_ready")

    def test_deterministic(self):
        a = C2ST(seed=3).calculate(test_data=self.test_drift, reference_data=self.ref)
        b = C2ST(seed=3).calculate(test_data=self.test_drift, reference_data=self.ref)
        self.assertEqual(a.score, b.score)

    def test_output_fields(self):
        res = C2ST().calculate(test_data=self.test_drift, reference_data=self.ref)
        for key in ("p_value", "accuracy", "chance_level", "k_neighbors", "n_test"):
            self.assertIn(key, res.details)


if __name__ == "__main__":
    unittest.main()
