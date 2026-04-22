import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(".."))
from drift_detectors.multivariate.kdq_tree.detector import KDQTree
from drift_detectors.utility.drift_detection_output import ScoreDriftResult


class TestKDQTree(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.reference_data = np.random.normal(0, 1, size=(300, 5))
        self.test_data_no_drift = np.random.normal(0, 1, size=(300, 5))
        self.test_data_drift = np.random.normal(3.0, 1, size=(300, 5))  # stronger shift

    def test_stateless_drift_detected(self):
        detector = KDQTree(k_neighbors=25, alpha=0.1)
        result = detector.calculate(reference_data=self.reference_data, test_data=self.test_data_drift)

        print("Stateless Drift Score:", result.score, result.details)
        self.assertTrue(result.drift, f"Expected drift, got score={result.score}")

    def test_stateless_no_drift(self):
        detector = KDQTree(k_neighbors=25, alpha=0.1)
        result = detector.calculate(reference_data=self.reference_data, test_data=self.test_data_no_drift)

        print("Stateless No-Drift Score:", result.score, result.details)
        self.assertFalse(result.drift, f"Unexpected drift, score={result.score}")

    def test_offline_stateful_drift_detected(self):
        detector = KDQTree(k_neighbors=25, alpha=0.1)
        detector.set_reference_data(self.reference_data)
        result = detector.calculate(test_data=self.test_data_drift)

        print("Offline Drift Score:", result.score, result.details)
        self.assertTrue(result.drift, f"Expected drift, got score={result.score}")

    def test_online_accumulated_drift(self):
        detector = KDQTree(k_neighbors=25, alpha=0.1, online=True)
        detector.set_reference_data(self.reference_data)

        for i in range(0, len(self.test_data_drift), 50):
            batch = self.test_data_drift[i:i+50]
            result = detector.calculate(test_data=batch)

        print("Online Drift Score:", result.score, result.details)
        self.assertTrue(result.drift, f"Expected drift, got score={result.score}")

    def test_metadata_fields(self):
        detector = KDQTree(k_neighbors=15, alpha=0.05, ks_method="exact")
        result = detector.calculate(reference_data=self.reference_data, test_data=self.test_data_drift)

        metadata = result.details
        self.assertIn("alpha", metadata)
        self.assertIn("k_neighbors", metadata)
        self.assertIn("ks_method", metadata)
        self.assertIn("n_patches", metadata)
        self.assertIn("reference_size", metadata)
        self.assertIn("test_size", metadata)

        self.assertAlmostEqual(metadata["alpha"], 0.05)
        self.assertEqual(metadata["k_neighbors"], 15)
        self.assertEqual(metadata["ks_method"], "exact")


if __name__ == "__main__":
    unittest.main()
