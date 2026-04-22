import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(".."))
from drift_detectors.multivariate.pca_cd.detector import PCA_CD
from drift_detectors.utility.drift_detection_output import ScoreDriftResult


class TestPCACD(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.detector = PCA_CD(n_components=2, csd_threshold=0.03, kl_threshold=0.05)
        self.reference_data = np.random.normal(0, 1, size=(1000, 5))
        self.test_data_drift = np.random.normal(3.5, 1, size=(1000, 5))
        self.test_data_no_drift = np.random.normal(0, 1, size=(1000, 5))
        self.detector.set_reference_data(self.reference_data)

    def test_not_ready(self):
        result = self.detector.calculate(np.random.normal(0, 1, size=5))
        self.assertIsNone(result.score)
        self.assertFalse(result.drift)
        self.assertEqual(result.details.get("status"), "not_ready")

    def test_offline_drift(self):
        result = self.detector.calculate(test_data=self.test_data_drift)
        self.assertTrue(result.drift)

    def test_online_drift(self):
        detector = PCA_CD(n_components=2, csd_threshold=0.02, kl_threshold=0.005, online=True)
        detector.set_reference_data(self.reference_data)
        for i in range(0, len(self.test_data_drift), 50):
            batch = self.test_data_drift[i:i+50]
            result = detector.calculate(test_data=batch)
        self.assertTrue(result.drift)

    def test_no_drift(self):
        result = self.detector.calculate(test_data=self.test_data_no_drift)
        self.assertFalse(result.drift)

    def test_metadata(self):
        result = self.detector.calculate(test_data=self.test_data_drift)
        self.assertIn("csd_threshold", result.details)
        self.assertIn("kl_threshold", result.details)
        self.assertAlmostEqual(result.details["csd_threshold"], 0.03)
        self.assertAlmostEqual(result.details["kl_threshold"], 0.05)


if __name__ == "__main__":
    unittest.main()
