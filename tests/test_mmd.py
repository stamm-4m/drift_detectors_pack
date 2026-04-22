import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(".."))
from drift_detectors.multivariate.mmd.detector import MMDDetector
from drift_detectors.utility.drift_detection_output import ScoreDriftResult


class TestMMDDetector(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.detector = MMDDetector(gamma=1.0, threshold=0.01)
        self.reference_data = np.random.normal(0, 1, size=(1000, 5))
        self.test_data_drift = np.random.normal(5.0, 1, size=(1000, 5))
        self.test_data_no_drift = np.random.normal(0, 1, size=(1000, 5))
        self.detector.set_reference_data(self.reference_data)

    def test_not_ready(self):
        empty_detector = MMDDetector(gamma=1.0, threshold=0.1)
        with self.assertRaises(ValueError):
            empty_detector.calculate(np.random.normal(0, 1, size=(1, 5)))

    def test_offline_drift(self):
        result = self.detector.calculate(test_data=self.test_data_drift)
        print("Offline Drift Score:", result.score)
        self.assertTrue(result.drift)

    def test_online_drift(self):
        detector = MMDDetector(gamma=1.0, threshold=0.01, online=True)
        detector.set_reference_data(self.reference_data)
        for i in range(0, len(self.test_data_drift), 100):
            batch = self.test_data_drift[i:i+100]
            result = detector.calculate(test_data=batch)
        print("Online Drift Score:", result.score)
        self.assertTrue(result.drift)

    def test_no_drift(self):
        result = self.detector.calculate(test_data=self.test_data_no_drift)
        print("No Drift Score:", result.score)
        self.assertFalse(result.drift)

    def test_metadata(self):
        result = self.detector.calculate(test_data=self.test_data_drift)
        self.assertIn("gamma", result.details)
        self.assertIn("threshold", result.details)
        self.assertIn("reference_size", result.details)
        self.assertIn("test_size", result.details)
        self.assertAlmostEqual(result.details["gamma"], 1.0)
        self.assertAlmostEqual(result.details["threshold"], 0.01)


if __name__ == "__main__":
    unittest.main()
