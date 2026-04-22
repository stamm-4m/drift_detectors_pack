import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(".."))
from drift_detectors.univariate.ks.detector import KSDetector
from drift_detectors.utility.drift_detection_output import ScoreDriftResult


class TestKSDetector(unittest.TestCase):

    def setUp(self):
        self.reference = np.random.normal(0, 1, 1000)
        self.detector = KSDetector(alpha=0.05)

    def test_stateless_no_drift(self):
        test = np.random.normal(0, 1, 1000)
        result = self.detector.calculate(reference_data=self.reference, test_data=test)
        self.assertIsInstance(result, ScoreDriftResult)
        self.assertFalse(result.drift)

    def test_stateless_drift(self):
        test = np.random.normal(1.0, 1, 1000)
        result = self.detector.calculate(reference_data=self.reference, test_data=test)
        self.assertTrue(result.drift)
        self.assertGreater(result.score, 0.1)

    def test_stateful_offline(self):
        self.detector.set_reference_data(self.reference)
        test = np.random.normal(0.5, 1, 1000)
        result = self.detector.calculate(test_data=test)
        self.assertIsInstance(result, ScoreDriftResult)
        self.assertIn("p_value", result.details)

    def test_stateful_online_mode(self):
        detector = KSDetector(alpha=0.05, online=True)
        detector.set_reference_data(self.reference)

        stream = np.random.normal(0.5, 1, 500)
        for i in range(0, len(stream), 100):
            batch = stream[i:i+100]
            result = detector.calculate(test_data=batch)

        self.assertIsInstance(result, ScoreDriftResult)
        self.assertIn("test_size", result.details)
        self.assertGreaterEqual(result.details["test_size"], 100)

    def test_identical_input(self):
        result = self.detector.calculate(reference_data=self.reference, test_data=self.reference)
        self.assertAlmostEqual(result.score, 0.0, places=3)
        self.assertFalse(result.drift)

    def test_metadata_fields(self):
        test = np.random.normal(0.5, 1, 1000)
        result = self.detector.calculate(reference_data=self.reference, test_data=test, alpha=0.03)
        self.assertIn("p_value", result.details)
        self.assertIn("alpha", result.details)
        self.assertAlmostEqual(result.details["alpha"], 0.03, places=6)


if __name__ == "__main__":
    unittest.main()
