import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(".."))

from drift_detectors.univariate.psi.detector import PSI
from drift_detectors.utility.drift_detection_output import ScoreDriftResult


class TestPSI(unittest.TestCase):
    def test_stateless_drift_detected(self):
        detector = PSI()
        ref = np.random.normal(0, 1, 1000)
        test = np.random.normal(0.6, 1, 1000)
        result = detector.calculate(reference_data=ref, test_data=test)
        self.assertTrue(result.drift)
        self.assertGreater(result.score, 0)

    def test_stateless_no_drift(self):
        detector = PSI()
        ref = np.random.normal(0, 1, 1000)
        test = np.random.normal(0, 1, 1000)
        result = detector.calculate(reference_data=ref, test_data=test)
        self.assertFalse(result.drift)
        self.assertLess(result.score, 0.1)

    def test_stateful_offline(self):
        ref = np.random.normal(0, 1, 1000)
        test = np.random.normal(0.4, 1, 1000)
        detector = PSI(reference_data=ref)
        result = detector.calculate(test_data=test)
        self.assertTrue(result.drift)

    def test_stateful_online_accumulation(self):
        ref = np.random.normal(0, 1, 1000)
        test_batches = [np.random.normal(0.5, 1, 200) for _ in range(5)]
        detector = PSI(reference_data=ref, online=True)

        for i, batch in enumerate(test_batches):
            result = detector.calculate(test_data=batch)
            self.assertIsInstance(result, ScoreDriftResult)
            self.assertIn("test_size", result.details)
            self.assertEqual(result.details["test_size"], (i + 1) * 200)

        self.assertTrue(result.drift)

    def test_identical_data_no_drift(self):
        data = np.random.normal(0, 1, 1000)
        detector = PSI()
        result = detector.calculate(reference_data=data, test_data=data)
        self.assertAlmostEqual(result.score, 0.0, places=5)
        self.assertFalse(result.drift)

    def test_empty_input_raises(self):
        detector = PSI()
        with self.assertRaises(ValueError):
            detector.calculate(reference_data=np.array([]), test_data=np.array([1, 2]))

    def test_output_fields(self):
        detector = PSI()
        result = detector.calculate(
            reference_data=np.random.normal(0, 1, 1000),
            test_data=np.random.normal(0.3, 1, 1000),
            bins=15,
            epsilon=1e-7
        )
        self.assertEqual(result.details["bins"], 15)
        self.assertEqual(result.details["epsilon"], 1e-7)
        self.assertIn("mode", result.details)
        self.assertIn("reference_size", result.details)

    def test_metadata_loaded(self):
        detector = PSI()
        metadata = detector.metadata
        self.assertIsInstance(metadata, dict)
        self.assertIn("name", metadata)
        self.assertIn("description", metadata)

    def test_reference_override(self):
        ref_1 = np.random.normal(0, 1, 1000)
        ref_2 = np.random.normal(2, 1, 1000)
        test = np.random.normal(0, 1, 1000)

        detector = PSI(reference_data=ref_1)
        result1 = detector.calculate(test_data=test)
        result2 = detector.calculate(reference_data=ref_2, test_data=test)

        self.assertNotEqual(result1.score, result2.score)


if __name__ == "__main__":
    unittest.main()
