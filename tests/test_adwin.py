import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(".."))

from drift_detectors.univariate.adwin.detector import Adwin
from drift_detectors.utility.drift_detection_output import StreamingDriftResult


class TestAdwin(unittest.TestCase):
    def setUp(self):
        self.detector = Adwin()
        self.online_detector = Adwin(online=True)

    def test_output_type(self):
        """Ensure ADWIN returns a StreamingDriftResult object."""
        data = np.random.normal(0, 1, 200)
        result = self.detector.calculate(data)
        self.assertIsInstance(result, StreamingDriftResult)

    def test_drift_detection_stateless(self):
        """ADWIN should detect drift in batch mode when shift occurs."""
        np.random.seed(42)
        stream = np.concatenate([
            np.random.normal(0, 1, 500),
            np.random.normal(2, 1, 500)
        ])
        result = self.detector.calculate(test_data=stream)
        self.assertTrue(result.drift)
        self.assertGreaterEqual(result.last_index, 0)
        self.assertIn("delta", result.details)

    def test_no_drift_stateless(self):
        """Should not detect drift when distribution is stable."""
        np.random.seed(42)
        stream = np.random.normal(0, 1, 1000)
        result = self.detector.calculate(test_data=stream)
        self.assertFalse(result.drift)
        self.assertEqual(result.last_index, -1)

    def test_drift_detection_online(self):
        """Online detector should accumulate and detect drift over time."""
        np.random.seed(42)
        stream = np.concatenate([
            np.random.normal(0, 1, 400),
            np.random.normal(3, 1, 400)
        ])
        drift_found = False
        for i, val in enumerate(stream):
            result = self.online_detector.calculate(test_data=[val])
            if result.drift:
                drift_found = True
                self.assertGreaterEqual(result.last_index, 0)
                break
        self.assertTrue(drift_found)

    def test_metadata_contains_mode(self):
        """Metadata should include mode (online/offline)."""
        result = self.detector.calculate(np.random.normal(0, 1, 100))
        self.assertIn("mode", result.details)
        self.assertEqual(result.details["mode"], "offline")

        result_online = self.online_detector.calculate(np.random.normal(0, 1, 100))
        self.assertIn("mode", result_online.details)
        self.assertEqual(result_online.details["mode"], "online")

    def test_custom_delta_changes_detection_sensitivity(self):
        """Lower delta should increase sensitivity (more likely to detect drift)."""
        np.random.seed(42)
        stream = np.concatenate([
            np.random.normal(0, 1, 300),
            np.random.normal(2, 1, 300)
        ])
        low_sensitivity = Adwin(delta=0.01)
        high_sensitivity = Adwin(delta=0.0001)

        r1 = low_sensitivity.calculate(stream)
        r2 = high_sensitivity.calculate(stream)

        self.assertTrue(r2.drift)
        # Not always guaranteed that r1 doesn't detect, but should be stricter
        self.assertGreaterEqual(r2.last_index, r1.last_index)

if __name__ == "__main__":
    unittest.main()
