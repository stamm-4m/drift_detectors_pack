import numpy as np
import unittest

from drift_detectors.univariate.hddm_a.detector import HDDM_A
from drift_detectors.utility.drift_detection_output import StreamingDriftResult


class TestHDDMA(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(7)

    def test_drift_detected(self) -> None:
        stream = np.concatenate([
            self.rng.normal(0.0, 0.5, 800),
            self.rng.normal(3.0, 0.5, 1500),
        ])
        result = HDDM_A(drift_confidence=0.05).calculate(stream)
        self.assertIsInstance(result, StreamingDriftResult)
        self.assertTrue(result.drift)
        self.assertGreaterEqual(result.last_index, 0)

    def test_no_drift_on_stationary_stream(self) -> None:
        stream = self.rng.normal(0.0, 1.0, 2000)
        result = HDDM_A(drift_confidence=1e-6).calculate(stream)
        self.assertFalse(result.drift)
        self.assertEqual(result.last_index, -1)

    def test_metadata_loaded(self) -> None:
        det = HDDM_A()
        self.assertEqual(det.metadata.get("acronym"), "HDDM-A")

    def test_warning_field_present(self) -> None:
        stream = self.rng.normal(0.0, 1.0, 200)
        res = HDDM_A().calculate(stream)
        self.assertIn("warning", res.details)

    def test_empty_input_raises(self) -> None:
        with self.assertRaises(ValueError):
            HDDM_A().calculate(np.array([]))


if __name__ == "__main__":
    unittest.main()
