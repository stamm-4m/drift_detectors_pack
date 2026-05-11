import numpy as np
import unittest

from drift_detectors.univariate.page_hinkley.detector import PageHinkley
from drift_detectors.utility.drift_detection_output import StreamingDriftResult


class TestPageHinkley(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

    def test_drift_detected(self) -> None:
        stream = np.concatenate([
            self.rng.normal(0.0, 1.0, 500),
            self.rng.normal(3.0, 1.0, 500),
        ])
        result = PageHinkley(delta=0.005, lambda_=20.0).calculate(stream)
        self.assertIsInstance(result, StreamingDriftResult)
        self.assertTrue(result.drift)
        self.assertGreater(result.last_index, 0)

    def test_no_drift_on_stationary_stream(self) -> None:
        stream = self.rng.normal(0.0, 1.0, 1000)
        result = PageHinkley(delta=0.005, lambda_=200.0).calculate(stream)
        self.assertFalse(result.drift)
        self.assertEqual(result.last_index, -1)

    def test_metadata_loaded(self) -> None:
        det = PageHinkley()
        self.assertIsInstance(det.metadata, dict)
        self.assertIn("name", det.metadata)
        self.assertEqual(det.metadata["acronym"], "PH")

    def test_online_state_persistence(self) -> None:
        det = PageHinkley(online=True, delta=0.005, lambda_=20.0)
        # Stationary first half
        for _ in range(300):
            det.calculate([self.rng.normal()])
        # Then a clear shift
        drift_seen = False
        for _ in range(500):
            res = det.calculate([self.rng.normal(loc=4.0)])
            if res.drift:
                drift_seen = True
                break
        self.assertTrue(drift_seen)

    def test_empty_input_raises(self) -> None:
        with self.assertRaises(ValueError):
            PageHinkley().calculate(np.array([]))


if __name__ == "__main__":
    unittest.main()
