import numpy as np
import unittest

from drift_detectors.univariate.eddm.detector import EDDM
from drift_detectors.utility.drift_detection_output import StreamingDriftResult


def _structured_error_stream(
    rng: np.random.Generator,
    length: int,
    error_spacing: int,
    jitter: int = 2,
) -> np.ndarray:
    """
    Build a structured error stream where errors occur every
    ``error_spacing`` (+/- jitter) steps. This mirrors how a real
    classifier produces *correlated* errors — the regime EDDM was
    designed for — rather than an IID Bernoulli stream, on which any
    inter-error-distance method becomes pathologically noisy.
    """
    positions = []
    cursor = 0
    while cursor < length:
        cursor += max(1, error_spacing + int(rng.integers(-jitter, jitter + 1)))
        if cursor < length:
            positions.append(cursor)
    out = np.zeros(length, dtype=int)
    out[positions] = 1
    return out


class TestEDDM(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(11)

    def test_drift_detected_on_burst(self) -> None:
        # Stable inter-error spacing of ~50, then bursty errors every ~5 steps.
        stable = _structured_error_stream(self.rng, length=4000, error_spacing=50)
        bursty = _structured_error_stream(self.rng, length=2000, error_spacing=5)
        result = EDDM().calculate(np.concatenate([stable, bursty]))
        self.assertIsInstance(result, StreamingDriftResult)
        self.assertTrue(result.drift)
        self.assertGreaterEqual(result.last_index, 0)

    def test_no_drift_on_stable_structured_stream(self) -> None:
        # Stable structured spacing across the whole window — no drift expected.
        stream = _structured_error_stream(self.rng, length=8000, error_spacing=40)
        result = EDDM().calculate(stream)
        self.assertFalse(result.drift)
        self.assertEqual(result.last_index, -1)

    def test_metadata_loaded(self) -> None:
        det = EDDM()
        self.assertEqual(det.metadata.get("acronym"), "EDDM")

    def test_handles_no_errors(self) -> None:
        result = EDDM().calculate(np.zeros(500, dtype=int))
        self.assertFalse(result.drift)
        self.assertEqual(result.details["n_errors"], 0)

    def test_empty_input_raises(self) -> None:
        with self.assertRaises(ValueError):
            EDDM().calculate(np.array([]))

    def test_warning_field_present(self) -> None:
        stream = _structured_error_stream(self.rng, length=2000, error_spacing=30)
        res = EDDM().calculate(stream)
        self.assertIn("warning", res.details)


if __name__ == "__main__":
    unittest.main()
