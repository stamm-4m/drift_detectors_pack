"""Behavioural tests for the streaming change detectors.

Streaming detectors have an inherent average-run-length: on a long stationary
stream they can raise the occasional false alarm (they are not p<alpha tests).
The fair criteria are therefore (a) high detection power on a real change and
(b) a *bounded* false-alarm rate on stationary data -- not zero.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(".."))
from drift_detectors import CUSUM, DDM, ECDD, KSWIN, RDDM, STEPD

ERROR_BASED = [DDM, RDDM, STEPD]
VALUE_BASED = [CUSUM, ECDD, KSWIN]
ALL = ERROR_BASED + VALUE_BASED


def _err_stat(s, n=2000):
    return (np.random.default_rng(s).random(n) < 0.1).astype(float)


def _err_shift(s):
    r = np.random.default_rng(s)
    return np.concatenate([(r.random(1000) < 0.1).astype(float),
                           (r.random(1000) < 0.5).astype(float)])


def _val_stat(s, n=2000):
    return np.random.default_rng(s).normal(0, 1, n)


def _val_shift(s):
    r = np.random.default_rng(s)
    return np.concatenate([r.normal(0, 1, 1000), r.normal(2, 1, 1000)])


class TestStreaming(unittest.TestCase):
    def test_empty_raises(self):
        for D in ALL:
            with self.assertRaises(ValueError):
                D().calculate(np.array([]))

    def test_high_detection_power(self):
        for D in ERROR_BASED:
            hits = sum(D().calculate(_err_shift(s)).drift for s in range(10))
            self.assertGreaterEqual(hits, 9, f"{D.__name__} detection {hits}/10")
        for D in VALUE_BASED:
            hits = sum(D().calculate(_val_shift(s)).drift for s in range(10))
            self.assertGreaterEqual(hits, 9, f"{D.__name__} detection {hits}/10")

    def test_bounded_false_alarm_rate(self):
        for D in ERROR_BASED:
            fp = sum(D().calculate(_err_stat(s)).drift for s in range(20))
            self.assertLessEqual(fp, 6, f"{D.__name__} false alarms {fp}/20")
        for D in VALUE_BASED:
            fp = sum(D().calculate(_val_stat(s)).drift for s in range(20))
            self.assertLessEqual(fp, 6, f"{D.__name__} false alarms {fp}/20")

    def test_metadata_streaming(self):
        for D in ALL:
            det = D()
            self.assertEqual(det.metadata.get("mode"), "streaming")
            self.assertEqual(det.metadata.get("type"), "univariate")
            self.assertIn("references", det.metadata)

    def test_result_shape(self):
        r = DDM().calculate(_err_shift(0))
        self.assertTrue(hasattr(r, "last_index") and hasattr(r, "drift"))
        self.assertIn("warning", r.details)


if __name__ == "__main__":
    unittest.main()
