"""Unit tests for ModelDisagreementMetric (v0.4.0 predictions API + pluggable metrics)."""
import numpy as np
import unittest

from drift_detectors.model_based.model_disagreement.detector import (
    ModelDisagreementMetric,
)
from drift_detectors.model_based.disagreement_metrics import (
    MSEDisagreement, PearsonDisagreement, SpearmanDisagreement, DisagreementMetric,
)
from drift_detectors.utility.drift_detection_output import ScoreDriftResult


class TestModelDisagreementMetric(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(0)
        self.y_true = np.linspace(0, 10, 200)

    def _y(self, noise_std: float, scale: float = 1.0, bias: float = 0.0) -> np.ndarray:
        return scale * self.y_true + bias + noise_std * self.rng.normal(size=self.y_true.size)

    def test_zero_disagreement_when_predictions_identical(self) -> None:
        y = self._y(0.1)
        det = ModelDisagreementMetric()
        res = det.calculate(predictions=[y, y.copy(), y.copy()])
        self.assertIsInstance(res, ScoreDriftResult)
        self.assertAlmostEqual(res.score, 0.0, places=6)
        self.assertFalse(res.drift)

    def test_disagreement_grows_with_perturbation(self) -> None:
        y0 = self._y(0.1)
        small = ModelDisagreementMetric().calculate(
            predictions=[y0, self._y(0.5)]
        ).score
        large = ModelDisagreementMetric().calculate(
            predictions=[y0, self._y(2.0, scale=1.5, bias=2.0)]
        ).score
        self.assertGreater(large, small)

    def test_drift_flag_at_threshold(self) -> None:
        det = ModelDisagreementMetric(threshold=0.05)
        res = det.calculate(
            predictions=[self._y(0.1), self._y(2.0, scale=1.5, bias=2.0)]
        )
        self.assertTrue(res.drift)
        self.assertGreaterEqual(res.score, 0.05)

    def test_default_metric_set(self) -> None:
        det = ModelDisagreementMetric()
        self.assertEqual(det.metric_names, ["mse", "pearson", "spearman"])

    def test_metric_pluggability(self) -> None:
        det = ModelDisagreementMetric(metrics=[MSEDisagreement(), PearsonDisagreement()])
        self.assertEqual(det.metric_names, ["mse", "pearson"])
        res = det.calculate(predictions=[self._y(0.1), self._y(0.5)])
        self.assertIn("mse", res.details["metric_means"])
        self.assertIn("pearson", res.details["metric_means"])
        self.assertNotIn("spearman", res.details["metric_means"])

    def test_user_defined_metric(self) -> None:
        class MeanAbsDiff(DisagreementMetric):
            name = "mean_abs"
            kind = "error"
            def pair(self, a, b, *, scale=1.0):
                return float(min(np.mean(np.abs(a - b)) / max(scale, 1e-12), 1.0))
        det = ModelDisagreementMetric(metrics=[MeanAbsDiff()])
        res = det.calculate(predictions=[self._y(0.1), self._y(0.5)])
        self.assertIn("mean_abs", res.details["metric_means"])

    def test_score_by_kind(self) -> None:
        """The detector should aggregate metrics by kind (error / correlation)
        and never average across families.
        """
        det = ModelDisagreementMetric()
        res = det.calculate(predictions=[self._y(0.1), self._y(1.0, scale=1.2, bias=1.0)])
        self.assertIn("score_by_kind", res.details)
        self.assertIn("error",       res.details["score_by_kind"])
        self.assertIn("correlation", res.details["score_by_kind"])
        # The headline `score` is the error-family aggregate.
        self.assertAlmostEqual(res.score, res.details["score_by_kind"]["error"], places=8)

    def test_kind_attribute_on_default_metrics(self) -> None:
        det = ModelDisagreementMetric()
        kinds = {m.name: m.kind for m in det._metrics}
        self.assertEqual(kinds, {"mse": "error", "pearson": "correlation", "spearman": "correlation"})

    def test_pairwise_matrix_shape(self) -> None:
        preds = [self._y(s) for s in (0.1, 0.5, 1.0, 0.3)]
        res = ModelDisagreementMetric().calculate(predictions=preds)
        for m, mat in res.details["pairwise"].items():
            arr = np.asarray(mat)
            self.assertEqual(arr.shape, (4, 4))
            self.assertTrue(np.allclose(arr, arr.T))

    def test_requires_two_predictions(self) -> None:
        with self.assertRaises(ValueError):
            ModelDisagreementMetric().calculate(predictions=[self._y(0.1)])

    def test_requires_predictions_kwarg(self) -> None:
        with self.assertRaises(ValueError):
            ModelDisagreementMetric().calculate(test_data=np.zeros(10))

    def test_metadata_loaded(self) -> None:
        det = ModelDisagreementMetric()
        self.assertEqual(det.metadata.get("acronym"), "MDM")


class TestDisagreementMetricsIndividually(unittest.TestCase):
    def setUp(self) -> None:
        self.a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.c = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

    def test_mse_zero_on_equal(self):
        self.assertAlmostEqual(MSEDisagreement().pair(self.a, self.b, scale=2.0), 0.0)

    def test_mse_clamps_to_one(self):
        big = self.a + 1e6
        self.assertAlmostEqual(MSEDisagreement().pair(self.a, big, scale=1.0), 1.0)

    def test_pearson_anticorrelation_is_one(self):
        self.assertAlmostEqual(PearsonDisagreement().pair(self.a, self.c), 1.0, places=4)

    def test_spearman_anticorrelation_is_one(self):
        self.assertAlmostEqual(SpearmanDisagreement().pair(self.a, self.c), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
