"""ModelDisagreementMetric (MDM) -- the model-based drift module.

MDM operationalises the model-divergence panel of the STAMM platform
(Suarez et al., 2026). When ground-truth labels are delayed, the
pair-wise disagreement between several co-deployed models on the live
inputs is the only continuously-available proxy for performance
degradation, and MDM aggregates that signal into a drift score.

The module is *pluggable*. The set of pair-wise statistics is not
hard-coded: instead, MDM accepts a list of ``DisagreementMetric``
instances (see ``drift_detectors.model_based.disagreement_metrics``).
Users can turn metrics on or off, register new ones (subclass
``DisagreementMetric`` and pass an instance), or use the default trio
(MSE, Pearson, Spearman).

API note. As of v0.4.0, MDM consumes *predictions* rather than callables.
This matches the STAMM model registry, which exports per-batch
simulations rather than the model objects themselves: it removes a
heavy coupling between the monitoring layer and the model registry.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from drift_detectors.drift_detector import DriftDetector
from drift_detectors.utility.drift_detection_output import ScoreDriftResult
from drift_detectors.model_based.disagreement_metrics import (
    DisagreementMetric, MSEDisagreement, PearsonDisagreement, SpearmanDisagreement,
)


def _default_metrics() -> List[DisagreementMetric]:
    """Default trio: magnitude (MSE), linear correlation, rank correlation."""
    return [MSEDisagreement(), PearsonDisagreement(), SpearmanDisagreement()]


class ModelDisagreementMetric(DriftDetector):
    """Pair-wise model disagreement detector with pluggable metrics.

    Parameters
    ----------
    metrics : sequence of :class:`DisagreementMetric`, optional
        Which pair-wise statistics to aggregate. Defaults to MSE +
        Pearson + Spearman. Supply an empty list to disable a metric, or
        register your own by subclassing ``DisagreementMetric``.
    threshold : float, default=0.25
        Aggregate disagreement above which a drift flag is raised.
    online : bool, default=False
        Reserved for streaming use; currently behaves as ``False``.

    Notes
    -----
    The ``calculate`` method takes a sequence of *prediction arrays*,
    not callables. Each prediction array is the live output of one
    co-deployed model on the same set of inputs; the model objects
    themselves do not need to be shared with the detector. This is the
    pattern used by the STAMM model registry, where each registered
    model exports its per-batch simulation as a 1-D array.
    """

    def __init__(
        self,
        metrics: Optional[Sequence[DisagreementMetric]] = None,
        threshold: float = 0.25,
        online: bool = False,
        # legacy parameter kept for backwards compatibility:
        models: Optional[Sequence] = None,
    ) -> None:
        super().__init__(reference_data=None, online=online)
        self._metrics: List[DisagreementMetric] = list(metrics) if metrics else _default_metrics()
        self._threshold = float(threshold)
        # ``models`` is not used by the new predictions API; we keep the
        # parameter on the constructor signature so that pre-v0.4.0 user
        # code continues to import without crashing. Predictions are now
        # passed to ``calculate``.
        self._legacy_models = list(models) if models is not None else None

    # ----------------------------------------------------------------- API
    def add_metric(self, metric: DisagreementMetric) -> None:
        self._metrics.append(metric)

    @property
    def metric_names(self) -> List[str]:
        return [m.name for m in self._metrics]

    def calculate(
        self,
        test_data: Optional[np.ndarray] = None,
        reference_data: Optional[np.ndarray] = None,
        predictions: Optional[Sequence[np.ndarray]] = None,
        threshold: Optional[float] = None,
    ) -> ScoreDriftResult:
        """Compute pair-wise model disagreement.

        Parameters
        ----------
        predictions : sequence of np.ndarray, required
            One 1-D array per co-deployed model, all of equal length.
            These are the per-batch simulations exported by the model
            registry; MDM does not need access to the model objects
            themselves.
        test_data, reference_data
            Unused by this detector; kept on the signature so that the
            shipped ``BenchmarkRunner`` and dashboards can call every
            detector with the same ``calculate(test_data, reference_data)``
            signature without special-casing the model-based family.
        threshold : float, optional
            Override the construction-time threshold for this call.
        """
        if predictions is None:
            raise ValueError(
                "ModelDisagreementMetric.calculate() requires `predictions=[...]` "
                "(one 1-D array per co-deployed model, all of equal length)."
            )
        preds = np.vstack([np.asarray(p, dtype=float).ravel() for p in predictions])
        n_models, n_samples = preds.shape
        if n_models < 2:
            raise ValueError("At least two models are required to compute disagreement.")
        if n_samples < 2:
            return ScoreDriftResult(
                score=0.0, drift=False,
                details={"status": "not_enough_samples", "n_models": n_models},
            )

        thr = self._threshold if threshold is None else float(threshold)

        scale = float(np.std(preds))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0

        # Per-pair, per-metric matrix
        pair_matrices: Dict[str, np.ndarray] = {
            m.name: np.zeros((n_models, n_models), dtype=float) for m in self._metrics
        }
        per_metric_values: Dict[str, List[float]] = {m.name: [] for m in self._metrics}
        for i in range(n_models):
            for j in range(i + 1, n_models):
                a, b = preds[i], preds[j]
                for m in self._metrics:
                    v = float(m.pair(a, b, scale=scale))
                    per_metric_values[m.name].append(v)
                    pair_matrices[m.name][i, j] = pair_matrices[m.name][j, i] = v

        # Aggregate: mean over metrics of mean over pairs.
        per_metric_means: Dict[str, float] = {
            n: float(np.mean(v)) if v else 0.0 for n, v in per_metric_values.items()
        }
        aggregate = float(np.mean(list(per_metric_means.values()))) if per_metric_means else 0.0

        return ScoreDriftResult(
            score=aggregate,
            drift=aggregate >= thr,
            details={
                "threshold": thr,
                "n_models": n_models,
                "n_samples": n_samples,
                "metrics": list(per_metric_means.keys()),
                "metric_means": per_metric_means,
                "pairwise": {n: pair_matrices[n].tolist() for n in pair_matrices},
                "mode": "online" if self.is_online() else "offline",
            },
        )
