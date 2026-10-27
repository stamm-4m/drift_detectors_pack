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

from collections.abc import Sequence

import numpy as np

from drift_detectors.drift_detector import DriftDetector
from drift_detectors.model_based.disagreement_metrics import (
    DisagreementMetric,
    MSEDisagreement,
    PearsonDisagreement,
    SpearmanDisagreement,
)
from drift_detectors.utility.drift_detection_output import ScoreDriftResult


def _check_scale(scale):
    if isinstance(scale, str):
        if scale not in ("window", "reference"):
            raise ValueError(f"scale must be 'window', 'reference' or a positive number, not {scale!r}")
        return scale
    value = float(scale)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"scale must be a positive number, not {scale!r}")
    return value


def _nanmean(values) -> float | None:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def _default_metrics() -> list[DisagreementMetric]:
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
    scale : "window" | "reference" | float, default="window"
        Scale of the metrics that use one (MSE: RMSE / scale and DTW, both
        capped at 1).
        ``"window"``: standard deviation of the predictions given to
        ``calculate`` (behaviour up to 0.5.1; saturates on short windows or
        flat signals). ``"reference"``: standard deviation of
        ``reference_data`` (one array of past predictions per model, e.g. the
        run so far). A positive number: that fixed scale (e.g. the expected
        range of the predicted variable).
    decision : str, default="error"
        Which aggregate gives ``score`` and ``drift``: a metric family
        (``"error"``, ``"correlation"``, ``"temporal"``) or ``"any"`` (the
        largest defined family aggregate). With ``"error"`` and no
        error-family metric, the largest family is used (as up to 0.5.1).
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
        metrics: Sequence[DisagreementMetric] | None = None,
        threshold: float = 0.25,
        online: bool = False,
        # legacy parameter kept for backwards compatibility:
        models: Sequence | None = None,
        *,
        scale: str | float = "window",
        decision: str = "error",
    ) -> None:
        super().__init__(reference_data=None, online=online)
        self._metrics: list[DisagreementMetric] = list(metrics) if metrics else _default_metrics()
        self._threshold = float(threshold)
        self._scale = _check_scale(scale)
        self._decision = str(decision)
        # ``models`` is not used by the new predictions API; we keep the
        # parameter on the constructor signature so that pre-v0.4.0 user
        # code continues to import without crashing. Predictions are now
        # passed to ``calculate``.
        self._legacy_models = list(models) if models is not None else None

    # ----------------------------------------------------------------- API
    def add_metric(self, metric: DisagreementMetric) -> None:
        self._metrics.append(metric)

    @property
    def metric_names(self) -> list[str]:
        return [m.name for m in self._metrics]

    def calculate(
        self,
        test_data: np.ndarray | None = None,
        reference_data: np.ndarray | None = None,
        predictions: Sequence[np.ndarray] | None = None,
        threshold: float | None = None,
    ) -> ScoreDriftResult:
        """Compute pair-wise model disagreement.

        Parameters
        ----------
        predictions : sequence of np.ndarray, required
            One 1-D array per co-deployed model, all of equal length.
            These are the per-batch simulations exported by the model
            registry; MDM does not need access to the model objects
            themselves.
        reference_data : sequence of np.ndarray, optional
            Only with ``scale="reference"``: one array of past predictions
            per model (any length); their standard deviation is the scale.
        test_data
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

        # Scale of the metrics that use one (MSE, DTW; 0.5.2: configurable).
        if self._scale == "window":
            scale = float(np.std(preds))
            scale_source = "window"
        elif self._scale == "reference":
            if reference_data is None:
                raise ValueError("scale='reference' needs reference_data: one array of past "
                                 "predictions per model")
            ref = np.concatenate([np.asarray(r, dtype=float).ravel() for r in reference_data])
            ref = ref[np.isfinite(ref)]
            if ref.size < 2:
                raise ValueError("scale='reference': fewer than 2 reference predictions")
            scale = float(np.std(ref))
            scale_source = "reference"
        else:
            scale = float(self._scale)
            scale_source = "fixed"
        scale_used = scale
        if not np.isfinite(scale) or scale <= 1e-12:
            scale_used = 1.0
            scale_source += " (spread 0: 1.0 used)"

        # Per-pair, per-metric matrix; NaN = not defined for that pair
        # (e.g. a correlation with a constant series, 0.5.2)
        pair_matrices: dict[str, np.ndarray] = {
            m.name: np.zeros((n_models, n_models), dtype=float) for m in self._metrics
        }
        per_metric_values: dict[str, list[float]] = {m.name: [] for m in self._metrics}
        for i in range(n_models):
            for j in range(i + 1, n_models):
                a, b = preds[i], preds[j]
                for m in self._metrics:
                    v = float(m.pair(a, b, scale=scale_used))
                    per_metric_values[m.name].append(v)
                    pair_matrices[m.name][i, j] = pair_matrices[m.name][j, i] = v

        # Per-metric means over the pairs where the metric is defined
        per_metric_means: dict[str, float | None] = {
            n: _nanmean(v) for n, v in per_metric_values.items()
        }
        undefined_pairs = {n: int(sum(1 for x in v if not np.isfinite(x)))
                           for n, v in per_metric_values.items()}

        # Aggregate BY KIND: mean over the defined metrics of the same family,
        # never across families. ``score_by_kind`` is the user-facing summary.
        kind_buckets: dict[str, list] = {}
        for m in self._metrics:
            kind_buckets.setdefault(m.kind, []).append(per_metric_means[m.name])
        score_by_kind: dict[str, float | None] = {k: _nanmean(vs) for k, vs in kind_buckets.items()}
        defined = {k: v for k, v in score_by_kind.items() if v is not None}

        # Headline ``score``: the family chosen by ``decision`` (default the
        # error family: "are the soft sensors producing the same number?").
        decision = self._decision
        if decision == "any":
            primary = max(defined.values()) if defined else None
        elif decision == "error" and "error" not in score_by_kind:
            # No error-family metric configured: the largest family (as up to 0.5.1)
            primary = max(defined.values()) if defined else None
        elif decision in score_by_kind:
            primary = score_by_kind[decision]
        else:
            raise ValueError(f"decision {decision!r}: no metric of that family "
                             f"(families: {sorted(score_by_kind)}, or 'any')")
        details = {
            "threshold": thr,
            "decision": decision,
            "scale": self._scale,
            "scale_source": scale_source,
            "scale_value": scale,
            "n_models": n_models,
            "n_samples": n_samples,
            "metrics": list(per_metric_means.keys()),
            "metric_means": per_metric_means,
            "undefined_pairs": undefined_pairs,
            "score_by_kind": score_by_kind,
            "pairwise": {n: pair_matrices[n].tolist() for n in pair_matrices},
            "mode": "online" if self.is_online() else "offline",
        }
        if primary is None:
            details["status"] = "decision_undefined"
            return ScoreDriftResult(score=None, drift=False, details=details)
        return ScoreDriftResult(score=float(primary), drift=float(primary) >= thr, details=details)
