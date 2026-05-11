"""
Lightweight, reproducible benchmark runner for drift_detectors_pack.

Given a labelled dataset of (reference, test, expected_drift) tuples and a
catalogue of detectors, it records:

* drift decision (TP, FP, TN, FN per pair)
* drift score (when applicable)
* wall-clock runtime per detector call
* peak memory delta per detector call (best-effort, via ``tracemalloc``)

Results are returned as a list of :class:`BenchmarkResult` records and may be
exported to CSV via :meth:`BenchmarkRunner.to_csv`.
"""

from __future__ import annotations

import csv
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class BenchmarkResult:
    detector: str
    case_id: str
    expected_drift: Optional[bool]
    detected_drift: bool
    score: Optional[float]
    runtime_ms: float
    peak_memory_kb: float
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def outcome(self) -> str:
        """TP/FP/TN/FN classification when ground truth is known."""
        if self.expected_drift is None:
            return "unlabelled"
        if self.expected_drift and self.detected_drift:
            return "TP"
        if self.expected_drift and not self.detected_drift:
            return "FN"
        if (not self.expected_drift) and self.detected_drift:
            return "FP"
        return "TN"


# A detector "factory" returns a fresh detector instance on each call; this
# keeps state isolated between benchmark cases for fair comparison.
DetectorFactory = Callable[[], Any]


@dataclass
class BenchmarkCase:
    case_id: str
    reference: np.ndarray
    test: np.ndarray
    expected_drift: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    """Run a catalogue of detectors over a sequence of benchmark cases."""

    def __init__(
        self,
        detectors: Dict[str, DetectorFactory],
        cases: Sequence[BenchmarkCase],
    ) -> None:
        self._detectors = dict(detectors)
        self._cases = list(cases)
        self._results: List[BenchmarkResult] = []

    # ------------------------------------------------------------------ runtime
    def run(self) -> List[BenchmarkResult]:
        self._results = []
        for case in self._cases:
            for det_name, factory in self._detectors.items():
                self._results.append(self._run_one(det_name, factory, case))
        return list(self._results)

    @property
    def results(self) -> List[BenchmarkResult]:
        return list(self._results)

    # ------------------------------------------------------------------- export
    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "detector", "case_id", "expected_drift", "detected_drift",
                "outcome", "score", "runtime_ms", "peak_memory_kb",
            ])
            for r in self._results:
                writer.writerow([
                    r.detector, r.case_id,
                    "" if r.expected_drift is None else int(r.expected_drift),
                    int(r.detected_drift),
                    r.outcome,
                    "" if r.score is None else f"{r.score:.6g}",
                    f"{r.runtime_ms:.3f}",
                    f"{r.peak_memory_kb:.1f}",
                ])
        return path

    def summary_by_detector(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate counts and timing per detector."""
        out: Dict[str, Dict[str, Any]] = {}
        for r in self._results:
            d = out.setdefault(r.detector, {
                "n": 0, "TP": 0, "FP": 0, "TN": 0, "FN": 0,
                "runtime_ms_sum": 0.0, "peak_memory_kb_max": 0.0,
            })
            d["n"] += 1
            d[r.outcome] = d.get(r.outcome, 0) + 1
            d["runtime_ms_sum"] += r.runtime_ms
            d["peak_memory_kb_max"] = max(d["peak_memory_kb_max"], r.peak_memory_kb)

        # Derived metrics — guarded against zero-divisions.
        for d in out.values():
            tp, fp, tn, fn = d["TP"], d["FP"], d.get("TN", 0), d.get("FN", 0)
            d["precision"] = tp / max(tp + fp, 1)
            d["recall"] = tp / max(tp + fn, 1)
            d["specificity"] = tn / max(tn + fp, 1)
            d["accuracy"] = (tp + tn) / max(tp + tn + fp + fn, 1)
            d["mean_runtime_ms"] = d["runtime_ms_sum"] / max(d["n"], 1)
        return out

    # ----------------------------------------------------------------- internal
    @staticmethod
    def _run_one(
        det_name: str,
        factory: DetectorFactory,
        case: BenchmarkCase,
    ) -> BenchmarkResult:
        detector = factory()
        tracemalloc.start()
        t0 = time.perf_counter()
        try:
            result = detector.calculate(case.test, reference_data=case.reference)
        except TypeError:
            # Streaming detectors (ADWIN, Page-Hinkley, HDDM-A, EDDM)
            # do not accept ``reference_data``; feed the test stream directly.
            result = detector.calculate(case.test)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Result objects vary across detectors; surface a unified view.
        score = getattr(result, "score", None)
        drift = bool(getattr(result, "drift", False))
        details = dict(getattr(result, "details", {}) or {})

        return BenchmarkResult(
            detector=det_name,
            case_id=case.case_id,
            expected_drift=case.expected_drift,
            detected_drift=drift,
            score=None if score is None else float(score),
            runtime_ms=elapsed_ms,
            peak_memory_kb=peak / 1024.0,
            details=details,
        )


def discover_default_detectors() -> Dict[str, DetectorFactory]:
    """Return the standard catalogue of detector factories shipped with the package."""
    from drift_detectors import (
        Adwin,
        EDDM,
        HDDM_A,
        KDQTree,
        KSDetector,
        MMDDetector,
        PCA_CD,
        PSI,
        PageHinkley,
    )

    return {
        "PSI": lambda: PSI(),
        "KS": lambda: KSDetector(),
        "ADWIN": lambda: Adwin(),
        "PageHinkley": lambda: PageHinkley(),
        "HDDM-A": lambda: HDDM_A(),
        "EDDM": lambda: EDDM(),
        "MMD": lambda: MMDDetector(),
        "PCA-CD": lambda: PCA_CD(),
        "KDQ-Tree": lambda: KDQTree(),
    }


def cases_from_pairs(
    pairs: Iterable[Tuple[str, np.ndarray, np.ndarray, Optional[bool]]],
) -> List[BenchmarkCase]:
    """Convenience constructor: build ``BenchmarkCase`` objects from raw tuples."""
    return [
        BenchmarkCase(case_id=cid, reference=ref, test=tst, expected_drift=lbl)
        for cid, ref, tst, lbl in pairs
    ]
