"""Experiment 2 of the companion SoftwareX paper.

Model Disagreement Mechanism (MDM) across a structurally diverse panel of
four pre-trained soft sensors served by the STAMM model registry:

    RF    Random Forest          Acosta-Pavas et al. (2024)   interpretable
    CART  Decision Tree          Acosta-Pavas et al. (2024)   interpretable
    SVM   Support Vector Reg.    STAMM (2026)                 black-box
    LSTM  LSTM network           STAMM (2026)                 black-box

The two interpretable + two black-box mix is deliberate: disagreement
across learner families is a stronger degradation signal than disagreement
within one family (which shares inductive bias). The models are loaded
as-is from the registry (no retraining) and evaluated per growth phase on
the ten test batches 91-100, using the per-batch CSVs in data/test/.

Produces use_cases/IndPenSim/results/expB_per_phase.json.

Requirements (use-case only): scikit-learn>=1.5, tensorflow>=2.18, pyyaml.

Run with:
    python -m use_cases.IndPenSim.experiments.run_experiment_2
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
REPO = CASE.parent.parent
sys.path.insert(0, str(REPO))

from drift_detectors import ModelDisagreementMetric  # noqa: E402
from use_cases.IndPenSim.experiments.registry_models import FEATURES, load_all  # noqa: E402

TEST_DIR = CASE / "data" / "test"
OUT = CASE / "results" / "expB_per_phase.json"

TIME_COL = "time"
TARGET_COL = "penicillin_concentration"

PHASES = {
    "lag":        (0,   30),
    "log":        (30,  120),
    "stationary": (120, 200),
    "death":      (200, 300),
}
TEST_BATCHES = [91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
PANEL = ["RF", "CART", "SVM", "LSTM"]


def _rmse(y, yh):
    return float(math.sqrt(np.mean((y - yh) ** 2)))


def _mae(y, yh):
    return float(np.mean(np.abs(y - yh)))


def _cv(y, yh):
    m = float(np.mean(y))
    return float("nan") if abs(m) < 1e-12 else _rmse(y, yh) / m


def load_batch(batch: int) -> pd.DataFrame:
    path = TEST_DIR / f"batch_{batch}.csv"
    if not path.exists():
        raise SystemExit(f"missing test batch CSV: {path}")
    df = pd.read_csv(path)
    keep = [TIME_COL, TARGET_COL, *FEATURES]
    return df[keep].apply(pd.to_numeric, errors="coerce")


def phase_slice(df: pd.DataFrame, t0: float, t1: float):
    sub = df[(df[TIME_COL] >= t0) & (df[TIME_COL] < t1)].dropna(how="any")
    if len(sub) < 30:
        return np.empty((0, len(FEATURES))), np.empty((0,))
    return sub[FEATURES].to_numpy(dtype=float), sub[TARGET_COL].to_numpy(dtype=float)


def main():
    print("loading the four registry soft sensors (RF, CART, SVM, LSTM) ...")
    models = load_all()

    rows = []
    for b in TEST_BATCHES:
        df = load_batch(b)
        for ph, (t0, t1) in PHASES.items():
            X, y = phase_slice(df, t0, t1)
            if X.size == 0:
                continue
            preds = {name: models[name].predict(X) for name in PANEL}
            perf = {name: {"CV": _cv(y, p), "MAE": _mae(y, p), "RMSE": _rmse(y, p)}
                    for name, p in preds.items()}
            mdm = ModelDisagreementMetric(threshold=0.10).calculate(
                predictions=[preds[name] for name in PANEL]
            )
            rows.append({
                "batch": b,
                "phase": ph,
                "n": int(len(y)),
                "panel": PANEL,
                "perf": perf,
                "MDM_score": float(mdm.score),
                "MDM_score_by_kind": mdm.details["score_by_kind"],
                "MDM_metrics": mdm.details["metric_means"],
            })
            print(f"  batch {b} [{ph:10s}] n={len(y):4d}  MDM={float(mdm.score):.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nwrote {OUT}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
