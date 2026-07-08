"""Experiment 2 (LOCAL, dependency-light) -- Model Disagreement across the
four registry soft sensors RF, CART, SVM, LSTM.

This variant avoids pandas and the drift_detectors/river import chain: it
reads the per-batch CSVs with the standard-library csv module and computes
MDM inline in pure NumPy, reproducing drift_detectors.ModelDisagreementMetric
(error-kind aggregate = mean pairwise scale-normalised RMSE) exactly.

Only requirements: numpy, pyyaml, scikit-learn (RF/CART/SVM + scalers),
tensorflow (LSTM). No pandas, no river.

Run from the repo root:
    python -m use_cases.IndPenSim.experiments.run_experiment_2_local
or directly:
    python use_cases/IndPenSim/experiments/run_experiment_2_local.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from registry_models import FEATURES, load_all  # noqa: E402

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
TEST_DIR = CASE / "data" / "test"
OUT = CASE / "results" / "expB_per_phase.json"

TIME_COL = "time"
TARGET_COL = "penicillin_concentration"
PHASES = {"lag": (0, 30), "log": (30, 120), "stationary": (120, 200), "death": (200, 300)}
TEST_BATCHES = [91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
PANEL = ["RF", "CART", "SVM", "LSTM"]


# ----------------------------------------------------------------- CSV (no pandas)
def read_batch(batch: int):
    path = TEST_DIR / f"batch_{batch}.csv"
    if not path.exists():
        raise SystemExit(f"missing test batch CSV: {path}")
    cols = [TIME_COL, TARGET_COL, *FEATURES]
    data = {c: [] for c in cols}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                vals = {c: float(row[c]) for c in cols}
            except (ValueError, KeyError, TypeError):
                continue  # skip rows with blank/non-numeric cells in needed cols
            for c in cols:
                data[c].append(vals[c])
    return {c: np.asarray(v, dtype=float) for c, v in data.items()}


def phase_slice(batch_data, t0, t1):
    t = batch_data[TIME_COL]
    mask = (t >= t0) & (t < t1)
    if mask.sum() < 30:
        return np.empty((0, len(FEATURES))), np.empty((0,))
    X = np.column_stack([batch_data[f][mask] for f in FEATURES])
    y = batch_data[TARGET_COL][mask]
    return X, y


# ----------------------------------------------------------------- MDM (pure NumPy)
def _pearson(x, y):
    if x.size < 2:
        return 0.0
    xc, yc = x - x.mean(), y - y.mean()
    den = np.sqrt((xc * xc).sum() * (yc * yc).sum())
    if den <= 0 or not np.isfinite(den):
        return 0.0
    return float(np.clip((xc * yc).sum() / den, -1.0, 1.0))


def _rankdata(a):
    a = np.asarray(a, dtype=float).ravel()
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros_like(counts, dtype=float)
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    return ranks


def mdm(pred_list):
    """Reproduces ModelDisagreementMetric with the default MSE/Pearson/Spearman
    trio: returns (error_kind_score, score_by_kind, metric_means)."""
    preds = np.vstack([np.asarray(p, dtype=float).ravel() for p in pred_list])
    scale = float(np.std(preds))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    n = preds.shape[0]
    mse_v, pear_v, spear_v = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = preds[i], preds[j]
            rmse = float(np.sqrt(np.mean((a - b) ** 2)))
            mse_v.append(min(rmse / scale, 1.0))
            pear_v.append(float(np.clip(1.0 - _pearson(a, b), 0.0, 2.0)) / 2.0)
            spear_v.append(float(np.clip(1.0 - _pearson(_rankdata(a), _rankdata(b)), 0.0, 2.0)) / 2.0)
    means = {"mse": float(np.mean(mse_v)), "pearson": float(np.mean(pear_v)),
             "spearman": float(np.mean(spear_v))}
    score_by_kind = {"error": means["mse"],
                     "correlation": float(np.mean([means["pearson"], means["spearman"]]))}
    return score_by_kind["error"], score_by_kind, means


# ----------------------------------------------------------------- perf helpers
def _rmse(y, yh):
    return float(math.sqrt(np.mean((y - yh) ** 2)))


def _mae(y, yh):
    return float(np.mean(np.abs(y - yh)))


def _cv(y, yh):
    m = float(np.mean(y))
    return float("nan") if abs(m) < 1e-12 else _rmse(y, yh) / m


def main():
    print("loading the four registry soft sensors (RF, CART, SVM, LSTM) ...")
    models = load_all()

    rows = []
    for b in TEST_BATCHES:
        bd = read_batch(b)
        for ph, (t0, t1) in PHASES.items():
            X, y = phase_slice(bd, t0, t1)
            if X.size == 0:
                continue
            preds = {name: models[name].predict(X) for name in PANEL}
            perf = {name: {"CV": _cv(y, p), "MAE": _mae(y, p), "RMSE": _rmse(y, p)}
                    for name, p in preds.items()}
            score, by_kind, metric_means = mdm([preds[name] for name in PANEL])
            rows.append({
                "batch": b, "phase": ph, "n": int(len(y)), "panel": PANEL,
                "perf": perf, "MDM_score": float(score),
                "MDM_score_by_kind": by_kind, "MDM_metrics": metric_means,
            })
            print(f"  batch {b} [{ph:10s}] n={len(y):4d}  MDM={float(score):.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nwrote {OUT}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
