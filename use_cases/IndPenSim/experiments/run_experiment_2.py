"""Experiment 2 of the companion SoftwareX paper.

Model Disagreement Metric (MDM) across four interpretable soft sensors
(CART, M5, CUBIST, Random Forest) trained on the LSTM training set
(batches 1-60), evaluated on the ten test batches (91-100) per phase.

Produces use_cases/IndPenSim/results/expB_per_phase.json.

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

from drift_detectors import ModelDisagreementMetric
from use_cases.IndPenSim.experiments.soft_sensors import CART, M5, CUBIST, RandomForest

CSV  = CASE / "data" / "100_Batches_IndPenSim_V3.1.csv"
OUT  = CASE / "results" / "expB_per_phase.json"

PREDICTOR_LBL = ["Fs", "RPM", "T", "pH", "DO2", "V", "CO2_off", "O2_off"]
PREDICTOR_COLS = {
    "Fs":      "Sugar feed rate(Fs:L/h)",
    "RPM":     "Agitator RPM(RPM:RPM)",
    "T":       "Temperature(T:K)",
    "pH":      "pH(pH:pH)",
    "DO2":     "Dissolved oxygen concentration(DO2:mg/L)",
    "V":       "Vessel Volume(V:L)",
    "CO2_off": "carbon dioxide percent in off-gas(CO2outgas:%)",
    "O2_off":  "Oxygen in percent in off-gas(O2:O2  (%))",
}
TARGET_COL = "Penicillin concentration(P:g/L)"
TIME_COL = "Time (h)"
BATCH_COL = "Batch ID"

PHASES = {
    "lag":        (0,   30),
    "log":        (30,  120),
    "stationary": (120, 200),
    "death":      (200, 300),
}
REF_BATCHES  = list(range(1, 61))
TEST_BATCHES = [91, 92, 93, 94, 95, 96, 97, 98, 99, 100]


def load():
    if not CSV.exists():
        raise SystemExit(
            f"IndPenSim CSV not found at {CSV}.\n"
            "Run use_cases/IndPenSim/data/download_indpensim.py first.")
    cols = list(PREDICTOR_COLS.values()) + [TARGET_COL, TIME_COL, BATCH_COL]
    df = pd.read_csv(CSV, usecols=cols)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _r2(y, yh):
    ss_res = float(np.sum((y - yh) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if ss_tot < 1e-12 else 1 - ss_res / ss_tot


def _mae(y, yh):
    return float(np.mean(np.abs(y - yh)))


def _rmse(y, yh):
    return float(math.sqrt(np.mean((y - yh) ** 2)))


def _cv(y, yh):
    m = float(np.mean(y))
    return float("nan") if abs(m) < 1e-12 else _rmse(y, yh) / m


def phase_data(df, batch, t0, t1):
    sub = df[(df[BATCH_COL] == batch) & (df[TIME_COL] >= t0) & (df[TIME_COL] < t1)]
    sub = sub[[PREDICTOR_COLS[k] for k in PREDICTOR_LBL] + [TARGET_COL]].dropna(how="any")
    if len(sub) < 30:
        return np.empty((0,)), np.empty((0,))
    return (sub[[PREDICTOR_COLS[k] for k in PREDICTOR_LBL]].to_numpy(dtype=float),
            sub[TARGET_COL].to_numpy(dtype=float))


def main():
    df = load()
    print(f"loaded {CSV.name}: {len(df):,} rows")

    # ---- train on reference batches 1-60 (across all phases pooled) ----
    Xs, ys = [], []
    for b in REF_BATCHES:
        for ph, (t0, t1) in PHASES.items():
            X, y = phase_data(df, b, t0, t1)
            if X.size:
                Xs.append(X)
                ys.append(y)
    X_train = np.vstack(Xs)
    y_train = np.concatenate(ys)
    rng = np.random.default_rng(0)
    if len(X_train) > 5000:
        idx = rng.choice(len(X_train), size=5000, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
    print(f"training rows: {len(X_train):,}")

    print("   training CART ...")
    M_cart = CART(max_depth=8, min_samples_leaf=20).fit(X_train, y_train)
    print("   training M5 ...")
    M_m5 = M5(max_depth=4, min_samples_leaf=100).fit(X_train, y_train)
    print("   training CUBIST ...")
    M_cubist = CUBIST(max_depth=4, min_samples_leaf=100).fit(X_train, y_train)
    print("   training RF ...")
    M_rf = RandomForest(n_trees=15, max_depth=8, min_samples_leaf=20).fit(X_train, y_train)
    models = {"CART": M_cart, "M5": M_m5, "CUBIST": M_cubist, "RF": M_rf}

    rows = []
    for b in TEST_BATCHES:
        for ph, (t0, t1) in PHASES.items():
            X, y = phase_data(df, b, t0, t1)
            if X.size == 0:
                continue
            preds = {n: m.predict(X) for n, m in models.items()}
            perf = {n: {"CV": _cv(y, p), "MAE": _mae(y, p), "RMSE": _rmse(y, p)}
                    for n, p in preds.items()}
            mdm = ModelDisagreementMetric(threshold=0.10).calculate(
                predictions=[preds[n] for n in preds]
            )
            row = {
                "batch": b,
                "phase": ph,
                "n": int(len(y)),
                "perf": perf,
                "MDM_score": float(mdm.score),
                "MDM_score_by_kind": mdm.details["score_by_kind"],
                "MDM_metrics": mdm.details["metric_means"],
            }
            rows.append(row)
    OUT.write_text(json.dumps(rows, indent=2, default=str))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
