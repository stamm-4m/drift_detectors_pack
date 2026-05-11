"""Soft-sensor experiment for IndPenSim batches 91--100, mirroring the
protocol of Metcalfe et al. (2025) -- Computers and Chemical Engineering 194,
108991. Same predictors, same batch protocol, with a Ridge-regression soft
sensor used as a TensorFlow-free stand-in for the original LSTM."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
REF_DIR = REPO_ROOT / "use_cases" / "indepensim_data" / "reference_data"
TEST_DIR = REPO_ROOT / "use_cases" / "indepensim_data" / "test_data"

PREDICTOR_COLUMNS = [
    "temperature", "pH",
    "dissolved_oxygen_concentration", "sugar_feed_rate",
    "substrate_concentration", "oil_flow",
    "oxygen_in_percent_in_off_gas", "CO2_percent_in_off_gas",
    "ammonia_shots",
]
LEAN_PREDICTORS = [
    "temperature", "pH",
    "dissolved_oxygen_concentration", "substrate_concentration",
]
TARGET_COLUMN = "penicillin_concentration"
INPUT_WINDOW = 4


def _load_batch(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "experiment_ID" in df.columns:
        df = df.drop(columns=["experiment_ID"])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _build_supervised(df: pd.DataFrame, columns) -> Tuple[np.ndarray, np.ndarray]:
    cols = [c for c in columns if c in df.columns]
    sub = df[cols + [TARGET_COLUMN]].dropna(how="any").reset_index(drop=True)
    if len(sub) <= INPUT_WINDOW:
        return np.empty((0,)), np.empty((0,))
    P = len(cols)
    X_full = sub[cols].to_numpy(dtype=float)
    y_full = sub[TARGET_COLUMN].to_numpy(dtype=float)
    n = len(sub) - INPUT_WINDOW
    X = np.empty((n, INPUT_WINDOW * P), dtype=float)
    for t in range(n):
        X[t] = X_full[t : t + INPUT_WINDOW].ravel()
    y = y_full[INPUT_WINDOW:]
    return X, y


class RidgeSoftSensor:
    def __init__(self, alpha: float = 1.0, standardise: bool = True) -> None:
        self.alpha = float(alpha)
        self.standardise = bool(standardise)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeSoftSensor":
        if self.standardise:
            self._mu = X.mean(axis=0)
            sd = X.std(axis=0)
            self._sd = np.where(sd > 1e-12, sd, 1.0)
            Xs = (X - self._mu) / self._sd
        else:
            self._mu = np.zeros(X.shape[1])
            self._sd = np.ones(X.shape[1])
            Xs = X
        n, p = Xs.shape
        Xa = np.hstack([Xs, np.ones((n, 1))])
        A = Xa.T @ Xa
        I = np.eye(p + 1) * self.alpha
        I[-1, -1] = 0.0
        self._w = np.linalg.solve(A + I, Xa.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self._mu) / self._sd if self.standardise else X
        Xa = np.hstack([Xs, np.ones((Xs.shape[0], 1))])
        return Xa @ self._w


def _r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot

def _mae(y_true, y_pred):  return float(np.mean(np.abs(y_true - y_pred)))
def _mse(y_true, y_pred):  return float(np.mean((y_true - y_pred) ** 2))
def _rmse(y_true, y_pred): return float(math.sqrt(_mse(y_true, y_pred)))


def run_experiment() -> Dict:
    refs  = {p.stem: _load_batch(p) for p in sorted(REF_DIR.glob("batch_*.csv"))}
    tests = {p.stem: _load_batch(p) for p in sorted(TEST_DIR.glob("batch_*.csv"))}

    def _stack_train(columns):
        Xs, ys = [], []
        for df in refs.values():
            X, y = _build_supervised(df, columns)
            if X.size:
                Xs.append(X); ys.append(y)
        return np.vstack(Xs), np.concatenate(ys)

    X_train_A, y_train = _stack_train(PREDICTOR_COLUMNS)
    X_train_B, _       = _stack_train(LEAN_PREDICTORS)
    soft_sensor_A = RidgeSoftSensor(alpha=1.0, standardise=True).fit(X_train_A, y_train)
    soft_sensor_B = RidgeSoftSensor(alpha=1.0, standardise=True).fit(X_train_B, y_train)

    import sys
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import sandbox_shim  # noqa: F401
    except Exception:
        pass
    from drift_detectors import (PSI, KSDetector, EDDM, ModelDisagreementMetric)

    pred_ref = {}
    for ref_id, df in refs.items():
        X, _ = _build_supervised(df, PREDICTOR_COLUMNS)
        if X.size:
            pred_ref[ref_id] = soft_sensor_A.predict(X)
    X_ref_full, _ = next(((X, y) for X, y in
                         [(_build_supervised(df, PREDICTOR_COLUMNS)) for df in refs.values()]
                         if len(X) > 0))
    ref_T_uni = X_ref_full[:, 0]
    ref_pen   = np.concatenate(list(pred_ref.values()))

    per_batch = {}
    train_resid = y_train - soft_sensor_A.predict(X_train_A)
    err_threshold = float(np.std(train_resid) * 2.0)

    for tid in sorted(tests.keys()):
        X_test_A, y_test = _build_supervised(tests[tid], PREDICTOR_COLUMNS)
        X_test_B, _      = _build_supervised(tests[tid], LEAN_PREDICTORS)
        if X_test_A.size == 0:
            continue
        yhat_A = soft_sensor_A.predict(X_test_A)
        yhat_B = soft_sensor_B.predict(X_test_B)

        psi_T    = PSI().calculate(X_test_A[:, 0], ref_T_uni).score
        ks_T     = KSDetector().calculate(X_test_A[:, 0], ref_T_uni).details["p_value"]
        psi_pred = PSI().calculate(yhat_A, ref_pen).score

        residuals = y_test - yhat_A
        err_stream = (np.abs(residuals) > err_threshold).astype(int)
        eddm = EDDM(min_n_errors=10).calculate(err_stream)

        # MDM operates on the (yhat_A, yhat_B) pair via pass-through closures.
        common_idx = np.arange(len(yhat_A))[:, None].astype(float)
        mdm = ModelDisagreementMetric(
            models=[lambda I, p=yhat_A: p, lambda I, p=yhat_B: p],
            threshold=0.10,
        ).calculate(common_idx)

        per_batch[tid] = {
            "n_samples": int(len(y_test)),
            "R2_A":  _r2(y_test, yhat_A),
            "MAE_A": _mae(y_test, yhat_A),
            "MSE_A": _mse(y_test, yhat_A),
            "RMSE_A": _rmse(y_test, yhat_A),
            "R2_B":  _r2(y_test, yhat_B),
            "PSI_input_T":         float(psi_T),
            "KS_input_T_pvalue":   float(ks_T),
            "PSI_predicted_Y":     float(psi_pred),
            "EDDM_residual_drift": bool(eddm.drift),
            "EDDM_residual_n_err": int(eddm.details.get("n_errors", 0)),
            "MDM_score":           float(mdm.score),
            "MDM_drift":           bool(mdm.drift),
            "MDM_metrics":         mdm.details["metric_means"],
        }
    return per_batch


def main():
    out = run_experiment()
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "soft_sensor_experiment.json").open("w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"{'batch':10s} {'R2_A':>7s} {'R2_B':>7s} {'MAE':>7s} {'RMSE':>7s} "
          f"{'PSI_in':>7s} {'PSI_yhat':>9s} {'EDDM':>5s} {'MDM':>6s} {'MDMdr':>6s}")
    for bid, r in out.items():
        print(f"{bid:10s} {r['R2_A']:7.3f} {r['R2_B']:7.3f} {r['MAE_A']:7.3f} {r['RMSE_A']:7.3f} "
              f"{r['PSI_input_T']:7.3f} {r['PSI_predicted_Y']:9.3f} "
              f"{'Y' if r['EDDM_residual_drift'] else 'n':>5s} "
              f"{r['MDM_score']:6.3f} {'Y' if r['MDM_drift'] else 'n':>6s}")


if __name__ == "__main__":
    main()
