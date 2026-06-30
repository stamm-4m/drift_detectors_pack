"""Experiment 1 of the companion SoftwareX paper.

Drift detectors over the input distribution of test batches 91-100, computed
INDEPENDENTLY in each of the four canonical fermentation phases (lag, log,
stationary, death), against the same phase of the LSTM training set (batches
1-60 of IndPenSim).

Produces use_cases/IndPenSim/results/expA_per_phase.json and
         use_cases/IndPenSim/results/expA_multivar_per_phase.json.

Run with:
    python -m use_cases.IndPenSim.experiments.run_experiment_1
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
REPO = CASE.parent.parent
sys.path.insert(0, str(REPO))

from drift_detectors import PSI, KSDetector, MMDDetector, PCA_CD, KDQTree

CSV       = CASE / "data" / "100_Batches_IndPenSim_V3.1.csv"
OUT_UNI   = CASE / "results" / "expA_per_phase.json"
OUT_MULT  = CASE / "results" / "expA_multivar_per_phase.json"

PREDICTORS = {
    "Fs":      "Sugar feed rate(Fs:L/h)",
    "RPM":     "Agitator RPM(RPM:RPM)",
    "T":       "Temperature(T:K)",
    "pH":      "pH(pH:pH)",
    "DO2":     "Dissolved oxygen concentration(DO2:mg/L)",
    "V":       "Vessel Volume(V:L)",
    "CO2_off": "carbon dioxide percent in off-gas(CO2outgas:%)",
    "O2_off":  "Oxygen in percent in off-gas(O2:O2  (%))",
    "Aer":     "Aeration rate(Fg:L/h)",
}
TIME_COL = "Time (h)"
BATCH_COL = "Batch ID"

# Four canonical microbial growth phases. Time windows match Stanbury et al.
# (2016) and Shuler & Kargi (2017) for a 230-h fed-batch fermentation.
PHASES = {
    "lag":        (0,   30),
    "log":        (30,  120),
    "stationary": (120, 200),
    "death":      (200, 300),
}
REF_BATCHES  = list(range(1, 61))
TEST_BATCHES = [91, 92, 93, 94, 95, 96, 97, 98, 99, 100]


def load() -> pd.DataFrame:
    if not CSV.exists():
        raise SystemExit(
            f"IndPenSim CSV not found at {CSV}.\n"
            "Run use_cases/IndPenSim/data/download_indpensim.py first."
        )
    cols = list(PREDICTORS.values()) + [TIME_COL, BATCH_COL]
    df = pd.read_csv(CSV, usecols=cols)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def phase_slice(df, batch, t0, t1, cols):
    sub = df[(df[BATCH_COL] == batch) & (df[TIME_COL] >= t0) & (df[TIME_COL] < t1)]
    return sub[cols].dropna(how="any")


def main():
    df = load()
    print(f"loaded {CSV.name}: {len(df):,} rows")

    # univariate: per phase, per variable
    uni_rows = []
    for ph, (t0, t1) in PHASES.items():
        ref_per = {}
        for k, col in PREDICTORS.items():
            ref_per[k] = pd.concat(
                [df[(df[BATCH_COL] == b) & (df[TIME_COL] >= t0) & (df[TIME_COL] < t1)][col].dropna()
                 for b in REF_BATCHES]).to_numpy()
        for b in TEST_BATCHES:
            row = {"phase": ph, "batch": b}
            for k, col in PREDICTORS.items():
                test = df[(df[BATCH_COL] == b) & (df[TIME_COL] >= t0) & (df[TIME_COL] < t1)][col].dropna().to_numpy()
                if test.size < 30 or ref_per[k].size < 30:
                    row[f"PSI_{k}"] = float("nan"); row[f"KS_{k}"] = float("nan")
                    continue
                row[f"PSI_{k}"] = float(PSI().calculate(test, ref_per[k]).score)
                row[f"KS_{k}"]  = float(KSDetector().calculate(test, ref_per[k]).score)
            uni_rows.append(row)
        print(f"  phase {ph}: {len([r for r in uni_rows if r['phase']==ph])} rows")
    OUT_UNI.write_text(json.dumps(uni_rows, indent=2, default=str))
    print(f"wrote {OUT_UNI}")

    # multivariate: per phase, joint
    mv_rows = []
    rng = np.random.default_rng(0)
    cols = [PREDICTORS[k] for k in ("Fs", "RPM", "T", "pH", "DO2", "V", "CO2_off", "O2_off")]
    for ph, (t0, t1) in PHASES.items():
        ref = pd.concat(
            [phase_slice(df, b, t0, t1, cols) for b in REF_BATCHES]).to_numpy()
        if len(ref) > 2000:
            ref = ref[rng.choice(len(ref), size=2000, replace=False)]
        for b in TEST_BATCHES:
            test = phase_slice(df, b, t0, t1, cols).to_numpy()
            if test.size == 0:
                continue
            if len(test) > 2000:
                test = test[rng.choice(len(test), size=2000, replace=False)]
            mv_rows.append({
                "phase": ph, "batch": b,
                "MMD_med": float(MMDDetector(gamma="median").calculate(test, ref).score),
                "PCA_CD":  float(PCA_CD().calculate(test, ref).score),
                "KDQ_Tree": float(KDQTree(k_neighbors=25).calculate(test, ref).score),
            })
        print(f"  multivariate phase {ph}: {len([r for r in mv_rows if r['phase']==ph])} rows")
    OUT_MULT.write_text(json.dumps(mv_rows, indent=2, default=str))
    print(f"wrote {OUT_MULT}")


if __name__ == "__main__":
    main()
