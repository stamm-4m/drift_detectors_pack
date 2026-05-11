"""IndPenSim benchmarking utilities for drift_detectors_pack."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pandas is required for the IndPenSim benchmark utilities. "
        'Install it with: pip install "stamm-drift-detectors[benchmark]"'
    ) from exc

from benchmarks.runner import BenchmarkCase

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REF_DIR = REPO_ROOT / "use_cases" / "IndPenSim" / "data" / "reference"
DEFAULT_TEST_DIR = REPO_ROOT / "use_cases" / "IndPenSim" / "data" / "test"

DEFAULT_PROCESS_VARS = [
    "penicillin_concentration",
    "substrate_concentration",
    "dissolved_oxygen_concentration",
    "pH",
    "temperature",
    "vessel_weight",
    "CO2_percent_in_off_gas",
    "PAA_concentration",
]


def _read_batch_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "experiment_ID" in df.columns:
        df = df.drop(columns=["experiment_ID"])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_indpensim_batches(directory: Path) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for path in sorted(Path(directory).glob("batch_*.csv")):
        out[path.stem] = _read_batch_csv(path)
    return out


def _expected_drift_label(ref_id: str, test_id: str) -> Optional[bool]:
    ref_num = ref_id.replace("batch_", "").replace(".0", "")
    test_num = test_id.replace("batch_", "").replace(".0", "")
    if ref_num == test_num:
        return False
    return True


def build_univariate_cases(
    variable: str,
    *,
    reference_dir: Path = DEFAULT_REF_DIR,
    test_dir: Path = DEFAULT_TEST_DIR,
    include_self_split_negatives: bool = True,
) -> List[BenchmarkCase]:
    refs = load_indpensim_batches(reference_dir)
    tests = load_indpensim_batches(test_dir)

    cases: List[BenchmarkCase] = []
    for ref_id, ref_df in refs.items():
        if variable not in ref_df.columns:
            continue
        ref_arr = ref_df[variable].dropna().to_numpy(dtype=float)
        if ref_arr.size < 30:
            continue

        if include_self_split_negatives and ref_arr.size >= 200:
            mid = ref_arr.size // 2
            cases.append(BenchmarkCase(
                case_id=f"{variable}::{ref_id}_selfsplit",
                reference=ref_arr[:mid], test=ref_arr[mid:],
                expected_drift=False,
                metadata={"variable": variable, "kind": "self_split"},
            ))

        for test_id, test_df in tests.items():
            if variable not in test_df.columns:
                continue
            test_arr = test_df[variable].dropna().to_numpy(dtype=float)
            if test_arr.size < 30:
                continue
            cases.append(BenchmarkCase(
                case_id=f"{variable}::{ref_id}_vs_{test_id}",
                reference=ref_arr, test=test_arr,
                expected_drift=_expected_drift_label(ref_id, test_id),
                metadata={"variable": variable, "kind": "cross_batch"},
            ))

    if include_self_split_negatives:
        for test_id, test_df in tests.items():
            if variable not in test_df.columns:
                continue
            arr = test_df[variable].dropna().to_numpy(dtype=float)
            if arr.size < 200:
                continue
            mid = arr.size // 2
            cases.append(BenchmarkCase(
                case_id=f"{variable}::{test_id}_selfsplit",
                reference=arr[:mid], test=arr[mid:],
                expected_drift=False,
                metadata={"variable": variable, "kind": "self_split"},
            ))
    return cases


def build_multivariate_cases(
    variables: Sequence[str] = DEFAULT_PROCESS_VARS,
    *,
    reference_dir: Path = DEFAULT_REF_DIR,
    test_dir: Path = DEFAULT_TEST_DIR,
    include_self_split_negatives: bool = True,
) -> List[BenchmarkCase]:
    refs = load_indpensim_batches(reference_dir)
    tests = load_indpensim_batches(test_dir)

    def _stack(df: pd.DataFrame) -> Optional[np.ndarray]:
        cols = [v for v in variables if v in df.columns]
        if not cols:
            return None
        sub = df[cols].dropna(how="any")
        if len(sub) < 30:
            return None
        return sub.to_numpy(dtype=float)

    cases: List[BenchmarkCase] = []
    for ref_id, ref_df in refs.items():
        ref_arr = _stack(ref_df)
        if ref_arr is None:
            continue
        if include_self_split_negatives and len(ref_arr) >= 200:
            mid = len(ref_arr) // 2
            cases.append(BenchmarkCase(
                case_id=f"multivariate::{ref_id}_selfsplit",
                reference=ref_arr[:mid], test=ref_arr[mid:],
                expected_drift=False,
                metadata={"kind": "self_split"},
            ))

        for test_id, test_df in tests.items():
            test_arr = _stack(test_df)
            if test_arr is None:
                continue
            shared_cols = [v for v in variables if v in ref_df.columns and v in test_df.columns]
            ref_sub = ref_df[shared_cols].dropna(how="any").to_numpy(dtype=float)
            test_sub = test_df[shared_cols].dropna(how="any").to_numpy(dtype=float)
            if len(ref_sub) < 30 or len(test_sub) < 30:
                continue
            cases.append(BenchmarkCase(
                case_id=f"multivariate::{ref_id}_vs_{test_id}",
                reference=ref_sub, test=test_sub,
                expected_drift=_expected_drift_label(ref_id, test_id),
                metadata={"variables": shared_cols, "n_variables": len(shared_cols), "kind": "cross_batch"},
            ))

    if include_self_split_negatives:
        for test_id, test_df in tests.items():
            arr = _stack(test_df)
            if arr is None or len(arr) < 200:
                continue
            mid = len(arr) // 2
            cases.append(BenchmarkCase(
                case_id=f"multivariate::{test_id}_selfsplit",
                reference=arr[:mid], test=arr[mid:],
                expected_drift=False,
                metadata={"kind": "self_split"},
            ))
    return cases
