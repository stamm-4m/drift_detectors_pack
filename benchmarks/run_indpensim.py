"""
CLI for running the IndPenSim benchmark across the full detector catalogue.

Examples
--------
    python -m benchmarks.run_indpensim --output results/
    python -m benchmarks.run_indpensim --variable penicillin_concentration
    python -m benchmarks.run_indpensim --multivariate --output results/multi/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.indpensim import (
    DEFAULT_PROCESS_VARS,
    build_multivariate_cases,
    build_univariate_cases,
)
from benchmarks.runner import BenchmarkRunner, discover_default_detectors


UNIVARIATE_DETECTORS = ["PSI", "KS", "ADWIN", "PageHinkley", "HDDM-A"]
MULTIVARIATE_DETECTORS = ["MMD", "PCA-CD", "KDQ-Tree"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run drift_detectors_pack on IndPenSim.")
    parser.add_argument(
        "--variable",
        default="penicillin_concentration",
        help="Process variable for univariate detectors (ignored if --multivariate).",
    )
    parser.add_argument(
        "--multivariate",
        action="store_true",
        help="Run multivariate detectors on the default variable bundle.",
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Directory in which to write the benchmark CSV.",
    )
    parser.add_argument(
        "--detectors",
        nargs="*",
        default=None,
        help="Subset of detector names to run. Defaults to all suitable for the mode.",
    )
    args = parser.parse_args()

    all_factories = discover_default_detectors()
    if args.multivariate:
        cases = build_multivariate_cases(DEFAULT_PROCESS_VARS)
        default_subset = MULTIVARIATE_DETECTORS
        out_name = "indpensim_multivariate.csv"
    else:
        cases = build_univariate_cases(args.variable)
        default_subset = UNIVARIATE_DETECTORS
        out_name = f"indpensim_univariate_{args.variable}.csv"

    chosen = args.detectors or default_subset
    factories = {n: f for n, f in all_factories.items() if n in chosen}

    print(
        f"Running {len(factories)} detectors over {len(cases)} cases "
        f"({'multivariate' if args.multivariate else f'univariate :: {args.variable}'})."
    )
    runner = BenchmarkRunner(factories, cases)
    runner.run()

    out_path = Path(args.output) / out_name
    runner.to_csv(out_path)
    print(f"Wrote {len(runner.results)} result rows to {out_path}")

    # Print a tidy summary to stdout.
    print("\nSummary by detector:")
    print(f"{'detector':12s}  {'n':>4s}  {'TP':>4s}  {'FP':>4s}  {'TN':>4s}  {'FN':>4s}  "
          f"{'precision':>9s}  {'recall':>6s}  {'mean_ms':>8s}")
    for det, m in runner.summary_by_detector().items():
        print(f"{det:12s}  {m['n']:4d}  {m['TP']:4d}  {m['FP']:4d}  "
              f"{m.get('TN',0):4d}  {m.get('FN',0):4d}  "
              f"{m['precision']:9.3f}  {m['recall']:6.3f}  {m['mean_runtime_ms']:8.2f}")


if __name__ == "__main__":
    main()
