"""Cross-platform cleanup: delete the zero-byte placeholder files left by
the repo-alignment pass, then remove any directories that become empty.

Background. After aligning the repository with the final 12-page AI4D 2026
paper, a number of files were superseded:
  - the old use_cases/indepensim_data/ folder (now use_cases/IndPenSim/data/test/)
  - the top-level results/ folder (now use_cases/IndPenSim/results/)
  - the legacy results/legacy/ artifacts from earlier iterations
  - the benchmarks/soft_sensor*.py tombstones
  - the deprecated tests/test_model_disagreement_metric.py
  - the sandbox_shim.py stub
  - the legacy use_cases/psi_Indpensim.py demo

These were truncated to 0 bytes in the read-only sandbox where the
alignment was performed; this script removes them on a normal checkout.
Safe to re-run. Works on Windows, macOS, and Linux.

Usage (from the repo root):
    python tools/cleanup.py
"""
from __future__ import annotations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TARGETS = [
    "use_cases/indepensim_data/README.md",
    "use_cases/indepensim_data/reference_data/batch_92.0.csv",
    "use_cases/indepensim_data/reference_data/batch_93.0.csv",
    "use_cases/indepensim_data/test_data/batch_91.0.csv",
    "use_cases/indepensim_data/test_data/batch_94.0.csv",
    "use_cases/indepensim_data/test_data/batch_95.0.csv",
    "use_cases/indepensim_data/test_data/batch_96.0.csv",
    "use_cases/indepensim_data/test_data/batch_97.0.csv",
    "use_cases/indepensim_data/test_data/batch_98.0.csv",
    "use_cases/indepensim_data/test_data/batch_99.0.csv",
    "use_cases/indepensim_data/test_data/batch_100.0.csv",
    "use_cases/psi_Indpensim.py",
    "use_cases/IndPenSim/experiments/README_DEPRECATED.txt",
    "results/expA_per_phase.json",
    "results/expA_multivar_per_phase.json",
    "results/expB_per_phase.json",
    "results/expB_model_disagreement.json",
    "results/figures/fig_class_diagram.png",
    "results/figures/fig_fault_timelines.png",
    "results/figures/fig_stamm_integration.png",
    "results/legacy/expA_drift_inputs.json",
    "results/legacy/indpensim_multivariate.csv",
    "results/legacy/indpensim_univariate_CO2_percent_in_off_gas.csv",
    "results/legacy/indpensim_univariate_dissolved_oxygen_concentration.csv",
    "results/legacy/indpensim_univariate_pH.csv",
    "results/legacy/indpensim_univariate_penicillin_concentration.csv",
    "results/legacy/indpensim_univariate_substrate_concentration.csv",
    "results/legacy/multivariate_summary.json",
    "results/legacy/soft_sensor_experiment.json",
    "results/legacy/univariate_summary.json",
    "sandbox_shim.py",
    "tests/test_model_disagreement_metric.py",
    "benchmarks/soft_sensor.py",
    "benchmarks/soft_sensors.py",
]

EMPTY_DIRS = [
    "use_cases/indepensim_data/reference_data",
    "use_cases/indepensim_data/test_data",
    "use_cases/indepensim_data",
    "results/figures",
    "results/legacy",
    "results",
]

removed_files = 0
removed_dirs = 0

for rel in TARGETS:
    path = REPO_ROOT / rel
    if path.is_file() and path.stat().st_size == 0:
        path.unlink()
        print(f"  rm  {rel}")
        removed_files += 1

for rel in EMPTY_DIRS:
    path = REPO_ROOT / rel
    if path.is_dir() and not any(path.iterdir()):
        path.rmdir()
        print(f"  rmdir  {rel}")
        removed_dirs += 1

print(f"\nDone. Removed {removed_files} files and {removed_dirs} directories.")
