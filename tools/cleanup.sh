#!/usr/bin/env bash
# Remove the empty placeholder files left by the repo-alignment pass.
#
# Background. After aligning the repository with the final 12-page AI4D 2026
# paper, a number of files were superseded:
#   - the old `use_cases/indepensim_data/` folder (now `use_cases/IndPenSim/data/test/`)
#   - the top-level `results/` folder (now `use_cases/IndPenSim/results/`)
#   - the legacy `results/legacy/` artifacts from earlier iterations
#   - the `benchmarks/soft_sensor*.py` tombstones
#   - the deprecated `tests/test_model_disagreement_metric.py`
#   - the `sandbox_shim.py` stub
#   - the legacy `use_cases/psi_Indpensim.py` demo
#
# These files were emptied (truncated to 0 bytes) inside the read-only sandbox
# where the alignment was performed; this script does the actual `rm` on a
# normal checkout. Safe to re-run --- only zero-byte files are removed.
#
# Usage:
#   bash tools/cleanup.sh

set -euo pipefail
cd "$(dirname "$0")/.."

# Files left as zero-byte placeholders by the alignment pass
TARGETS=(
    "use_cases/indepensim_data/README.md"
    "use_cases/indepensim_data/reference_data/batch_92.0.csv"
    "use_cases/indepensim_data/reference_data/batch_93.0.csv"
    "use_cases/indepensim_data/test_data/batch_91.0.csv"
    "use_cases/indepensim_data/test_data/batch_94.0.csv"
    "use_cases/indepensim_data/test_data/batch_95.0.csv"
    "use_cases/indepensim_data/test_data/batch_96.0.csv"
    "use_cases/indepensim_data/test_data/batch_97.0.csv"
    "use_cases/indepensim_data/test_data/batch_98.0.csv"
    "use_cases/indepensim_data/test_data/batch_99.0.csv"
    "use_cases/indepensim_data/test_data/batch_100.0.csv"
    "use_cases/psi_Indpensim.py"
    "use_cases/IndPenSim/experiments/README_DEPRECATED.txt"
    "results/expA_per_phase.json"
    "results/expA_multivar_per_phase.json"
    "results/expB_per_phase.json"
    "results/expB_model_disagreement.json"
    "results/figures/fig_class_diagram.png"
    "results/figures/fig_fault_timelines.png"
    "results/figures/fig_stamm_integration.png"
    "results/legacy/expA_drift_inputs.json"
    "results/legacy/indpensim_multivariate.csv"
    "results/legacy/indpensim_univariate_CO2_percent_in_off_gas.csv"
    "results/legacy/indpensim_univariate_dissolved_oxygen_concentration.csv"
    "results/legacy/indpensim_univariate_pH.csv"
    "results/legacy/indpensim_univariate_penicillin_concentration.csv"
    "results/legacy/indpensim_univariate_substrate_concentration.csv"
    "results/legacy/multivariate_summary.json"
    "results/legacy/soft_sensor_experiment.json"
    "results/legacy/univariate_summary.json"
    "sandbox_shim.py"
    "tests/test_model_disagreement_metric.py"
    "benchmarks/soft_sensor.py"
    "benchmarks/soft_sensors.py"
)

removed=0
for f in "${TARGETS[@]}"; do
    if [[ -f "$f" && ! -s "$f" ]]; then
        rm -- "$f"
        echo "  rm $f"
        ((removed++))
    fi
done

# Empty directories left behind
EMPTY_DIRS=(
    "use_cases/indepensim_data/reference_data"
    "use_cases/indepensim_data/test_data"
    "use_cases/indepensim_data"
    "results/figures"
    "results/legacy"
    "results"
)
for d in "${EMPTY_DIRS[@]}"; do
    if [[ -d "$d" ]] && ! find "$d" -mindepth 1 -print -quit | grep -q .; then
        rmdir -- "$d"
        echo "  rmdir $d"
        ((removed++))
    fi
done

echo "Done. Removed $removed items."
