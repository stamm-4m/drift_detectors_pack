# IndPenSim experiments — AI4D 2026 paper

Two scripts reproduce the paper's two experiments. Both write JSON to
`use_cases/IndPenSim/results/`.

| Script                | What it does                                                                                          | Output                                |
| --------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------- |
| `run_experiment_1.py` | Drift detectors (PSI, KS, MMD, PCA-CD, KDQ-Tree) over the input distribution, per phase, per batch. | `expA_per_phase.json`, `expA_multivar_per_phase.json` |
| `run_experiment_2.py` | Trains CART, M5, CUBIST and Random Forest soft sensors on batches 1–60 and computes Model Disagreement (MDM) per phase, per batch. | `expB_per_phase.json`                  |

Run them from the repository root:

```bash
python -m use_cases.IndPenSim.experiments.run_experiment_1
python -m use_cases.IndPenSim.experiments.run_experiment_2
```

Both scripts expect the full IndPenSim CSV at
`use_cases/IndPenSim/data/100_Batches_IndPenSim_V3.1.csv`. Fetch it with
`python use_cases/IndPenSim/data/download_indpensim.py` if missing.

## Soft sensors

`soft_sensors.py` is a pure-NumPy reimplementation of the four
interpretable learners studied by Acosta-Pavas *et al.* (2024):

- **CART** — recursive binary splits on MSE
- **M5** — CART with a per-leaf ridge regression
- **CUBIST** — rule-extracting M5 with averaged firing rules
- **RandomForest** — bag of CART trees with feature bagging

These are faithful to the algorithmic core but light by design (no
scikit-learn, no R packages) so that the experiments can be reproduced in
constrained environments.
