# IndPenSim experiments — companion SoftwareX paper

Two scripts reproduce the paper's two experiments. Both write JSON to
`use_cases/IndPenSim/results/`.

| Script                | What it does                                                                                          | Output                                |
| --------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------- |
| `run_experiment_1.py` | Drift detectors (PSI, KS, MMD, PCA-CD, KDQ-Tree) over the input distribution, per phase, per batch. | `expA_per_phase.json`, `expA_multivar_per_phase.json` |
| `run_experiment_2.py` | Loads four pre-trained registry soft sensors (RF, CART, SVM, LSTM), runs them on the test batches and computes Model Disagreement (MDM) per phase, per batch. | `expB_per_phase.json`                  |

Run them from the repository root:

```bash
python -m use_cases.IndPenSim.experiments.run_experiment_1
python -m use_cases.IndPenSim.experiments.run_experiment_2
```

`run_experiment_1.py` expects the full IndPenSim CSV at
`use_cases/IndPenSim/data/100_Batches_IndPenSim_V3.1.csv` (fetch it with
`python use_cases/IndPenSim/data/download_indpensim.py` if missing).
`run_experiment_2.py` is self-contained: it uses the per-batch test CSVs
in `data/test/` and the vendored models in `models/`.

## Soft sensors (Experiment 2 panel)

Experiment 2 loads four pre-trained soft sensors from the STAMM model
registry (`models/`, with their `metadata.yaml` configs in
`models/configs/`) via `registry_models.py`:

- **RF** — Random Forest, `sklearn` (Acosta-Pavas *et al.*, 2024) — interpretable
- **CART** — Decision Tree, `sklearn` (Acosta-Pavas *et al.*, 2024) — interpretable
- **SVM** — Support Vector Regressor, `sklearn`, Z-score scalers (STAMM, 2026) — black-box
- **LSTM** — LSTM network, `keras`, Min-Max scalers (STAMM, 2026) — black-box

The two-interpretable + two-black-box mix is deliberate: disagreement
*across* learner families is a stronger degradation signal than
disagreement within one family. Each model is loaded as-is (no retraining),
with feature order and scalers read from its registry config.

**Requirements (use-case only):** `scikit-learn>=1.5` (RF/CART/SVM and the
scalers) and `tensorflow>=2.18` (the LSTM), plus `pandas` and `pyyaml`:

```bash
pip install "scikit-learn>=1.5" "tensorflow>=2.18" pandas pyyaml
```

The legacy pure-NumPy learners in `soft_sensors.py` (CART, M5, CUBIST,
RandomForest) are retained for reference but are no longer used by the
experiments.
