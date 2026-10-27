# drift_detectors_pack

A lightweight, open-source Python toolkit for **data and concept drift
detection**, designed to be *dependency-frugal*, *reproducible*, and easy to
deploy as the drift-detection layer of an MLOps stack — including
[STAMM](https://stamm.inrae.fr), the soft-sensor monitoring and maintenance
framework into which this package is integrated.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python >= 3.10](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://www.python.org/)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-green.svg)](.github/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-0.5.3-orange.svg)](pyproject.toml)

## What's new in 0.5.3

* Every constructor parameter that changes a result is documented in
  `metadata.yaml`: `score_threshold` and `use_fisher` (KDQ-tree),
  `min_consecutive_drift_signals` (EDDM) and `seed` (PCA reconstruction).
  The KDQ-tree output description now says that drift is
  `score >= score_threshold` (it said `>= alpha`). No code change.
* New test: every `__init__` parameter of a detector is in its metadata,
  except `reference_data`, `online` and `models`, which are supplied by the
  caller.

## What's new in 0.5.2

* **MDM `scale`**: the scale of the metrics that use one (MSE = RMSE / scale,
  DTW) is configurable: `"window"` (default, as before: standard deviation of the
  predictions given to `calculate`, which saturates at 1 on short windows or
  flat signals), `"reference"` (standard deviation of past predictions passed
  as `reference_data`, e.g. the run so far) or a positive number (e.g. the
  expected range of the predicted variable).
* **MDM `decision`**: the family that gives `score` and `drift`: `"error"`
  (default, as before), `"correlation"`, `"temporal"` or `"any"`.
* **Undefined correlations are reported, not invented**: Pearson, Spearman,
  lagged correlation and ACF divergence return NaN when a series is constant
  (until 0.5.1 Pearson / Spearman gave 0.5, which looked like a real value).
  MDM averages over the defined pairs, counts the others in
  `details["undefined_pairs"]`, and returns `score=None, drift=False` with
  `details["status"] = "decision_undefined"` when the deciding family is not
  defined.
* The metadata of `model_disagreement` documents `scale` and `decision`; the
  `disagreement_metrics` catalogue lists the six shipped metrics.

## What's new in 0.5.1

* Every `metadata.yaml` declares **`input_kind`**, the kind of data the
  detector consumes: `signal` (process values: distributional, sequential and
  multivariate detectors), `error` (a binary error stream of a model, 1 =
  error; needs ground truth: DDM, EDDM, RDDM, STEPD) or `predictions` (the
  outputs of two or more co-deployed models: MDM). Platforms such as the
  STAMM model registry use it to decide which data to feed each detector.
* `tests/test_metadata_schema.py` checks the required metadata keys of every
  detector.

## What's new in 0.5.0

* The **Model Disagreement Metric** (`MDM`) is now a pluggable orchestrator: pass
  `metrics=[...]` with any combination of `MSEDisagreement`,
  `PearsonDisagreement`, `SpearmanDisagreement`, or your own subclass of
  `DisagreementMetric`. The new API takes a list of prediction arrays
  (`predictions=[y1, y2, ...]`) rather than model callables, matching the
  STAMM model registry's per-batch export format.
* `MMDDetector` ships a `gamma='median'` option (median-distance heuristic),
  essential for unscaled industrial data with mixed-scale features.
* **Catalogue expanded to 30 detectors** across three families: nine
  distributional and ten sequential/error-based univariate detectors, ten
  multivariate detectors, and the model-based MDM (whose disagreement
  metrics now include three time-series-aware ones: DTW, lagged
  cross-correlation and an autocorrelation-function divergence).
* **NumPy-only runtime.** Every detector now runs on NumPy alone: `scipy`,
  `scikit-learn` and `river` were replaced by internal implementations
  (`utility/_stats.py`, `utility/_adwin.py`) and moved to the optional
  `compat` extra, used only by `tests/test_stats_equivalence.py` to verify
  the re-implementations match the reference libraries. Runtime dependencies
  are just `numpy` and `PyYAML` — no transitive `pandas`, no compiled ML
  stack, installs anywhere.
* New **IndPenSim use case** at `use_cases/IndPenSim/` that reproduces the
  experiments of the companion SoftwareX paper. Experiment 2 loads a
  structurally diverse four-model panel from the STAMM model registry ---
  RF and CART (Acosta-Pavas et al., 2024) and SVM and LSTM (STAMM, 2026)
  --- via a metadata-driven loader (`registry_models.py`).
* New per-phase methodology: each detector is run independently in the four
  canonical fermentation phases (lag, log/exponential, stationary, death),
  respecting the non-stationarity of fed-batch fermentation.
* New [`ARCHITECTURE.md`](ARCHITECTURE.md) and a self-describing
  `metadata.yaml` for every detector.


## Why this package?

Most AI systems silently degrade as the world they were trained on changes —
a phenomenon called **drift**. Detecting drift is the cornerstone of
responsible model maintenance, but the existing ecosystem is fragmented
across libraries with very different APIs, dependency trees, and quality of
documentation. The fragmentation is particularly costly in operating regimes
where ground-truth labels arrive offline hours-to-days late (industrial
soft sensors in bioprocesses) and in deployments where small dependency
footprints matter.

`drift_detectors_pack` unifies a curated catalogue of detectors behind a
single `DriftDetector.calculate()` interface and ships a self-describing
`metadata.yaml` per detector so dashboards and platforms can introspect
the catalogue without special-casing.

## Catalogue

| Name | Family | Mode | Reference |
| --- | --- | --- | --- |
| PSI | Univariate, distributional | batch | Wu & Olson, 2010 |
| KS | Univariate, distributional | batch | Smirnov, 1948 |
| Jensen-Shannon | Univariate, distributional | batch | Lin, 1991 |
| Hellinger | Univariate, distributional | batch | Hellinger, 1909 |
| Total Variation | Univariate, distributional | batch | Levin & Peres, 2017 |
| Chi-square | Univariate, distributional | batch | Pearson, 1900 |
| Bhattacharyya | Univariate, distributional | batch | Bhattacharyya, 1943 |
| Wasserstein-1 | Univariate, distributional | batch | Kantorovich, 1960 |
| Cramer-von Mises | Univariate, distributional | batch | Anderson, 1962 |
| ADWIN | Univariate, sequential | streaming | Bifet & Gavaldà, 2007 |
| Page-Hinkley | Univariate, sequential | streaming | Page, 1954 |
| HDDM-A | Univariate, sequential | streaming | Frías-Blanco et al., 2015 |
| EDDM | Univariate, error-based | streaming | Baena-García et al., 2006 |
| DDM | Univariate, error-based | streaming | Gama et al., 2004 |
| RDDM | Univariate, error-based | streaming | Barros et al., 2017 |
| CUSUM | Univariate, sequential | streaming | Page, 1954 |
| ECDD | Univariate, sequential | streaming | Ross et al., 2012 |
| KSWIN | Univariate, windowing | streaming | Raab et al., 2020 |
| STEPD | Univariate, error-based | streaming | Nishida & Yamauchi, 2007 |
| MMD | Multivariate, kernel | batch | Gretton et al., 2012 |
| MMD-linear | Multivariate, kernel | batch | Gretton et al., 2012 |
| PCA-CD | Multivariate, projection | batch | Qahtan et al., 2015 |
| PCA reconstruction | Multivariate, projection | batch | Qahtan et al., 2015 |
| KDQ-Tree | Multivariate, partition | batch | Dasu et al., 2006 |
| Energy Distance | Multivariate, distance | batch | Székely & Rizzo, 2013 |
| C2ST | Multivariate, classifier | batch | Lopez-Paz & Oquab, 2017 |
| HDDDM | Multivariate, histogram | batch | Ditzler & Polikar, 2011 |
| Multivariate KS | Multivariate, projection | batch | Cuesta-Albertos et al., 2007 |
| NN two-sample | Multivariate, neighbours | batch | Schilling, 1986 |
| MDM | Model-based, ensemble | batch | Suarez et al., 2026 (STAMM) |

## Install

```bash
pip install stamm-drift-detectors

# from source, editable
git clone https://github.com/stamm-4m/drift_detectors_pack.git
cd drift_detectors_pack
pip install -e ".[dev,benchmark]"
```

## Quick start

```python
import numpy as np
from drift_detectors import PSI, MMDDetector, ModelDisagreementMetric

# univariate
ref  = np.random.normal(0.0, 1.0, 1000)
test = np.random.normal(0.3, 1.0, 1000)
res = PSI().calculate(test, ref)
print(res.score, res.drift, res.details)

# multivariate, with the median-distance heuristic recommended for
# real-world industrial data with mixed-scale features
ref_mv  = np.random.normal(0.0, 1.0, size=(500, 4))
test_mv = np.random.normal(0.4, 1.0, size=(500, 4))
print(MMDDetector(gamma="median").calculate(test_mv, ref_mv))

# model disagreement across co-deployed soft sensors (predictions-in API):
# y1, y2, y3 are 1-D prediction arrays exported by each co-deployed model
mdm = ModelDisagreementMetric(threshold=0.25)
print(mdm.calculate(predictions=[y1, y2, y3]).score)
```

## Streaming detectors

```python
from drift_detectors import PageHinkley, HDDM_A
ph = PageHinkley(online=True)
hd = HDDM_A(online=True)
for x in stream:
    if ph.calculate([x]).drift:    # cumulative-deviation change point
        ...
    if hd.calculate([x]).drift:    # Hoeffding-bound change point
        ...
```

## Reproducing the paper experiments

The full case study lives at
[`use_cases/IndPenSim/`](use_cases/IndPenSim/), with its own README, a
fetcher script for the upstream Goldrick dataset, the two paper experiments
as runnable scripts, and the per-batch result tables in
[`use_cases/IndPenSim/results/README.md`](use_cases/IndPenSim/results/README.md):

```bash
# 1. fetch the 100-batch IndPenSim CSV (~21 MB)
python use_cases/IndPenSim/data/download_indpensim.py

# 2. run the two paper experiments
python -m use_cases.IndPenSim.experiments.run_experiment_1
python -m use_cases.IndPenSim.experiments.run_experiment_2

# 3. (optional) regenerate the paper figures
python use_cases/IndPenSim/figures/make_class_diagram.py
python use_cases/IndPenSim/figures/make_fault_timelines.py
```

A general-purpose `BenchmarkRunner` lives in [`benchmarks/`](benchmarks/) for
evaluating any detector against any labelled dataset --- not used by the paper
protocol but documented in [`benchmarks/README.md`](benchmarks/README.md).


## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the layered design, the result
dataclass schema, the `metadata.yaml` schema, and a step-by-step recipe
for adding a new detector. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for
the development workflow.

## STAMM integration

This package is component (V) of the
[STAMM](https://stamm.inrae.fr) MLOps platform for industrial soft sensors.
STAMM hosts heterogeneous models (CART, M5, CUBIST, Random Forest, GBM,
LSTM and others) behind a common REST endpoint, runs continuous drift
monitoring through this package, and exposes the model-divergence panel
that is the dashboard counterpart of the `ModelDisagreementMetric`
detector.

## Detector pack (STAMM model-registry upload)

The STAMM model registry installs this toolkit as an uploadable, versioned
*detector pack*: its `register` endpoint ingests a **slim `drift_detectors/`
archive** (the package only -- no data, tests, use cases or benchmarks),
reads the version, parses every detector's `metadata.yaml` into a read-only
catalogue, and pins one active version that the workflow orchestrator runs
server-side. Build that archive with:

```bash
python tools/build_detector_pack.py
# -> dist/drift_detectors_pack-<version>.zip   (drift_detectors/ + pyproject.toml)
```

Upload the resulting zip through the registry UI / `register` endpoint. Only
folders that contain both `detector.py` and `metadata.yaml` are registered as
detectors.
