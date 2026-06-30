# drift_detectors_pack

A lightweight, open-source Python toolkit for **data and concept drift
detection**, designed to be *dependency-frugal*, *reproducible*, and easy to
deploy as the drift-detection layer of an MLOps stack — including
[STAMM](https://stamm.inrae.fr), the soft-sensor monitoring and maintenance
framework into which this package is integrated.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python >= 3.10](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://www.python.org/)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-green.svg)](.github/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-0.4.0-orange.svg)](pyproject.toml)

## What's new in 0.4.0

* The **Model Disagreement Metric** (`MDM`) is now a pluggable orchestrator: pass
  `metrics=[...]` with any combination of `MSEDisagreement`,
  `PearsonDisagreement`, `SpearmanDisagreement`, or your own subclass of
  `DisagreementMetric`. The new API takes a list of prediction arrays
  (`predictions=[y1, y2, ...]`) rather than model callables, matching the
  STAMM model registry's per-batch export format.
* `MMDDetector` ships a `gamma='median'` option (median-distance heuristic),
  essential for unscaled industrial data with mixed-scale features.
* Three pure-NumPy streaming detectors (`PageHinkley`, `HDDM_A`, `EDDM`)
  remove a transitive dependency on `scikit-multiflow`.
* New **IndPenSim use case** at `use_cases/IndPenSim/` that reproduces the
  experiments of the companion SoftwareX paper, including a
  pure-NumPy reimplementation of the four interpretable soft sensors of
  Acosta-Pavas et al. (2024): CART, M5, CUBIST and Random Forest.
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

| Name         | Family                       | Mode      | Reference                                     |
| ------------ | ---------------------------- | --------- | --------------------------------------------- |
| PSI          | Univariate, distributional   | batch     | Wu & Olson, 2010                              |
| KS-test      | Univariate, distributional   | batch     | Smirnov, 1948                                 |
| ADWIN        | Univariate, sequential       | streaming | Bifet & Gavaldà, 2007                         |
| Page-Hinkley | Univariate, sequential       | streaming | Page, 1954                                    |
| HDDM-A       | Univariate, sequential       | streaming | Frías-Blanco *et al.*, 2015                   |
| EDDM         | Univariate, error-based      | streaming | Baena-García *et al.*, 2006                   |
| MMD          | Multivariate, kernel         | batch     | Gretton *et al.*, 2012                        |
| PCA-CD       | Multivariate, projection     | batch     | Qahtan *et al.*, 2015                         |
| KDQ-Tree     | Multivariate, partition      | batch     | Dasu *et al.*, 2006                           |
| MDM          | Model-based, ensemble        | batch     | Suarez *et al.*, 2026 (STAMM)                 |

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

# model disagreement across two co-deployed soft sensors
def linear(X):  return X @ w_lin + b_lin
def tree(X):    return rf.predict(X)
mdm = ModelDisagreementMetric(models=[linear, tree], threshold=0.25)
print(mdm.calculate(X_live).score)
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

## Citing

```bibtex
@article{galindez2026driftdetectors,
  title   = {drift\_detectors\_pack: A unified drift detection toolkit for
             soft sensor monitoring in industrial bioprocesses},
  author  = {Galindez, Elizabeth and Crowther, Matthew and Metcalfe, Brett and
             Koehorst, Jasper J. and Aristizabal Morales, Santiago and
             Su\'arez, Carlos and Daboussi, Fayza and Corrales, David Camilo},
  journal = {SoftwareX},
  year    = {2026},
  note    = {Under review}
}
```

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
