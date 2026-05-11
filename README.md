# drift_detectors_pack

A lightweight, open-source Python toolkit for **data and concept drift
detection**, designed to be *dependency-frugal*, *reproducible*, and easy to
deploy as the drift-detection layer of an MLOps stack — including
[STAMM](https://stamm.inrae.fr), the soft-sensor monitoring and maintenance
framework into which this package is integrated.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python >= 3.10](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://www.python.org/)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-green.svg)](.github/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-0.3.0-orange.svg)](pyproject.toml)

## What's in v0.3.0

* Ten unified detectors behind a single `DriftDetector.calculate()` interface.
* New **Model Disagreement Metric** (`MDM`) for soft-sensor regimes where
  ground-truth labels are delayed, mirroring the STAMM model-divergence panel.
* Three new pure-NumPy streaming detectors (`PageHinkley`, `HDDM_A`, `EDDM`)
  that remove a transitive dependency on `scikit-multiflow`.
* `MMDDetector` now supports `gamma='median'` (median-distance heuristic),
  which we show is essential for unscaled industrial data.
* Reproducible benchmarking harness (`benchmarks/`) with built-in IndPenSim
  loaders and an LSTM-style soft-sensor experiment.
* New [`ARCHITECTURE.md`](ARCHITECTURE.md) for developers and integrators.

## Why this package?

Most AI systems silently degrade as the world they were trained on changes —
a phenomenon called **drift**. Detecting drift is the cornerstone of
responsible model maintenance, but the existing ecosystem is fragmented
across libraries with very different APIs, dependency trees, and quality of
documentation. The fragmentation is particularly costly in operating regimes
where ground-truth labels arrive offline hours-to-days late (industrial
soft sensors, telemedicine triage, low-cost sensor networks), and in
deployments where small dependency footprints matter — e.g. AI deployments
in low- and middle-income contexts.

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
git clone https://gitlab.com/stamm-4m/drift_detectors_pack.git
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

## Reproducible benchmarks

The `benchmarks/` module provides a `BenchmarkRunner` that evaluates any
detector against any labelled dataset and reports detection rate, runtime,
and memory in a single CSV. The IndPenSim case study is wired in by default:

```bash
python -m benchmarks.run_indpensim --detectors all --output results/
python benchmarks/soft_sensor.py            # LSTM-style soft-sensor experiment
```

A companion paper (AI4D 2026 / CAEPIA 2026) describes the design choices,
discusses the new MDM detector, and reports two experiments: a wide
multi-detector sweep on IndPenSim and a soft-sensor degradation experiment
that reproduces the protocol of Metcalfe *et al.* (2025) on faulty test
batches 91–100.

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
@inproceedings{corrales2026driftdetectors,
  title     = {drift\_detectors\_pack: An Open-Source Drift Detection Toolkit
               for Soft-Sensor Monitoring in Industrial Bioprocesses},
  author    = {Corrales, David Camilo and Crowther, Matthew and Metcalfe, Brett and
               Koehorst, Jasper J. and Su\'arez Mu\~noz, Carlos Alberto},
  booktitle = {Proceedings of the 2nd Workshop on Artificial Intelligence for
               Development (AI4D 2026), CAEPIA 2026},
  year      = {2026}
}
```

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
