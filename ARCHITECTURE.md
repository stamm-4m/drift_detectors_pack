# drift_detectors_pack — Architecture

This document complements the README with the architectural detail referenced
from the Software description section of the companion SoftwareX paper. It is aimed at developers
adding a new detector and at platform engineers integrating the package
into a wider MLOps stack such as STAMM (https://stamm.inrae.fr).

## 1. Layered design

```
                       +---------------------------------+
   user-facing API     |      from drift_detectors       |
                       +---------------------------------+
                                    │
                                    ▼
                       +---------------------------------+
   abstraction layer   |   DriftDetector  (ABC) +        |
                       |   ScoreDriftResult /            |
                       |   StreamingDriftResult /        |
                       |   PointwiseDriftResult          |
                       +---------------------------------+
                                    │
       ┌────────────────────────────┼────────────────────────────────┐
       ▼                            ▼                                ▼
+--------------+         +--------------------+        +-----------------------+
| univariate/  |         | multivariate/      |        | model_based/          |
|   psi/       |         |   mmd/             |        |   model_disagreement/ |
|   ks/        |         |   pca_cd/          |        |                       |
|   adwin/     |         |   kdq_tree/        |        |                       |
|   page_      |         |                    |        |                       |
|     hinkley/ |         |                    |        |                       |
|   hddm_a/    |         |                    |        |                       |
|   eddm/      |         |                    |        |                       |
+--------------+         +--------------------+        +-----------------------+
       │                            │                                │
       └─────── each leaf folder is a self-describing module ────────┘
                  detector.py  +  metadata.yaml  +  usage.md
```

The package is split into three families that match the way drift problems
arise in practice:

* **`univariate/`** — one variable at a time. Two distributional detectors
  (`PSI`, `KS`), three sequential / change-point detectors (`Adwin`,
  `PageHinkley`, `HDDM_A`), and one error-based detector (`EDDM`).
* **`multivariate/`** — joint distribution over a feature matrix
  (`MMDDetector`, `PCA_CD`, `KDQTree`).
* **`model_based/`** — detectors that consume *predictions* from one or more
  deployed models rather than raw inputs. Currently `ModelDisagreementMetric`
  (Section 3 of the companion paper); this folder is the natural home for
  future deep-learning residual detectors.

Every detector subclasses `DriftDetector` and exposes the same
`calculate(test_data, reference_data=None)` entry point, so dashboards, CI
jobs, and downstream platforms can iterate over the catalogue without
special-casing the families.

## 2. Result dataclasses

Three result types cover the patterns observed across the literature:

| dataclass | when to use it | minimal fields |
|---|---|---|
| `ScoreDriftResult` | batch detectors that produce a numeric drift score (PSI, KS, MMD, PCA-CD, KDQ-Tree, MDM) | `score: float`, `drift: bool`, `details: dict` |
| `StreamingDriftResult` | streaming detectors that report the position of the most-recent drift (ADWIN, Page-Hinkley, HDDM-A, EDDM) | `last_index: int`, `drift: bool`, `details: dict` |
| `PointwiseDriftResult` | reserved for detectors that emit a list of change-points; not yet used in the shipped catalogue | `indices: list[int]`, `details: dict` |

The `details` dictionary always carries the configured parameters, the
mode (`"online"` or `"offline"`), and the reference / test sample sizes.
Detector-specific keys (`p_value`, `gamma`, `warning`, `metric_means`,
`pairwise`, ...) are documented in the corresponding `metadata.yaml`.

## 3. Self-describing metadata

Each detector ships a `metadata.yaml` next to its `detector.py`. The schema
has six top-level keys:

```yaml
name:        "Population Stability Index"
acronym:     "PSI"
version:     "1.0"
type:        "univariate"           # univariate | multivariate | model_based
mode:        "batch"                # batch | streaming
description: >
  ...natural-language summary, several sentences...
inputs:                             # named inputs the calculate() call expects
  test_data:
    description: ...
parameters:                         # constructor / call parameters
  threshold:
    description: ...
output:                             # what the result dataclass carries
  description: ...
  fields:
    score:        { description: ... }
    drift:        { description: ... }
    details.foo:  { description: ... }
references:                         # bibliographic anchors
  - "Page, E. S. (1954). Continuous inspection schemes. Biometrika 41, 100-115."
```

Two helpers consume this metadata:

* `drift_detector.load_metadata_for_class(cls)` — load the YAML for one
  class. Used by `DriftDetector.__init__` so that every instance has a
  populated `self.metadata` dict.
* `drift_detector.get_metadata()` — recursively walks the package and
  returns `{detector_path: metadata_dict}`, suitable for rendering a
  dashboard catalogue (this is what the STAMM dashboard does).

The metadata files are shipped as package data via `pyproject.toml`
(`[tool.setuptools.package-data]`), so they are present in the installed
wheel and can be loaded by importing applications without special handling.

## 4. The Model Disagreement Metric (MDM) — design notes

`drift_detectors.model_based.model_disagreement.ModelDisagreementMetric` is
the package's first model-based detector. It targets the operating regime
described by the STAMM paper: industrial soft sensors whose ground-truth
labels arrive offline hours-to-days late, leaving residual-based monitoring
unavailable for the bulk of the operating window. In that regime, a
growing pair-wise disagreement between co-deployed models is the only
continuously-available proxy for performance degradation.

The detector takes a sequence of `models = [f1, f2, ...]` where each
`fi(X) -> y_hat` is a callable returning a 1-D NumPy array. For each pair
`(i, j)` it computes three normalised disagreement statistics:

| metric         | formula                                            | what it captures             |
|----------------|----------------------------------------------------|------------------------------|
| `mse`          | `min(sqrt(mean((y_i - y_j)**2)) / std(stack), 1)`  | output-magnitude divergence  |
| `pearson`      | `(1 - corr(y_i, y_j)) / 2`                         | linear-correlation divergence|
| `spearman`     | `(1 - corr(rank(y_i), rank(y_j))) / 2`             | rank-correlation divergence  |

Each statistic lies in `[0, 1]`. The aggregate score is the mean over the
three statistics of the mean over pairs; `details["pairwise"]` exposes the
per-pair matrix per statistic so a dashboard can attribute the divergence
to specific model pairs (this is what STAMM's "Model divergence" view
does).

## 5. Adding a new detector

1. Create a folder under `src/drift_detectors/<family>/<your_name>/`.
2. Add `detector.py` — class subclassing `DriftDetector`, implementing
   `calculate(...)` and returning the appropriate result dataclass.
3. Add `metadata.yaml` — follow the schema in §3.
4. Add `usage.md` — a minimal worked example, mock data only.
5. Re-export the class from `drift_detectors/__init__.py` and add it to
   `__all__`.
6. Add unit tests under `tests/test_<your_name>.py`. Every detector ships
   at least four tests: `*_drift_detected`, `*_no_drift`,
   `*_metadata_loaded`, `*_output_fields`.
7. (Optional) Wire the detector into `benchmarks.runner.discover_default_detectors()`
   so it appears in the IndPenSim reproducible benchmarks.

## 6. Footprint

The package depends only on `numpy`, `scipy`, `scikit-learn`, `river` and
`PyYAML`. On the IndPenSim case study the largest detector (KDQ-Tree)
runs in under 25 KiB of allocator-level memory delta and ~260 ms per case;
the streaming detectors (Page-Hinkley, HDDM-A, EDDM) are essentially
memory-free at steady state. The whole package is under 200 MiB once
installed alongside its dependencies, which makes it tractable on a
Raspberry Pi-class machine.
