# Drift Detection Framework
A Python library for detecting data drift in univariate and multivariate data.

## Features
- Unified `DriftDetector` interface with a single `calculate()` method
- Supports statistical and model-based drift detection
- Works with static datasets and data streams
- Built-in detectors:
  - PSI, KS Test, ADWIN, MMD, PCA-CD, KDQ-Tree
- Human-readable YAML metadata per detector
- Structured result outputs (e.g., `ScoreDriftResult`, `StreamingDriftResult`, etc.)

## Resources
Resources for running detectors exist in several places.
1. Within each drift detector, a `usage.md` file exists with an example using mock data.
2. Within the `use_case` directory, several examples exist with detectors used with real data.

## Quick Example
```python
from drift_detectors.univariate.psi.detector import PSI
import numpy as np

ref = np.random.normal(0, 1, 1000)
test = np.random.normal(0.3, 1, 1000)

detector = PSI()
result = detector.calculate(ref, test)

print("PSI Score:", result.score)
print("Drift Detected:", result.drift)
```

## Data/Function flow
1. Pick drift detector.
   1. `from drift_detectors.univariate.psi.detector import PSI`
2. initialise detector.
   1. `detector = PSI()`
3. detect drift.
   1. `results = detector.calculate(ref_data, test_data)`
4. Analyse results.
   1. `results.score # Amount of drift`
   2. `results.drift # has drift`
   3. `results.details # run metadata`

## Existing Detectors
| Name     | Type         | Online | Example |
| -------- | ------------ | ------ | ------- |
| PSI      | Univariate   | No     | N/A     |
| KS       | Univariate   | No     | N/A     |
| ADWIN    | Univariate   | Yes    | N/A     |
| MMD      | Multivariate | No     | N/A     |
| PCA-CD   | Multivariate | No     | N/A     |
| KDQ-Tree | Multivariate | No     | N/A     |

## Adding a New Detector
1. Create a new module under univariate/ or multivariate/
2. Inherit from `DriftDetector`
3. Implement `calculate()` -> `DriftResult`
4. Add a `meta.yaml` file next to your detector