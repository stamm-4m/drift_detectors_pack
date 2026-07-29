# Hellinger Distance Drift Detection Method (HDDDM)
The **Hellinger Distance Drift Detection Method (HDDDM)** detector (Ditzler & Polikar, 2011) is a *multivariate*, batch drift detector. It computes a per-feature histogram Hellinger distance between the reference and test samples and averages it over all features. Inputs are 2-D arrays shaped `(n_samples, n_features)`: a *reference* set and a *test* set.

The score is bounded in `[0, 1]`: `0` when every marginal coincides, `1` when the marginals are disjoint. It is a fast, interpretable screen for feature-wise distribution shift.

Like all detectors in the package, it supports three usage patterns:

* **Stateless** — pass both reference and test data to `calculate()` each time.
* **Stateful (offline)** — store the reference once, then pass each test batch.
* **Stateful (online)** — store the reference and accumulate test rows incrementally across calls.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `threshold` | `0.2` | drift is flagged when `score >= threshold` |
| `bins` | `20` | histogram bins per feature |
| `online` | `False` | accumulate test rows across calls |

## Stateless Usage
```python
import numpy as np
from drift_detectors import HDDDM

rng = np.random.default_rng(42)
reference = rng.normal(0.0, 1.0, size=(500, 4))
test = rng.normal(0.6, 1.0, size=(400, 4))

detector = HDDDM()
result = detector.calculate(test_data=test, reference_data=reference)

print("Score          :", result.score)
print("Drift Detected :", result.drift)
print("Details        :", result.details)
```

## Stateful (Offline) Usage
```python
import numpy as np
from drift_detectors import HDDDM

rng = np.random.default_rng(42)
reference = rng.normal(0.0, 1.0, size=(500, 4))
test = rng.normal(0.6, 1.0, size=(400, 4))

detector = HDDDM(reference_data=reference)   # or detector.set_reference_data(reference)
result = detector.calculate(test_data=test)

print("Score          :", result.score)
print("Drift Detected :", result.drift)
```

## Stateful (Online) Usage
```python
import numpy as np
from drift_detectors import HDDDM

rng = np.random.default_rng(42)
reference = rng.normal(0.0, 1.0, size=(500, 4))
stream = rng.normal(0.6, 1.0, size=(600, 4))

detector = HDDDM(reference_data=reference, online=True)

# Test rows accumulate in the internal buffer across calls
for i in range(0, len(stream), 100):
    result = detector.calculate(test_data=stream[i:i + 100])
    print(f"Batch {i // 100 + 1}  score={result.score}  drift={result.drift}")
```

## Result

`calculate()` returns a `ScoreDriftResult` whose `details` include `threshold`, `bins`, `n_features`, `mode`, `reference_size`, `test_size`. Use `result.score` for the raw statistic and `result.drift` for the boolean decision.
