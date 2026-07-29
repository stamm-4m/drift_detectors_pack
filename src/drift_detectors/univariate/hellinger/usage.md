# Hellinger Distance
The **Hellinger Distance** detector is a univariate, distributional drift detector. It measures the Hellinger distance between the reference and test histograms. It compares a *reference* dataset (for example the training distribution of one feature) against a *test* dataset (recent live data) and flags drift when the distance exceeds a threshold.

The score is bounded in `[0, 1]`: `0` when the histograms coincide and `1` when they have disjoint support.

Like all detectors in the package, it supports three usage patterns:

* **Stateless** — pass both reference and test data to `calculate()` each time.
* **Stateful (offline)** — store the reference once, then pass each test batch.
* **Stateful (online)** — store the reference and accumulate test data incrementally across calls.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `threshold` | `0.1` | drift is flagged when `score >= threshold` |
| `bins` | `20` | number of histogram bins used to estimate the two distributions |
| `epsilon` | `1e-8` | smoothing added to empty bins to avoid division by zero |
| `online` | `False` | when `True`, test data is accumulated across `calculate()` calls |

## Stateless Usage
```python
import numpy as np
from drift_detectors import HellingerDistance

np.random.seed(42)
reference = np.random.normal(0.0, 1.0, size=1000)
test = np.random.normal(0.5, 1.0, size=1000)

detector = HellingerDistance()
result = detector.calculate(test_data=test, reference_data=reference)

print("Score          :", result.score)
print("Drift Detected :", result.drift)
print("Details        :", result.details)
```

## Stateful (Offline) Usage
```python
import numpy as np
from drift_detectors import HellingerDistance

np.random.seed(42)
reference = np.random.normal(0.0, 1.0, size=1000)
test = np.random.normal(0.5, 1.0, size=1000)

# Reference stored once, at construction (or via detector.set_reference_data(reference))
detector = HellingerDistance(reference_data=reference)
result = detector.calculate(test_data=test)

print("Score          :", result.score)
print("Drift Detected :", result.drift)
print("Reference Size :", result.details["reference_size"])
print("Test Size      :", result.details["test_size"])
```

## Stateful (Online) Usage
```python
import numpy as np
from drift_detectors import HellingerDistance

np.random.seed(42)
reference = np.random.normal(0.0, 1.0, size=1000)
stream = np.random.normal(0.5, 1.0, size=500)

detector = HellingerDistance(reference_data=reference, online=True)

# Test data accumulates in the internal buffer across calls
for i in range(0, len(stream), 50):
    batch = stream[i:i + 50]
    result = detector.calculate(test_data=batch)
    print(f"Batch {i // 50 + 1}  score={result.score:.4f}  drift={result.drift}"
          f"  test_size={result.details['test_size']}")
```

## Result

`calculate()` returns a `ScoreDriftResult` with:

* `score` — the Hellinger Distance value (see interpretation above);
* `drift` — `True` when `score >= threshold`;
* `details` — a dict with `threshold`, `bins`, `metric`, `mode` (`"online"`/`"offline"`), `reference_size` and `test_size`.

You can override the decision threshold per call, e.g. `detector.calculate(test_data=test, threshold=0.15)`.
