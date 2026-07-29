# Classifier Two-Sample Test (C2ST)
The **Classifier Two-Sample Test (C2ST)** detector (Lopez-Paz & Oquab, 2017) is a *multivariate*, batch drift detector. It trains a k-nearest-neighbour classifier to tell reference samples (label 0) from test samples (label 1) and measures its held-out accuracy. Inputs are 2-D arrays shaped `(n_samples, n_features)`: a *reference* set and a *test* set.

Classes are balanced internally, so the chance level is exactly `0.5`: a held-out accuracy near `0.5` means no drift, while a higher accuracy means the two distributions are separable. Drift is decided by a one-sided binomial test on the held-out accuracy.

Like all detectors in the package, it supports three usage patterns:

* **Stateless** — pass both reference and test data to `calculate()` each time.
* **Stateful (offline)** — store the reference once, then pass each test batch.
* **Stateful (online)** — store the reference and accumulate test rows incrementally across calls.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `k_neighbors` | `5` | neighbours used by the kNN classifier |
| `test_size` | `0.3` | fraction held out to measure accuracy |
| `alpha` | `0.05` | significance level for the binomial test |
| `max_samples` | `1000` | cap per group (larger groups are subsampled) |
| `seed` | `0` | RNG seed for the split/subsampling |
| `online` | `False` | accumulate test rows across calls |

## Stateless Usage
```python
import numpy as np
from drift_detectors import C2ST

rng = np.random.default_rng(42)
reference = rng.normal(0.0, 1.0, size=(500, 4))
test = rng.normal(0.6, 1.0, size=(400, 4))

detector = C2ST()
result = detector.calculate(test_data=test, reference_data=reference)

print("Score          :", result.score)
print("Drift Detected :", result.drift)
print("Details        :", result.details)
```

## Stateful (Offline) Usage
```python
import numpy as np
from drift_detectors import C2ST

rng = np.random.default_rng(42)
reference = rng.normal(0.0, 1.0, size=(500, 4))
test = rng.normal(0.6, 1.0, size=(400, 4))

detector = C2ST(reference_data=reference)   # or detector.set_reference_data(reference)
result = detector.calculate(test_data=test)

print("Score          :", result.score)
print("Drift Detected :", result.drift)
```

## Stateful (Online) Usage
```python
import numpy as np
from drift_detectors import C2ST

rng = np.random.default_rng(42)
reference = rng.normal(0.0, 1.0, size=(500, 4))
stream = rng.normal(0.6, 1.0, size=(600, 4))

detector = C2ST(reference_data=reference, online=True)

# Test rows accumulate in the internal buffer across calls
for i in range(0, len(stream), 100):
    result = detector.calculate(test_data=stream[i:i + 100])
    print(f"Batch {i // 100 + 1}  score={result.score}  drift={result.drift}")
```

## Result

`calculate()` returns a `ScoreDriftResult` whose `details` include `p_value`, the held-out accuracy, `alpha`, `mode` and sizes. Use `result.score` for the raw statistic and `result.drift` for the boolean decision.
