# Multivariate KS via Random Projections
The **Multivariate KS via Random Projections** detector (Cuesta-Albertos et al., 2007) is a *multivariate*, batch drift detector. It projects the reference and test samples onto `n_projections` random unit directions and runs a univariate two-sample KS test on each projection. Inputs are 2-D arrays shaped `(n_samples, n_features)`: a *reference* set and a *test* set.

The score is the **fraction of projections** that show significant drift, so it lies in `[0, 1]`. Random projections give a cheap, distribution-free way to detect multivariate shift that pure marginal tests would miss.

Like all detectors in the package, it supports three usage patterns:

* **Stateless** — pass both reference and test data to `calculate()` each time.
* **Stateful (offline)** — store the reference once, then pass each test batch.
* **Stateful (online)** — store the reference and accumulate test rows incrementally across calls.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `n_projections` | `50` | number of random directions tested |
| `alpha` | `0.01` | per-projection KS significance level |
| `score_threshold` | `0.1` | fraction of significant projections above which drift is flagged |
| `seed` | `0` | RNG seed for the projections (reproducibility) |
| `online` | `False` | accumulate test rows across calls |

## Stateless Usage
```python
import numpy as np
from drift_detectors import MultivariateKS

rng = np.random.default_rng(42)
reference = rng.normal(0.0, 1.0, size=(500, 4))
test = rng.normal(0.6, 1.0, size=(400, 4))

detector = MultivariateKS()
result = detector.calculate(test_data=test, reference_data=reference)

print("Score          :", result.score)
print("Drift Detected :", result.drift)
print("Details        :", result.details)
```

## Stateful (Offline) Usage
```python
import numpy as np
from drift_detectors import MultivariateKS

rng = np.random.default_rng(42)
reference = rng.normal(0.0, 1.0, size=(500, 4))
test = rng.normal(0.6, 1.0, size=(400, 4))

detector = MultivariateKS(reference_data=reference)   # or detector.set_reference_data(reference)
result = detector.calculate(test_data=test)

print("Score          :", result.score)
print("Drift Detected :", result.drift)
```

## Stateful (Online) Usage
```python
import numpy as np
from drift_detectors import MultivariateKS

rng = np.random.default_rng(42)
reference = rng.normal(0.0, 1.0, size=(500, 4))
stream = rng.normal(0.6, 1.0, size=(600, 4))

detector = MultivariateKS(reference_data=reference, online=True)

# Test rows accumulate in the internal buffer across calls
for i in range(0, len(stream), 100):
    result = detector.calculate(test_data=stream[i:i + 100])
    print(f"Batch {i // 100 + 1}  score={result.score}  drift={result.drift}")
```

## Result

`calculate()` returns a `ScoreDriftResult` whose `details` include `score_threshold`, `n_projections`, `alpha`, `mode` and sizes. Use `result.score` for the raw statistic and `result.drift` for the boolean decision.
