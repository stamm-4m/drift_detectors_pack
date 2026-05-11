# Page-Hinkley — usage

```python
import numpy as np
from drift_detectors import PageHinkley

# stationary segment + abrupt mean shift
stream = np.concatenate([
    np.random.normal(0.0, 1.0, 500),
    np.random.normal(2.0, 1.0, 500),
])

ph = PageHinkley(delta=0.005, lambda_=20.0)
result = ph.calculate(stream)
print(result.drift, result.last_index, result.details)
```

For incremental monitoring, set `online=True` and feed one (or a small batch
of) observation(s) per call:

```python
ph = PageHinkley(online=True, delta=0.005, lambda_=20.0)
for value in stream:
    result = ph.calculate([value])
    if result.drift:
        print("drift detected!")
        break
```
