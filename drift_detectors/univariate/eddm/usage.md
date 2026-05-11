# EDDM — usage

```python
import numpy as np
from drift_detectors import EDDM

# Synthetic error stream: ~5% error rate, then ~30% after the change.
rng = np.random.default_rng(0)
stream = np.concatenate([
    (rng.random(2000) < 0.05).astype(int),
    (rng.random(2000) < 0.30).astype(int),
])

detector = EDDM()
result = detector.calculate(stream)
print(result.drift, result.last_index, result.details)
```

For continuous residuals, threshold them first:

```python
errors = (np.abs(residuals) > tolerance).astype(int)
EDDM().calculate(errors)
```
