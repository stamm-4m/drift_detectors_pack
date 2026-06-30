# HDDM-A — usage

```python
import numpy as np
from drift_detectors import HDDM_A

stream = np.concatenate([
    np.random.normal(0.0, 1.0, 800),
    np.random.normal(2.5, 1.0, 800),
])

detector = HDDM_A(drift_confidence=0.001, warning_confidence=0.005)
result = detector.calculate(stream)
print(result.drift, result.last_index, result.details)
```

In streaming mode, set `online=True` and call once per observation:

```python
det = HDDM_A(online=True)
for x in stream:
    res = det.calculate([x])
    if res.drift:
        print("drift!", res.last_index)
        break
```
