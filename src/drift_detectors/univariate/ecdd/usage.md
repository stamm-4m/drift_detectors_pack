# EWMA for Concept Drift Detection (ECDD) — usage

```python
import numpy as np
from drift_detectors import ECDD

det = ECDD(online=True)
stream = np.concatenate([np.zeros(500), np.ones(500)])  # a change at t=500
for x in stream:
    if det.calculate([x]).drift:
        print("drift")
        break
```
