# Hellinger Distance Drift Detection Method (HDDDM) — usage

```python
import numpy as np
from drift_detectors import HDDDM

ref  = np.random.normal(0.0, 1.0, size=(400, 4))
test = np.random.normal(0.5, 1.0, size=(300, 4))
print(HDDDM().calculate(test_data=test, reference_data=ref))
```
