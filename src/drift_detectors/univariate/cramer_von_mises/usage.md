# Cramer-von Mises (2-sample) — usage

```python
import numpy as np
from drift_detectors import CramerVonMises

ref  = np.random.normal(0.0, 1.0, 1000)
test = np.random.normal(0.5, 1.0, 1000)

det = CramerVonMises()
res = det.calculate(test_data=test, reference_data=ref)
print(res.score, res.drift)
```
