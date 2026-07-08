# Bhattacharyya distance — usage

```python
import numpy as np
from drift_detectors import Bhattacharyya

ref  = np.random.normal(0.0, 1.0, 1000)
test = np.random.normal(0.5, 1.0, 1000)

det = Bhattacharyya()
res = det.calculate(test_data=test, reference_data=ref)
print(res.score, res.drift)
```
