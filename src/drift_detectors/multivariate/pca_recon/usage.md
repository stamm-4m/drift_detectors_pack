# PCA reconstruction-error drift (PCAReconstruction) — usage

```python
import numpy as np
from drift_detectors import PCAReconstruction

ref  = np.random.normal(0.0, 1.0, size=(400, 4))
test = np.random.normal(0.5, 1.0, size=(300, 4))
print(PCAReconstruction().calculate(test_data=test, reference_data=ref))
```
