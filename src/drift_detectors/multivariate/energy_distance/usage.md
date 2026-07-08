# Energy Distance — usage

```python
import numpy as np
from drift_detectors import EnergyDistance

ref  = np.random.normal(0.0, 1.0, size=(400, 4))
test = np.random.normal(0.5, 1.0, size=(300, 4))

det = EnergyDistance(alpha=0.05, n_permutations=199)
res = det.calculate(test_data=test, reference_data=ref)
print(res.score, res.drift, res.details["p_value"])
```

The score is the energy-distance statistic; `drift` is decided by a
label-permutation test (`p_value < alpha`). Large groups are subsampled to
`max_samples` per group so the permutation test stays fast; set `seed` for
reproducible results.
