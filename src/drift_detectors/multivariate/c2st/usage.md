# Classifier Two-Sample Test (C2ST) — usage

```python
import numpy as np
from drift_detectors import C2ST

ref  = np.random.normal(0.0, 1.0, size=(500, 4))
test = np.random.normal(0.6, 1.0, size=(500, 4))

det = C2ST(k_neighbors=5, alpha=0.05)
res = det.calculate(test_data=test, reference_data=ref)
print(res.score, res.drift, res.details["p_value"])
```

`score` is the held-out accuracy of a kNN classifier trained to tell the
reference and test samples apart: ~0.5 means no drift, higher means the two
distributions are separable. Classes are balanced internally so the chance
level is exactly 0.5; `drift` is decided by a one-sided binomial test.
