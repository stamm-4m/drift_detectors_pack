# Maximum Mean Discrepancy (MMD)
The Maximum Mean Discrepancy (MMD) test is a kernel-based method for detecting multivariate distribution shift. It compares the mean embeddings of reference and test datasets in a Reproducing Kernel Hilbert Space (RKHS), using the RBF kernel. This implementation supports both offline and online (incremental) modes.

## Stateless Usage
```python
import numpy as np
from drift_detectors.multivariate.mmd.detector import MMDDetector

np.random.seed(42)
reference = np.random.normal(0, 1, size=(300, 5))
test = np.random.normal(0.4, 1, size=(300, 5))

detector = MMDDetector(gamma=1.0, threshold=0.05)
result = detector.calculate(reference_data=reference, test_data=test)

print("MMD Score       :", result.score)
print("Drift Detected  :", result.drift)
print("Details         :", result.details)
```

## Stateful (Offline) Usage
```python
import numpy as np
from drift_detectors.multivariate.mmd.detector import MMDDetector

np.random.seed(42)
reference = np.random.normal(0, 1, size=(300, 5))
test = np.random.normal(0.4, 1, size=(300, 5))

detector = MMDDetector(gamma=1.0, threshold=0.05)
detector.set_reference_data(reference)

result = detector.calculate(test_data=test)

print("MMD Score       :", result.score)
print("Drift Detected  :", result.drift)
print("Reference Size  :", result.details['reference_size'])
print("Test Size       :", result.details['test_size'])
```

## Stateful (Online) Usage
```python
import numpy as np
from drift_detectors.multivariate.mmd.detector import MMDDetector

np.random.seed(42)
reference = np.random.normal(0, 1, size=(300, 5))
stream = np.random.normal(0.4, 1, size=(600, 5))

detector = MMDDetector(gamma=1.0, threshold=0.05, online=True)
detector.set_reference_data(reference)

for i in range(0, len(stream), 100):
    batch = stream[i:i + 100]
    result = detector.calculate(test_data=batch)

    print(f"Batch {i//100 + 1}")
    print("  MMD Score       :", result.score)
    print("  Drift Detected  :", result.drift)
    print("  Test Size       :", result.details['test_size'])
```