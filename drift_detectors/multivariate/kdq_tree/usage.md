# KDQ-tree Drift Detection (KDQTree)
KDQTree is a multivariate drift detection method that compares local neighborhood distributions between reference and test datasets using k-nearest neighbors and feature-wise Kolmogorov–Smirnov (KS) tests. Drift is scored by the proportion of neighborhoods where the average KS p-value falls below a significance threshold.

Like all detectors, this implementation supports three usage patterns:
* Stateless: Pass both reference and test data directly to calculate()
* Stateful (offline): Store reference once, pass test batch each time
* Stateful (online): Store reference and accumulate test data incrementally

## Stateless Usage
```python
import numpy as np
from drift_detectors.multivariate.kdqtree.detector import KDQTree

np.random.seed(42)
reference = np.random.normal(0, 1, size=(300, 5))
test = np.random.normal(0.5, 1, size=(300, 5))

detector = KDQTree(k_neighbors=25, alpha=0.05)
result = detector.calculate(reference_data=reference, test_data=test)

print("Drift Score     :", result.score)
print("Drift Detected  :", result.drift)
print("Details         :", result.details)
```

## Stateful (Offline) Usage
```python
import numpy as np
from drift_detectors.multivariate.kdqtree.detector import KDQTree

np.random.seed(42)
reference = np.random.normal(0, 1, size=(300, 5))
test = np.random.normal(0.5, 1, size=(300, 5))

detector = KDQTree(k_neighbors=25, alpha=0.05)
detector.set_reference_data(reference)

result = detector.calculate(test_data=test)

print("Drift Score     :", result.score)
print("Drift Detected  :", result.drift)
print("Reference Size  :", result.details['reference_size'])
print("Test Size       :", result.details['test_size'])
```

## Stateful (Online) Usage
```python
import numpy as np
from drift_detectors.multivariate.kdqtree.detector import KDQTree

np.random.seed(42)
reference = np.random.normal(0, 1, size=(300, 5))
stream = np.random.normal(0.5, 1, size=(500, 5))

detector = KDQTree(k_neighbors=25, alpha=0.05, online=True)
detector.set_reference_data(reference)

# Simulate streaming test data
for i in range(0, len(stream), 50):
    batch = stream[i:i + 50]
    result = detector.calculate(test_data=batch)

    print(f"Batch {i//50 + 1}")
    print("  Drift Score     :", result.score)
    print("  Drift Detected  :", result.drift)
    print("  Test Size       :", result.details['test_size'])
```