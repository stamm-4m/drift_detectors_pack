# PCA-based Change Detection (PCA-CD)
PCA-CD is a multivariate drift detection method that leverages Principal Component Analysis to monitor structural changes in the data. It projects both reference and test datasets into a reduced-dimensional PCA space and compares them using:
* CSD: cumulative squared differences of component means
* KL divergence: symmetric divergence of component-wise variances

It is well-suited for high-dimensional or correlated data.
Like all detectors, this implementation supports three usage patterns:
* Stateless: Pass both reference and test data directly to calculate()
* Stateful (offline): Store reference once, pass test batch each time
* Stateful (online): Store reference and accumulate test data incrementally

## Stateless Usage
```python
import numpy as np
from drift_detectors.multivariate.pca_cd.detector import PCA_CD

np.random.seed(42)
reference = np.random.normal(0, 1, size=(200, 5))
test = np.random.normal(0.5, 1, size=(200, 5))

detector = PCA_CD(n_components=2)
result = detector.calculate(reference_data=reference, test_data=test)

print("Drift Score     :", result.score)
print("Drift Detected  :", result.drift)
print("CSD             :", result.details['csd'])
print("KL Divergence   :", result.details['kl'])
```

## Stateful (Offline) Usage
```python
import numpy as np
from drift_detectors.multivariate.pca_cd.detector import PCA_CD

np.random.seed(42)
reference = np.random.normal(0, 1, size=(200, 5))
test = np.random.normal(0.5, 1, size=(200, 5))

detector = PCA_CD(n_components=2)
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
from drift_detectors.multivariate.pca_cd.detector import PCA_CD

np.random.seed(42)
reference = np.random.normal(0, 1, size=(200, 5))
stream = np.random.normal(0.5, 1, size=(500, 5))

detector = PCA_CD(n_components=2, online=True)
detector.set_reference_data(reference)

# Simulate incoming stream
for i in range(0, len(stream), 50):
    batch = stream[i:i + 50]
    result = detector.calculate(test_data=batch)

    print(f"Batch {i//50 + 1}")
    print("  Drift Score     :", result.score)
    print("  Drift Detected  :", result.drift)
    print("  CSD             :", result.details['csd'])
    print("  KL Divergence   :", result.details['kl'])
    print("  Test Size       :", result.details['test_size'])
```