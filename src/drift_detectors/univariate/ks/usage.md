# Kolmogorov–Smirnov Test (KS)
The Kolmogorov–Smirnov (KS) test is a univariate, non-parametric method for comparing two empirical distributions. It measures the maximum difference between their empirical cumulative distribution functions (ECDFs). This makes it well-suited for detecting drift between a reference dataset and a test stream or batch.

Like all detectors, this implementation supports three usage patterns:
* Stateless: Pass both reference and test data directly to calculate()
* Stateful (offline): Store reference once, pass test batch each time
* Stateful (online): Store reference and accumulate test data incrementally

## Stateless Usage
```python
import numpy as np
from drift_detectors.univariate.ks.detector import KSDetector

np.random.seed(42)
reference = np.random.normal(0, 1, size=1000)
test = np.random.normal(0.3, 1, size=1000)

detector = KSDetector(alpha=0.05)
result = detector.calculate(reference_data=reference, test_data=test)

print("KS Statistic   :", result.score)
print("Drift Detected :", result.drift)
print("Details        :", result.details)
```

## Stateful (Offline) Usage
```python
import numpy as np
from drift_detectors.univariate.ks.detector import KSDetector

np.random.seed(42)
reference = np.random.normal(0, 1, size=1000)
test = np.random.normal(0.3, 1, size=1000)

detector = KSDetector(alpha=0.05)
detector.set_reference_data(reference)

result = detector.calculate(test_data=test)

print("KS Statistic   :", result.score)
print("Drift Detected :", result.drift)
print("Reference Size :", result.details['reference_size'])
print("Test Size      :", result.details['test_size'])
```

## Stateful (Online) Usage
```python
import numpy as np
from drift_detectors.univariate.ks.detector import KSDetector

np.random.seed(42)
reference = np.random.normal(0, 1, size=1000)
stream = np.random.normal(0.3, 1, size=500)

detector = KSDetector(alpha=0.05, online=True)
detector.set_reference_data(reference)

# Feed data in small batches
for i in range(0, len(stream), 50):
    batch = stream[i:i+50]
    result = detector.calculate(test_data=batch)

    print(f"Batch {i//50 + 1}")
    print("  KS Statistic   :", result.score)
    print("  Drift Detected :", result.drift)
    print("  Test Size      :", result.details['test_size'])
```