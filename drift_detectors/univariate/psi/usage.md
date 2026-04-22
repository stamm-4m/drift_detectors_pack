# Population Stability Index (PSI)
The Population Stability Index (PSI) is a univariate drift detection method that quantifies the shift in distribution between two datasets — typically a reference dataset (e.g., training data) and a new test dataset (e.g., live data). It’s commonly used in production monitoring to detect feature-level distributional changes over time.

Like all detectors, this detector supports three usage patterns:
* Stateless (explicit reference and test each time)
* Stateful (offline): reference stored at init or via set_reference_data
* Stateful (online): reference is stored, and test data is accumulated over time

## Stateless Usage
```python
import numpy as np
from drift_detectors.univariate.psi.detector import PSI

np.random.seed(42)
reference = np.random.normal(0.0, 1.0, size=1000)
test = np.random.normal(0.3, 1.0, size=1000)

detector = PSI()
result = detector.calculate(reference_data=reference, test_data=test)

print("PSI Score      :", result.score)
print("Drift Detected :", result.drift)
print("Details        :", result.details)
```

## Stateful (Offline) Usage
```python
import numpy as np
from drift_detectors.univariate.psi.detector import PSI

np.random.seed(42)
reference = np.random.normal(0.0, 1.0, size=1000)
test = np.random.normal(0.3, 1.0, size=1000)

# Reference set at init
detector = PSI(reference_data=reference)  
# Alternatively: detector.set_reference_data(reference)

result = detector.calculate(test_data=test)

print("PSI Score      :", result.score)
print("Drift Detected :", result.drift)
```

## Stateful (Online) Usage
```python
import numpy as np
from drift_detectors.univariate.psi.detector import PSI

np.random.seed(42)
reference = np.random.normal(0.0, 1.0, size=1000)
# Simulated stream data
stream = np.random.normal(0.3, 1.0, size=500)

detector = PSI(reference_data=reference, online=True)

# Incrementally add data to test window
for i in range(0, len(stream), 50):
    batch = stream[i:i+50]
    result = detector.calculate(test_data=batch)

    print(f"Batch {i//50 + 1}")
    print("  PSI Score     :", result.score)
    print("  Drift Detected:", result.drift)
    print("  Test Size     :", result.details['test_size'])
```