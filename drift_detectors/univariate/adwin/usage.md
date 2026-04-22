# ADaptive WINdowing (ADWIN)
ADWIN is a univariate drift detection algorithm designed for streaming scenarios. It maintains a dynamically sized window over incoming data and uses statistical tests (Hoeffding bounds) to detect changes in the average value — signaling drift. It is well-suited to continuous, time-ordered data and supports both batch and streaming analysis.

Like all detectors, this detector supports three usage patterns:
* Stateless (explicit test data each time)
* Stateful (offline): ADWIN is reset for each call
* Stateful (online): the ADWIN instance is preserved and accumulates drift state across calls

## Stateless Usage
```python
import numpy as np
from drift_detectors.univariate.adwin.detector import Adwin

np.random.seed(42)
# Simulate full stream with a distributional shift
stream = np.concatenate([
    np.random.normal(0.0, 1.0, size=500),
    np.random.normal(2.0, 1.0, size=500)
])

detector = Adwin()  # Stateless by default
result = detector.calculate(test_data=stream)

print("Drift Detected :", result.drift)
print("Last Drift Index:", result.last_index)
print("Details         :", result.details)
```

## Stateful (Offline) Usage
```python
import numpy as np
from drift_detectors.univariate.adwin.detector import Adwin

np.random.seed(42)
stream = np.concatenate([
    np.random.normal(0.0, 1.0, size=500),
    np.random.normal(2.0, 1.0, size=500)
])

detector = Adwin(online=False)  # Explicit but redundant
result = detector.calculate(test_data=stream)

print("Drift Detected :", result.drift)
print("Last Drift Index:", result.last_index)
```

## Stateful (Online) Usage
```python
import numpy as np
from drift_detectors.univariate.adwin.detector import Adwin

np.random.seed(42)
stream = np.concatenate([
    np.random.normal(0.0, 1.0, size=500),
    np.random.normal(2.0, 1.0, size=500)
])

detector = Adwin(online=True)

# Simulate a stream: feed data in batches
for i in range(0, len(stream), 50):
    batch = stream[i:i+50]
    result = detector.calculate(test_data=batch)

    print(f"Batch {i//50 + 1}")
    print("  Drift Detected :", result.drift)
    print("  Last Index     :", result.last_index)
    print("  Details        :", result.details)
```