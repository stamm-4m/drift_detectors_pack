# Reactive Drift Detection Method (RDDM)
The **Reactive Drift Detection Method (RDDM)** detector (Barros et al., 2017) is a univariate, sequential (streaming) drift detector. It consumes a **binary error stream** (values `> 0.5` are treated as errors) and reports whether a change point occurred.

A more reactive variant of DDM: it uses more sensitive thresholds and a `max_concept_size`, so that if a concept stays stable for too long the running `(p_min, s_min)` baseline is recomputed and the detector stays responsive to slow drift.

Unlike the distributional detectors, streaming detectors do not take a reference/test pair. They are used in two modes:

* **Batch (offline)** — pass a whole array to `calculate()`. The internal state is reset on each call and the array is scanned start to finish.
* **Online** — construct with `online=True` and feed one value (or a small batch) per call. State persists across calls, so the detector behaves as a true single-pass stream monitor.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `min_instances` | `129` | minimum observations before drift can be raised |
| `warning_level` | `3.0` | std multiple for the warning zone |
| `drift_level` | `4.0` | std multiple for the drift decision |
| `max_concept_size` | `40000` | stable-run length after which the baseline is refreshed |
| `online` | `False` | persist state across `calculate()` calls |

## Batch (Offline) Usage
```python
import numpy as np
from drift_detectors import RDDM

# A stationary segment followed by an abrupt change at t = 500
stream = np.concatenate([np.zeros(500), np.ones(500)])

detector = RDDM()
result = detector.calculate(stream)

print("Drift Detected :", result.drift)
print("Change Index   :", result.last_index)   # -1 if no drift
print("Details        :", result.details)       # includes the 'warning' flag
```

## Online (Streaming) Usage
```python
import numpy as np
from drift_detectors import RDDM

stream = np.concatenate([np.zeros(500), np.ones(500)])

detector = RDDM(online=True)
for t, x in enumerate(stream):
    result = detector.calculate([x])   # feed one observation at a time
    if result.details["warning"]:
        pass                            # optional early-warning zone
    if result.drift:
        print(f"drift detected at t = {t}")
        break
```

## Feeding residuals instead of a 0/1 stream

This detector expects a **binary error stream**. To monitor a continuous model error, threshold it first:

```python
errors = (np.abs(residuals) > tolerance).astype(int)
RDDM().calculate(errors)
```

## Result

`calculate()` returns a `StreamingDriftResult` with:

* `drift` — `True` if a change point fired within this call;
* `last_index` — index (within the values passed to this call) of the last drift point, or `-1`;
* `details` — a dict with a `warning` flag, the `metric` name and the `mode` (`"online"`/`"offline"`).
