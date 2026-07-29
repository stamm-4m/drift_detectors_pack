# Drift Detection Method (DDM)
The **Drift Detection Method (DDM)** detector (Gama et al., 2004) is a univariate, sequential (streaming) drift detector. It consumes a **binary error stream** (values `> 0.5` are treated as errors, `<= 0.5` as correct) and reports whether a change point occurred.

It tracks the online error rate `p` and its standard deviation `s`, records the minimum of `p + s`, raises a *warning* when `p + s >= p_min + warning_level * s_min` and signals *drift* when `p + s >= p_min + drift_level * s_min`.

Unlike the distributional detectors, streaming detectors do not take a reference/test pair. They are used in two modes:

* **Batch (offline)** — pass a whole array to `calculate()`. The internal state is reset on each call and the array is scanned start to finish.
* **Online** — construct with `online=True` and feed one value (or a small batch) per call. State persists across calls, so the detector behaves as a true single-pass stream monitor.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `min_instances` | `30` | minimum observations before drift can be raised |
| `warning_level` | `2.0` | std multiple for the warning zone |
| `drift_level` | `4.0` | std multiple for the drift decision |
| `online` | `False` | persist state across `calculate()` calls |

## Batch (Offline) Usage
```python
import numpy as np
from drift_detectors import DDM

# A stationary segment followed by an abrupt change at t = 500
stream = np.concatenate([np.zeros(500), np.ones(500)])

detector = DDM()
result = detector.calculate(stream)

print("Drift Detected :", result.drift)
print("Change Index   :", result.last_index)   # -1 if no drift
print("Details        :", result.details)       # includes the 'warning' flag
```

## Online (Streaming) Usage
```python
import numpy as np
from drift_detectors import DDM

stream = np.concatenate([np.zeros(500), np.ones(500)])

detector = DDM(online=True)
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
DDM().calculate(errors)
```

## Result

`calculate()` returns a `StreamingDriftResult` with:

* `drift` — `True` if a change point fired within this call;
* `last_index` — index (within the values passed to this call) of the last drift point, or `-1`;
* `details` — a dict with a `warning` flag, the `metric` name and the `mode` (`"online"`/`"offline"`).
