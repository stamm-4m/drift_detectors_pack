# EWMA for Concept Drift Detection (ECDD)
The **EWMA for Concept Drift Detection (ECDD)** detector (Ross et al., 2012) is a univariate, sequential (streaming) drift detector. It consumes any numeric value stream (for example a `0/1` error stream or continuous residuals) and reports whether a change point occurred.

An exponentially weighted moving average (EWMA) `z` of the stream is compared against a time-varying control limit `L * sigma_z`; drift fires when `|z - mean| > control_level * sigma_z`.

Unlike the distributional detectors, streaming detectors do not take a reference/test pair. They are used in two modes:

* **Batch (offline)** — pass a whole array to `calculate()`. The internal state is reset on each call and the array is scanned start to finish.
* **Online** — construct with `online=True` and feed one value (or a small batch) per call. State persists across calls, so the detector behaves as a true single-pass stream monitor.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `lambda_` | `0.2` | EWMA smoothing factor (smaller = smoother, slower) |
| `warning_level` | `3.0` | control-limit multiple for the warning zone |
| `control_level` | `4.5` | control-limit multiple for the drift decision |
| `min_instances` | `30` | warm-up length before drift can be raised |
| `online` | `False` | persist state across `calculate()` calls |

## Batch (Offline) Usage
```python
import numpy as np
from drift_detectors import ECDD

# A stationary segment followed by an abrupt change at t = 500
stream = np.concatenate([np.zeros(500), np.ones(500)])

detector = ECDD()
result = detector.calculate(stream)

print("Drift Detected :", result.drift)
print("Change Index   :", result.last_index)   # -1 if no drift
print("Details        :", result.details)       # includes the 'warning' flag
```

## Online (Streaming) Usage
```python
import numpy as np
from drift_detectors import ECDD

stream = np.concatenate([np.zeros(500), np.ones(500)])

detector = ECDD(online=True)
for t, x in enumerate(stream):
    result = detector.calculate([x])   # feed one observation at a time
    if result.details["warning"]:
        pass                            # optional early-warning zone
    if result.drift:
        print(f"drift detected at t = {t}")
        break
```

## Result

`calculate()` returns a `StreamingDriftResult` with:

* `drift` — `True` if a change point fired within this call;
* `last_index` — index (within the values passed to this call) of the last drift point, or `-1`;
* `details` — a dict with a `warning` flag, the `metric` name and the `mode` (`"online"`/`"offline"`).
