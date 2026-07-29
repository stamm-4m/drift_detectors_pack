# Cumulative Sum (CUSUM)
The **Cumulative Sum (CUSUM)** detector (Page, 1954) is a univariate, sequential (streaming) drift detector. It consumes a stream of numeric values (typically model residuals or any monitored signal) and reports whether a change point occurred.

A baseline mean and standard deviation are estimated from the first `min_instances` observations and then frozen. Each later value is standardised and its deviation beyond a slack `k` is accumulated into a positive and a negative sum; drift fires when either sum exceeds the threshold `h`, after which the detector resets.

Unlike the distributional detectors, streaming detectors do not take a reference/test pair. They are used in two modes:

* **Batch (offline)** — pass a whole array to `calculate()`. The internal state is reset on each call and the array is scanned start to finish.
* **Online** — construct with `online=True` and feed one value (or a small batch) per call. State persists across calls, so the detector behaves as a true single-pass stream monitor.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `k` | `0.5` | slack / reference value — deviations smaller than `k` std are ignored |
| `h` | `12.0` | decision threshold on the cumulative sum (larger = fewer false alarms) |
| `min_instances` | `30` | warm-up length used to freeze the baseline mean/std |
| `online` | `False` | persist state across `calculate()` calls |

## Batch (Offline) Usage
```python
import numpy as np
from drift_detectors import CUSUM

# A stationary segment followed by an abrupt change at t = 500
stream = np.concatenate([np.zeros(500), np.ones(500)])

detector = CUSUM()
result = detector.calculate(stream)

print("Drift Detected :", result.drift)
print("Change Index   :", result.last_index)   # -1 if no drift
print("Details        :", result.details)       # includes the 'warning' flag
```

## Online (Streaming) Usage
```python
import numpy as np
from drift_detectors import CUSUM

stream = np.concatenate([np.zeros(500), np.ones(500)])

detector = CUSUM(online=True)
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
